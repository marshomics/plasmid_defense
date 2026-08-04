"""Thin subprocess-based wrapper for calling R scripts.

rpy2 is stable enough for interactive use but brittle in long-running batch
pipelines (shared-library version mismatches, thread-safety surprises inside
parallel joblib workers). We instead write inputs as TSVs to a scratch
directory, invoke an R script via ``Rscript``, read the TSV it writes back,
and return a DataFrame. Slower per call, orders of magnitude more reliable.

Every R script in ``r_scripts/`` follows the same contract:

    Rscript <script>.R <tree_path> <data_tsv> <args_json> <out_tsv>

where ``args_json`` holds the list of response/predictor columns, evolutionary
model choice, and any method-specific parameters.
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


R_SCRIPT_DIR = Path(__file__).parent / "r_scripts"


@dataclass
class RCallResult:
    """Container for the result of an R subprocess call."""
    dataframe: Optional[pd.DataFrame]
    stdout: str
    stderr: str
    returncode: int
    error: Optional[str] = None
    # Path the R script wrote its main table to. Exposed so callers can pick up
    # companion sidecar files (e.g. phylo_residuals.R writes
    # "<out_path>.status.tsv" recording which predictors were actually
    # decorrelated rather than silently falling back to raw values).
    output_path: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.dataframe is not None


class SharedFrame:
    """A phylo_data frame serialised ONCE and reused by many R calls.

    Why this exists
    ---------------
    Every ``call_r_script`` invocation used to write the entire species x
    feature frame to disk. At 39,681 species x ~460 columns that is ~40 MB per
    call, and a full pipeline run makes ~4,800 R calls -- roughly 190 GB of
    temporary writes, with up to 20 workers streaming concurrently onto a
    shared scratch filesystem.

    That volume is not the wall-clock bottleneck (it is a few hours), but it is
    almost certainly the cause of the SIGBUS ("Bus error") crashes that killed
    LOCO and the misclassification Monte Carlo. A bus error on a write path is
    the classic signature of filesystem pressure or an mmap'd region becoming
    invalid, and the optimisation log already records intermittent bus errors
    traced to ``/tmp`` fill events.

    Stages that call R many times over the SAME underlying data now write it
    once and pass:

      * ``row_filter_file``  -- a one-column list of tips to keep, or
      * ``override_file``    -- a small keyed table of columns to replace.

    Both are tiny (a few hundred KB), so per-call I/O drops by three orders of
    magnitude. This is a reliability fix first and a modest speed fix second.
    """

    def __init__(self, frame: pd.DataFrame, path: Path, key: str = "tip"):
        self.path = Path(path)
        self.key = key
        self.columns = list(frame.columns)
        self.n_rows = len(frame)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            frame.to_csv(self.path, sep="\t", index=False)

    def size_mb(self) -> float:
        return self.path.stat().st_size / 1e6 if self.path.exists() else 0.0


def write_shared_frame(frame: pd.DataFrame, workdir: Path, name: str,
                       logger: logging.Logger, key: str = "tip") -> SharedFrame:
    """Serialise ``frame`` once for reuse across many R calls."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    sf = SharedFrame(frame, workdir / f"shared_{name}.tsv", key=key)
    logger.info(
        f"Shared frame '{name}': {sf.n_rows:,} rows x {len(sf.columns)} cols "
        f"({sf.size_mb():.0f} MB) written once and reused by every R call in "
        f"this stage")
    return sf


def call_r_script(script_name: str, *, tree_path: str,
                  data: Optional[pd.DataFrame] = None,
                  args: Dict[str, Any], logger: logging.Logger,
                  r_executable: str = "Rscript",
                  workdir: Optional[Path] = None,
                  timeout: Optional[float] = None,
                  shared: Optional[SharedFrame] = None,
                  keep_tips: Optional[Any] = None,
                  overrides: Optional[pd.DataFrame] = None,
                  max_retries: int = 1,
                  env: Optional[Dict[str, str]] = None) -> RCallResult:
    """Invoke an R script with the standard (tree, data, args, out) signature.

    Two ways to supply the data:

      * ``data=`` — serialise this frame for this call (original behaviour).
      * ``shared=`` — reuse an already-written :class:`SharedFrame`, optionally
        narrowed by ``keep_tips`` (an iterable of tip labels) and/or patched by
        ``overrides`` (a small frame keyed on the shared frame's key column
        whose columns replace the shared frame's). Both are written as small
        side files, so the 40 MB frame is not re-serialised.

    ``max_retries`` re-runs a call that died from a *signal* (negative return
    code — SIGBUS, SIGKILL, SIGSEGV) rather than from an R-level error. Signal
    deaths at this scale are usually transient resource contention; an R error
    is deterministic and is never retried.
    """
    script_path = R_SCRIPT_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"R script not found: {script_path}")

    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="defense_r_"))
    workdir.mkdir(parents=True, exist_ok=True)
    out_tsv = workdir / "out.tsv"
    args_json = workdir / "args.json"

    args = dict(args)
    if shared is not None:
        data_tsv = shared.path
        if keep_tips is not None:
            keep_path = workdir / "keep_tips.tsv"
            pd.DataFrame({shared.key: list(keep_tips)}).to_csv(
                keep_path, sep="\t", index=False)
            args["row_filter_file"] = str(keep_path)
        if overrides is not None and not overrides.empty:
            ov_path = workdir / "overrides.tsv"
            overrides.to_csv(ov_path, sep="\t", index=False)
            args["override_file"] = str(ov_path)
        args.setdefault("shared_key", shared.key)
    elif data is not None:
        data_tsv = workdir / "data.tsv"
        data.to_csv(data_tsv, sep="\t", index=False)
    else:
        raise ValueError("call_r_script needs either data= or shared=")

    with open(args_json, "w") as fh:
        json.dump(args, fh)

    cmd = [r_executable, "--vanilla", str(script_path),
           str(tree_path), str(data_tsv), str(args_json), str(out_tsv)]
    logger.debug(f"R call: {' '.join(cmd)}")

    run_env = None
    if env:
        run_env = dict(os.environ)
        run_env.update({k: str(v) for k, v in env.items()})

    attempt, proc = 0, None
    while True:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout, env=run_env)
        except subprocess.TimeoutExpired as e:
            return RCallResult(None, e.stdout or "", e.stderr or "", -1,
                               error=f"Timed out after {timeout}s")
        except FileNotFoundError as e:
            return RCallResult(None, "", "", -1,
                               error=f"Rscript not found: {e}. Install R and set "
                               "config.r_executable if it's not on PATH.")
        # Negative return code == killed by a signal. SIGBUS (-7) and SIGKILL
        # (-9) at this scale are transient resource contention, not a bug in
        # the model, so a bounded retry is worth it. R-level errors return
        # positive codes and are deterministic — never retried.
        if proc.returncode >= 0 or attempt >= max_retries:
            break
        attempt += 1
        logger.warning(
            f"R script {script_name} killed by signal {-proc.returncode} "
            f"(attempt {attempt}/{max_retries}); retrying after backoff")
        time.sleep(min(60, 5 * (2 ** attempt)))

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if proc.returncode != 0 or not out_tsv.exists():
        truncated = stderr[:2000] + ("…" if len(stderr) > 2000 else "")
        signal_note = ""
        if proc.returncode < 0:
            signal_note = (f" Process was killed by signal {-proc.returncode}. "
                           f"If this is SIGBUS(7) or SIGKILL(9), suspect "
                           f"scratch-filesystem pressure or the h_vmem cap "
                           f"rather than a modelling error.")
        logger.warning(
            f"R script {script_name} failed (rc={proc.returncode}).{signal_note}\n"
            f"stderr:\n{truncated}"
        )
        return RCallResult(None, stdout, stderr, proc.returncode,
                           error=stderr.strip().splitlines()[-1] if stderr.strip() else None)

    try:
        df = pd.read_csv(out_tsv, sep="\t")
    except Exception as e:
        return RCallResult(None, stdout, stderr, proc.returncode,
                           error=f"Unable to read R output TSV: {e}")

    return RCallResult(df, stdout, stderr, proc.returncode,
                       output_path=out_tsv)


def ensure_r_packages(r_executable: str, packages: list, logger: logging.Logger) -> None:
    """Check that the listed R packages are installed. Logs a warning if any
    is missing; downstream R scripts will error out with a specific message.

    Rscript forwards every argument after ``-e <script>`` to
    ``commandArgs(trailingOnly=TRUE)`` verbatim, including the legacy
    ``--args`` separator that is only meaningful to ``R CMD BATCH`` /
    ``R -e``. Passing it here would make the check_script report
    ``--args`` as a missing package. So we skip the separator entirely
    and instead hand the package list in as a quoted R vector inside
    the script itself.
    """
    pkg_vector = ", ".join(f'"{p}"' for p in packages)
    check_script = (
        f"cat(setdiff(c({pkg_vector}), rownames(installed.packages())), sep='\\n')"
    )
    try:
        proc = subprocess.run(
            [r_executable, "-e", check_script],
            capture_output=True, text=True, timeout=60,
        )
        missing = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if missing:
            logger.warning(
                f"Missing R packages: {missing}. Install with "
                f"install.packages({missing!r}) in R. Tier 2 / Tier 3 tests "
                "that require these packages will be skipped."
            )
        else:
            logger.info(f"All required R packages present: {packages}")
    except Exception as e:
        logger.warning(f"Could not verify R packages ({e}); continuing")
