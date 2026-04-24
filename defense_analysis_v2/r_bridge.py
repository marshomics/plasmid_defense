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

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.dataframe is not None


def call_r_script(script_name: str, *, tree_path: str, data: pd.DataFrame,
                  args: Dict[str, Any], logger: logging.Logger,
                  r_executable: str = "Rscript",
                  workdir: Optional[Path] = None,
                  timeout: Optional[float] = None) -> RCallResult:
    """Invoke an R script with the standard (tree, data, args, out) signature.

    ``script_name`` is the file name inside ``r_scripts/`` (e.g.
    ``"phyloglm_uni.R"``). The function returns an RCallResult; callers should
    check ``.ok`` and fall back gracefully when a particular test fails (e.g.
    phyloglm raising because of a singular covariance matrix for a specific
    rare defense system should not crash the whole pipeline).
    """
    script_path = R_SCRIPT_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"R script not found: {script_path}")

    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="defense_r_"))
    workdir.mkdir(parents=True, exist_ok=True)
    data_tsv = workdir / "data.tsv"
    out_tsv = workdir / "out.tsv"
    args_json = workdir / "args.json"

    data.to_csv(data_tsv, sep="\t", index=False)
    with open(args_json, "w") as fh:
        json.dump(args, fh)

    cmd = [r_executable, "--vanilla", str(script_path),
           str(tree_path), str(data_tsv), str(args_json), str(out_tsv)]
    logger.debug(f"R call: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return RCallResult(None, e.stdout or "", e.stderr or "", -1,
                           error=f"Timed out after {timeout}s")
    except FileNotFoundError as e:
        return RCallResult(None, "", "", -1,
                           error=f"Rscript not found: {e}. Install R and set "
                           "config.r_executable if it's not on PATH.")

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if proc.returncode != 0 or not out_tsv.exists():
        # Show enough stderr that embedded diagnostics (e.g. from
        # phyloglm_uni.R's "Too few matched tips" branch, which dumps
        # column names and sample values) aren't truncated.
        truncated = stderr[:2000] + ("…" if len(stderr) > 2000 else "")
        logger.warning(
            f"R script {script_name} failed (rc={proc.returncode}).\n"
            f"stderr:\n{truncated}"
        )
        return RCallResult(None, stdout, stderr, proc.returncode,
                           error=stderr.strip().splitlines()[-1] if stderr.strip() else None)

    try:
        df = pd.read_csv(out_tsv, sep="\t")
    except Exception as e:
        return RCallResult(None, stdout, stderr, proc.returncode,
                           error=f"Unable to read R output TSV: {e}")

    return RCallResult(df, stdout, stderr, proc.returncode)


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
