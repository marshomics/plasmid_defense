#!/usr/bin/env python3
"""Pre-submission checks for defense_analysis_v2 v3.2.

Run this before `./run_pipeline_sge.sh both`. It takes a few minutes and
catches the failures that would otherwise surface hours or days into a job:

  * a missing R package -> the stage dies on first call
  * a violated species-level plasmid invariant -> aggregation raises and the
    core job aborts after loading 342,000 rows
  * an entry-mode table that cannot be joined to species -> entry_mode is
    silently skipped
  * a tree/data tip mismatch -> "Too few matched tips (0)" from every R script
  * an unwritable output or scratch directory -> everything fails at the end

Usage:
    python preflight.py                    # uses config.py defaults
    python preflight.py --output-dir DIR   # override where results will go

Exit code 0 = safe to submit; 1 = at least one blocking problem.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from defense_analysis_v2.config import Config  # noqa: E402

OK, WARN, FAIL = "  [ok]  ", "  [warn]", "  [FAIL]"
_problems: list[str] = []
_warnings: list[str] = []


def ok(msg): print(f"{OK} {msg}")
def warn(msg): _warnings.append(msg); print(f"{WARN} {msg}")
def fail(msg): _problems.append(msg); print(f"{FAIL} {msg}")


def section(title):
    print(f"\n{'-' * 72}\n{title}\n{'-' * 72}")


# ======================================================================

def check_python_package():
    section("1. Python package")
    try:
        import defense_analysis_v2 as d
        from defense_analysis_v2 import defense_plasmid_analysis as dpa
        ok(f"defense_analysis_v2 v{d.__version__}, "
           f"{len(dpa.ALL_STAGES)} stages registered")
    except Exception as e:
        fail(f"cannot import the package: {e}")
        return
    for mod in ("phylo_signal_fast", "cost_model", "tier3_entry_mode",
                "tier3_sister_pairs", "tier3_feature_control"):
        try:
            __import__(f"defense_analysis_v2.{mod}")
        except Exception as e:
            fail(f"module {mod} does not import: {e}")


def check_installed_cli():
    """The CONSOLE SCRIPT must be the same code as the importable package.

    This check exists because it failed in practice. `preflight.py` inserts its
    own directory on sys.path, so it imports the SOURCE tree. But
    `defense-plasmid-analyze` is a console script installed into the
    environment, and it resolves to whatever copy was last `pip install`ed.
    A stale install means preflight validates v3.2 while the pipeline runs v2 —
    and since run_pipeline_sge.sh invokes the console script, every submitted
    job would silently run the old code.
    """
    import shutil
    section("2. Installed CLI matches the source tree")

    exe = shutil.which("defense-plasmid-analyze")
    if not exe:
        fail("'defense-plasmid-analyze' is not on PATH — run `pip install -e .` "
             "from the repository root")
        return
    ok(f"console script: {exe}")

    try:
        from defense_analysis_v2 import defense_plasmid_analysis as dpa
        import defense_analysis_v2 as d
        source_stages = set(dpa.ALL_STAGES)
        source_path = Path(d.__file__).resolve().parent
    except Exception as e:
        fail(f"cannot import the source package to compare against: {e}")
        return

    try:
        proc = subprocess.run([exe, "--help"], capture_output=True, text=True,
                              timeout=120)
    except Exception as e:
        fail(f"could not run '{exe} --help': {e}")
        return
    help_text = (proc.stdout or "") + (proc.stderr or "")

    missing = sorted(st for st in source_stages if st not in help_text)
    if missing:
        fail(f"the installed CLI does not know about {len(missing)} stage(s) "
             f"present in the source: {missing[:6]}"
             f"{'...' if len(missing) > 6 else ''}. "
             f"The console script is running a STALE INSTALL. "
             f"Fix with: pip install -e {source_path.parent}")
        # Show where the installed copy actually lives, which is the usual
        # source of confusion. Resolve the interpreter from the console
        # script's shebang rather than assuming a sibling `python`.
        try:
            shebang = Path(exe).read_text(errors="ignore").splitlines()[0]
            interp = shebang[2:].strip().split()[0] if shebang.startswith("#!") else None
            if interp and Path(interp).exists():
                probe = subprocess.run(
                    [interp, "-c",
                     "import defense_analysis_v2 as d;"
                     "print(d.__file__, d.__version__)"],
                    capture_output=True, text=True, timeout=60)
                if probe.returncode == 0 and probe.stdout.strip():
                    warn(f"the CLI's environment imports: {probe.stdout.strip()}")
        except Exception:
            pass
        warn(f"the source tree preflight checked is: {source_path} "
             f"(v{d.__version__})")
        return
    ok(f"installed CLI exposes all {len(source_stages)} source stages "
       f"(v{d.__version__})")


def check_r(cfg: Config):
    section("3. R and its packages")
    required = ["ape", "phylolm", "phytools", "caper", "phyr", "nlme",
                "jsonlite"]
    script = (f'cat(setdiff(c({", ".join(chr(34) + p + chr(34) for p in required)}), '
              f'rownames(installed.packages())), sep="\\n")')
    try:
        proc = subprocess.run([cfg.r_executable, "-e", script],
                              capture_output=True, text=True, timeout=180)
    except FileNotFoundError:
        fail(f"'{cfg.r_executable}' not on PATH")
        return
    except subprocess.TimeoutExpired:
        fail("R took over 3 minutes just to list packages")
        return
    missing = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
    if missing:
        fail(f"missing R packages: {missing} — run Rscript install_r_packages.R")
    else:
        ok(f"all {len(required)} R packages present")

    # The ARMA_64BIT_WORD question. At pglmm_max_species = 15000 the expanded
    # design is (2N)^2 = 9.0e8 elements, comfortably under the 2^31 = 2.1e9
    # 32-bit sparse index ceiling. The v2 overflow was a FULL-TREE problem
    # (2 x 39,681)^2 = 6.3e9. So the flag is not required at the default cap.
    n = int(cfg.pglmm_max_species or 39681)
    worst = (2 * n) ** 2
    if worst > 2 ** 31:
        warn(f"pglmm_max_species={n:,} gives a worst-case sparse extent of "
             f"{worst:,} > 2^31. RcppArmadillo MUST be built with "
             f"ARMA_64BIT_WORD=1 or pglmm_mv will die with "
             f"'SpMat::init(): requested size is too large'.")
    else:
        ok(f"pglmm_max_species={n:,}: worst-case sparse extent {worst:,} "
           f"< 2^31, so ARMA_64BIT_WORD is not required")


def check_inputs(cfg: Config):
    section("4. Input files")
    for label, path in [("strain table (subtype)", cfg.input_file),
                        ("strain table (type)", cfg.input_file_type_level),
                        ("tree", cfg.tree_file),
                        ("plasmid metadata", cfg.plasmid_metadata_file),
                        ("entry-mode table", cfg.entry_mode_metadata_file),
                        ("genome covariates", cfg.genome_covariates_file)]:
        p = Path(path)
        if not p.exists():
            (fail if "strain table" in label or label == "tree" else warn)(
                f"{label}: NOT FOUND at {path}")
        elif not os.access(p, os.R_OK):
            fail(f"{label}: exists but is not readable: {path}")
        else:
            ok(f"{label}: {p.stat().st_size / 1e6:,.0f} MB")


def check_plasmid_invariant(cfg: Config, max_rows: int | None = None):
    """The pipeline ABORTS if has_plasmid varies within a species."""
    section("5. Species-level plasmid invariant (aborts the run if violated)")
    p = Path(cfg.input_file)
    if not p.exists():
        warn("skipped — strain table not found")
        return
    try:
        usecols = ["gtdb_species", "has_plasmid"]
        seen: dict[str, set] = {}
        n_rows = 0
        for chunk in pd.read_csv(p, sep="\t", usecols=usecols,
                                 chunksize=200_000, low_memory=False):
            n_rows += len(chunk)
            v = (chunk["has_plasmid"].astype(str).str.strip().str.lower()
                 == "yes")
            for sp, val in zip(chunk["gtdb_species"], v):
                seen.setdefault(sp, set()).add(bool(val))
            if max_rows and n_rows >= max_rows:
                warn(f"checked only the first {n_rows:,} rows (--quick)")
                break
        bad = [sp for sp, vals in seen.items() if len(vals) > 1]
        if bad:
            fail(f"{len(bad)} species have CONFLICTING has_plasmid values "
                 f"across strains, e.g. {bad[:5]}. "
                 f"io_utils.aggregate_to_species_level raises on this and the "
                 f"core job will abort. Fix the upstream propagation.")
        else:
            ok(f"invariant holds across {n_rows:,} strains / "
               f"{len(seen):,} species")
        pos = sum(1 for vals in seen.values() if True in vals)
        prev = pos / max(len(seen), 1)
        ok(f"species-level plasmid prevalence: {prev:.1%}")
        if prev > 0.90:
            warn(f"prevalence is {prev:.1%} — very little contrast in the "
                 f"outcome. Expect wide CIs and check the depth-band "
                 f"concordance carefully.")
    except ValueError as e:
        fail(f"strain table missing an expected column: {e}")
    except Exception as e:
        fail(f"could not read the strain table: {e}")


def check_entry_mode(cfg: Config):
    section("6. Entry-mode table (A4)")
    p = Path(cfg.entry_mode_metadata_file)
    if not p.exists():
        warn("skipped — entry-mode table not found; the entry_mode stage "
             "will be skipped")
        return
    try:
        head = pd.read_csv(p, sep="\t", nrows=5000, dtype=str, low_memory=False)
    except Exception as e:
        fail(f"cannot read the entry-mode table: {e}")
        return

    for col in (cfg.entry_mode_plasmid_id_column,
                cfg.entry_mode_conjugative_column):
        if col not in head.columns:
            fail(f"entry-mode table has no '{col}' column. "
                 f"Found: {list(head.columns)[:15]}")
            return
    ok(f"columns '{cfg.entry_mode_plasmid_id_column}' and "
       f"'{cfg.entry_mode_conjugative_column}' present")

    from defense_analysis_v2.tier3_entry_mode import _parse_conjugative
    vals = head[cfg.entry_mode_conjugative_column]
    parsed = vals.map(_parse_conjugative)
    unparsed = parsed.isna().sum()
    if unparsed > len(head) * 0.5:
        fail(f"{unparsed}/{len(head)} sampled rows have an unrecognised "
             f"conjugative value: {vals[parsed.isna()].value_counts().head(5).to_dict()}")
    elif unparsed:
        warn(f"{unparsed}/{len(head)} sampled rows will be dropped as "
             f"unparseable: {vals[parsed.isna()].value_counts().head(3).to_dict()}")
    else:
        ok("every sampled conjugative value parses")
    if parsed.notna().any():
        ok(f"sampled conjugative fraction: {parsed.dropna().mean():.1%}")

    # Species: either present here, or recoverable by joining plasmid_id.
    if cfg.entry_mode_species_column in head.columns:
        ok(f"species column '{cfg.entry_mode_species_column}' present")
    else:
        pm = Path(cfg.plasmid_metadata_file)
        if not pm.exists():
            fail("entry-mode table has no species column AND the main plasmid "
                 "metadata is missing, so species cannot be recovered")
            return
        try:
            pmh = pd.read_csv(pm, sep="\t", nrows=200, dtype=str,
                              low_memory=False)
            if cfg.plasmid_id_column not in pmh.columns:
                fail(f"no species column in the entry-mode table, and the main "
                     f"plasmid metadata has no '{cfg.plasmid_id_column}' to "
                     f"join on. Found: {list(pmh.columns)[:15]}")
            else:
                ok(f"species will be recovered by joining on "
                   f"'{cfg.plasmid_id_column}'")
        except Exception as e:
            warn(f"could not inspect the plasmid metadata: {e}")


def check_tree(cfg: Config):
    section("7. Tree")
    p = Path(cfg.tree_file)
    if not p.exists():
        warn("skipped — tree not found")
        return
    try:
        import dendropy
        t = dendropy.Tree.get(path=str(p), schema="newick",
                              preserve_underscores=True,
                              suppress_internal_node_taxa=True)
        tips = [n.taxon.label.strip().replace(" ", "_")
                for n in t.leaf_node_iter() if n.taxon]
        ok(f"tree loads: {len(tips):,} tips")
        dups = len(tips) - len(set(tips))
        if dups:
            warn(f"{dups:,} duplicate tip labels — tree_utils.dedupe_newick_file "
                 f"handles these, but confirm the dedupe log looks sane")
        sp = pd.read_csv(cfg.input_file, sep="\t", usecols=["gtdb_species"],
                         low_memory=False)["gtdb_species"].astype(str)
        sp_norm = set(sp.str.strip().str.replace(" ", "_"))
        overlap = len(sp_norm & set(tips))
        if overlap < 100:
            fail(f"only {overlap} species match tree tips — every R script "
                 f"will report 'Too few matched tips'. Check label "
                 f"normalisation.")
        else:
            ok(f"{overlap:,} species match tree tips "
               f"({100 * overlap / max(len(sp_norm), 1):.1f}% of species)")
    except Exception as e:
        fail(f"could not load or match the tree: {e}")


def check_writable(cfg: Config, output_dir: str | None):
    section("8. Output and scratch directories")
    for label, path in [("output", output_dir or cfg.output_dir),
                        ("scratch", "/ebio/abt3_scratch/jmarsh")]:
        p = Path(path)
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".preflight_probe"
            probe.write_text("x")
            probe.unlink()
            ok(f"{label} directory writable: {p}")
        except Exception as e:
            (fail if label == "output" else warn)(
                f"{label} directory not writable ({p}): {e}")


def check_preregistration(cfg: Config):
    section("9. Pre-registration (A4) — human check, not automatable")
    print("  The entry-mode confirmatory test derives ALL its inferential value")
    print("  from the mechanism partition having been fixed before you look at")
    print("  any entry-mode result. Current partition:")
    print(f"    predicted (dsDNA-restricting): "
          f"{', '.join(cfg.entry_mode_predicted_categories)}")
    print(f"    not predicted:                 "
          f"{', '.join(cfg.entry_mode_not_predicted_categories)}")
    print(f"    primary outcomes:              "
          f"{', '.join(cfg.primary_outcome_labels)}")
    warn("confirm the partition above is final BEFORE submitting; editing it "
         "after seeing results invalidates the test")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--output-dir", default=None,
        help=("Directory the pipeline will write to. Defaults to config.py's "
              "output_dir, which may NOT be what run_pipeline_sge.sh uses — "
              "pass the same OUTDIR as that script to check the right path."))
    ap.add_argument("--quick", action="store_true",
                    help="check only the first 1M strain rows for the invariant")
    ns = ap.parse_args()

    cfg = Config()
    if ns.output_dir:
        cfg = Config(output_dir=ns.output_dir)

    print("=" * 72)
    print("defense_analysis_v2 v3.2 — pre-submission checks")
    print("=" * 72)

    check_python_package()
    check_installed_cli()
    check_r(cfg)
    check_inputs(cfg)
    check_plasmid_invariant(cfg, max_rows=1_000_000 if ns.quick else None)
    check_entry_mode(cfg)
    check_tree(cfg)
    check_writable(cfg, ns.output_dir)
    if ns.output_dir is None:
        warn(f"checked config.py's output_dir ({cfg.output_dir}). "
             f"run_pipeline_sge.sh may use a different OUTDIR — re-run with "
             f"--output-dir <that path> to verify it.")
    check_preregistration(cfg)

    print("\n" + "=" * 72)
    if _problems:
        print(f"NOT READY — {len(_problems)} blocking problem(s):")
        for p in _problems:
            print(f"  * {p}")
        print("\nFix these before submitting.")
        return 1
    print("READY TO SUBMIT.")
    if _warnings:
        print(f"\n{len(_warnings)} warning(s) — not blocking, but read them:")
        for w in _warnings:
            print(f"  * {w}")
    print("\nNext:")
    print("  1. Smoke-test the R side on a small subset (recommended — the R")
    print("     scripts were revised extensively and have not been executed):")
    print("       defense-plasmid-analyze --input ... --tree ... \\")
    print("         --output-dir /tmp/smoke --granularity subtype_level \\")
    print("         --stages tier1 phyloglm pagels entry_mode consensus \\")
    print("         --n-jobs 4 --n-permutations 50")
    print("  2. ./run_pipeline_sge.sh both")
    return 0


if __name__ == "__main__":
    sys.exit(main())
