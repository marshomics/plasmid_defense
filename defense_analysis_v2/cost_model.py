"""Predict per-stage cost before submitting, instead of discovering it 25 days in.

Four stages previously failed only after consuming most of a wall-clock budget:
LOCO and the misclassification Monte Carlo died with SIGBUS partway through,
phylo_signal was killed at the 25-day ceiling, and PGLMM exceeded node memory.
In every case the cost was predictable from the data dimensions and the config
before a single model was fit.

``estimate_pipeline_cost`` walks the configured stages and reports, per stage:
the number of R invocations, the number of model fits, projected wall-clock at
a given worker count, peak memory, and whether it fits the cluster envelope.
Exposed on the CLI as ``--estimate-cost``.

Calibration comes from the timings recorded in
``docs/cluster_optimization_log.md``, so the numbers are anchored to this
cluster and this dataset rather than to generic complexity arguments. They are
order-of-magnitude planning figures, not promises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Calibration constants, from observed runs on the Tübingen MPI cluster
# ----------------------------------------------------------------------

# phyloglm: ~45 s per system on ~40k tips (observed: 435 systems in ~5.5 h).
# phylolm uses Ho & Ané's linear-time algorithm, so cost scales ~linearly in
# tips rather than cubically.
PHYLOGLM_SEC_PER_SYSTEM_AT_40K = 45.0
PHYLOGLM_REF_TIPS = 40_000

# fitPagel: ~4-8 h per 500-tip subsample over 435 systems -> ~50 s per system.
# Scales roughly linearly in tips for the Felsenstein traversal.
PAGEL_SEC_PER_SYSTEM_AT_500 = 50.0
PAGEL_REF_TIPS = 500

# phyr::pglmm: memory ~ O(N^2); ~25 GB and several hours at 15k tips.
PGLMM_GB_AT_15K = 25.0
PGLMM_HOURS_AT_15K = 4.0
PGLMM_REF_TIPS = 15_000

# Native D statistic: measured ~0.06 h for 452 columns x 1000 permutations at
# 5k tips; linear in nodes.
DSTAT_HOURS_PER_COL_PERM_TIP = 0.06 / (452 * 1000 * 5_000)

# One serialised species x feature frame.
BYTES_PER_CELL = 2.2

CLUSTER_MAX_MEM_GB = 128.0
CLUSTER_MAX_HOURS = 600.0


@dataclass
class StageCost:
    stage: str
    r_calls: int
    model_fits: int
    tips_per_fit: int
    serial_hours: float
    parallel_hours: float
    peak_mem_gb: float
    temp_io_gb: float
    note: str = ""

    @property
    def fits_wallclock(self) -> bool:
        return self.parallel_hours <= CLUSTER_MAX_HOURS

    @property
    def fits_memory(self) -> bool:
        return self.peak_mem_gb <= CLUSTER_MAX_MEM_GB


def _phyloglm_hours(n_systems: int, n_tips: int) -> float:
    scale = max(n_tips, 1) / PHYLOGLM_REF_TIPS
    return n_systems * PHYLOGLM_SEC_PER_SYSTEM_AT_40K * scale / 3600.0


def _pagel_hours(n_systems: int, n_tips: int, directional_frac: float) -> float:
    scale = max(n_tips, 1) / PAGEL_REF_TIPS
    base = n_systems * PAGEL_SEC_PER_SYSTEM_AT_500 * scale / 3600.0
    # Directional fits add two more model fits for the gated fraction.
    return base * (1.0 + 2.0 * directional_frac)


def estimate_pipeline_cost(config, n_species: int, n_defense: int,
                           n_outcomes: int, n_jobs: int = 20,
                           n_clades_gated: int = 50,
                           n_species_with_plasmids: int = 4_000,
                           stages: Optional[List[str]] = None) -> pd.DataFrame:
    """Per-stage cost projection. Returns a frame ordered by parallel hours."""
    from . import defense_plasmid_analysis as dpa

    stages = list(stages or dpa.DEFAULT_STAGES)
    stages = [dpa.STAGE_ALIASES.get(s, s) for s in stages]
    n_jobs = max(1, int(n_jobs))

    n_modes_all = len([m for m in config.covariate_modes])
    n_modes_primary = 1
    n_dir = 2 if config.run_bidirectional else 1
    frame_mb = n_species * (n_defense + 25) * BYTES_PER_CELL / 1e6

    out: List[StageCost] = []

    def add(stage, r_calls, fits, tips, hours_serial, mem_gb, note="",
            shared_io=True):
        # With the shared-frame bridge, only the first call in a stage writes
        # the big frame; the rest write small side files.
        io_gb = (frame_mb / 1000 if shared_io
                 else r_calls * frame_mb / 1000)
        out.append(StageCost(
            stage=stage, r_calls=r_calls, model_fits=fits, tips_per_fit=tips,
            serial_hours=hours_serial,
            parallel_hours=hours_serial / min(n_jobs, max(r_calls, 1)),
            peak_mem_gb=mem_gb, temp_io_gb=io_gb, note=note))

    for st in stages:
        if st == "phyloglm":
            calls = n_outcomes * n_dir * n_modes_all
            add(st, calls, calls * n_defense, n_species,
                _phyloglm_hours(n_defense, n_species) * calls, 6.0,
                f"{n_outcomes} outcomes x {n_dir} directions x {n_modes_all} modes")
        elif st == "pagels":
            calls = n_outcomes * config.pagels_n_subsamples
            # Directional fits are gated twice: to primary outcomes, and
            # within a call to systems that reject independence.
            frac_out = (len(config.primary_outcome_labels) / max(n_outcomes, 1)
                        if config.pagels_directional_primary_outcomes_only else 1.0)
            frac_sys = 0.15 if config.pagels_directional_only_if_dependent else 1.0
            dfrac = (frac_out * frac_sys
                     if config.pagels_fit_directional_models else 0.0)
            add(st, calls, calls * n_defense, config.pagels_subsample_size,
                _pagel_hours(n_defense, config.pagels_subsample_size, dfrac) * calls,
                8.0, f"directional fits on ~{dfrac:.0%} of system-calls")
        elif st == "pglmm_mv":
            tips = int(config.pglmm_max_species or n_species)
            calls = n_outcomes * n_modes_all * 2
            mem = PGLMM_GB_AT_15K * (tips / PGLMM_REF_TIPS) ** 2
            add(st, calls, calls, tips,
                PGLMM_HOURS_AT_15K * (tips / PGLMM_REF_TIPS) ** 2 * calls, mem,
                f"memory ~O(N^2); {tips:,} tips")
        elif st == "loco":
            modes = (n_modes_primary if config.loco_covariate_modes_primary_only
                     else n_modes_all)
            ranks = 1 if config.loco_ranks_primary_only else len(config.loco_ranks)
            calls = modes * ranks * n_clades_gated
            add(st, calls, calls * n_defense, n_species,
                _phyloglm_hours(n_defense, n_species) * calls, 6.0,
                f"{n_clades_gated} size-gated clades x {ranks} rank(s) x {modes} mode(s)")
        elif st == "within_clade_het":
            modes = n_modes_primary if config.loco_covariate_modes_primary_only else 2
            calls = modes * n_clades_gated
            tips = max(config.min_species_per_within_clade_fit,
                       n_species // max(n_clades_gated, 1))
            add(st, calls, calls * n_defense, tips,
                _phyloglm_hours(n_defense, tips) * calls, 4.0,
                "within-clade subsets are small, so each fit is cheap")
        elif st == "misclass_mc":
            grid = (config.misclass_fnr_grid_reduced
                    if config.misclass_use_reduced_grid else config.misclass_fnr_grid)
            nrep = (config.misclass_n_replicates_effective
                    if config.misclass_use_reduced_grid else config.misclass_n_replicates)
            modes = (n_modes_primary if config.misclass_primary_mode_only
                     else n_modes_all)
            nsys = (min(config.misclass_max_systems, n_defense)
                    if config.misclass_restrict_to_significant else n_defense)
            calls = len(grid) * nrep * modes
            add(st, calls, calls * nsys, n_species,
                _phyloglm_hours(nsys, n_species) * calls, 6.0,
                f"{len(grid)} FNR x {nrep} reps x {modes} mode(s), {nsys} systems")
        elif st == "negative_control":
            calls = config.negative_control_n_replicates
            add(st, calls, calls * n_defense, n_species,
                _phyloglm_hours(n_defense, n_species) * calls, 6.0,
                "replicates dispatched in parallel")
        elif st == "feature_control":
            n_feat = (config.feature_control_max_systems
                      * config.feature_control_n_per_system)
            add(st, 1, n_feat, n_species,
                _phyloglm_hours(n_feat, n_species), 6.0,
                f"{n_feat} synthetic features in one sweep")
        elif st == "phylo_signal":
            n_cols = n_defense + n_outcomes
            if getattr(config, "phylo_signal_engine", "native") == "native":
                hrs = (DSTAT_HOURS_PER_COL_PERM_TIP * n_cols
                       * config.n_permutations * n_species * 2)
                add(st, 0, n_cols, n_species, hrs, 4.0,
                    "native vectorised D; caper was killed at the 25-day ceiling",
                    shared_io=False)
            else:
                add(st, 1, n_cols, n_species, 900.0, 8.0,
                    "caper: EXCEEDS the wall-clock ceiling; use engine='native'")
        elif st == "entry_mode":
            tips = n_species_with_plasmids
            # Chunked across workers: amortises R start-up AND parallelises.
            calls = (min(n_jobs, n_defense) if config.entry_mode_batch_in_r
                     else n_defense)
            mem = PGLMM_GB_AT_15K * (tips / PGLMM_REF_TIPS) ** 2
            add(st, calls, n_defense, tips,
                PGLMM_HOURS_AT_15K * (tips / PGLMM_REF_TIPS) ** 2 * n_defense,
                mem,
                "batched into one R process" if config.entry_mode_batch_in_r
                else "one R invocation per system")
        elif st == "sister_pairs":
            add(st, 0, 0, n_species, 0.2, 2.0, "pure Python; no R calls",
                shared_io=False)
        elif st in ("depth_sens", "prev_feature_sens", "phylo_model_sens"):
            n_arms = {"depth_sens": 2,
                      "prev_feature_sens": 1,
                      "phylo_model_sens": (len(config.phyloglm_estimator_sensitivity)
                                           + len(config.phylo_lambda_sensitivity))}[st]
            calls = n_arms * max(1, n_modes_all - 1)
            add(st, calls, calls * n_defense, n_species,
                _phyloglm_hours(n_defense, n_species) * calls, 6.0)
        elif st in ("tier1", "clade_perm", "depth_match", "consensus",
                    "misclass_analytical", "defense_misclass",
                    "phylo_vs_nonphylo", "figures", "rf", "burden"):
            add(st, 4 if st in ("burden",) else 0, 0, n_species, 0.5, 4.0,
                "cheap / Python-side", shared_io=False)
        elif st == "lasso":
            add(st, 1, n_defense, n_species, 100.0, 12.0,
                "sequential nlme::gls per predictor; BLAS-threaded config")

    df = pd.DataFrame([vars(s) for s in out])
    if df.empty:
        return df
    df["fits_wallclock"] = df["parallel_hours"] <= CLUSTER_MAX_HOURS
    df["fits_memory"] = df["peak_mem_gb"] <= CLUSTER_MAX_MEM_GB
    return df.sort_values("parallel_hours", ascending=False).reset_index(drop=True)


def format_cost_report(df: pd.DataFrame, n_jobs: int = 20) -> str:
    """Human-readable projection, for the CLI."""
    if df is None or df.empty:
        return "No stages to estimate."
    lines = []
    lines.append("=" * 92)
    lines.append(f"PROJECTED COST  (workers = {n_jobs}; cluster envelope = "
                 f"{CLUSTER_MAX_MEM_GB:.0f} GB / {CLUSTER_MAX_HOURS:.0f} h)")
    lines.append("=" * 92)
    lines.append(f"{'stage':<20}{'R calls':>9}{'fits':>12}{'serial h':>11}"
                 f"{'parallel h':>12}{'mem GB':>9}{'IO GB':>8}{'ok':>5}")
    lines.append("-" * 92)
    for _, r in df.iterrows():
        ok = "yes" if (r["fits_wallclock"] and r["fits_memory"]) else "NO"
        lines.append(
            f"{r['stage']:<20}{int(r['r_calls']):>9,}{int(r['model_fits']):>12,}"
            f"{r['serial_hours']:>11,.1f}{r['parallel_hours']:>12,.1f}"
            f"{r['peak_mem_gb']:>9,.1f}{r['temp_io_gb']:>8,.1f}{ok:>5}")
    lines.append("-" * 92)
    total_par = df["parallel_hours"].sum()
    lines.append(f"{'TOTAL (sequential stages)':<20}"
                 f"{int(df['r_calls'].sum()):>9,}{int(df['model_fits'].sum()):>12,}"
                 f"{df['serial_hours'].sum():>11,.1f}{total_par:>12,.1f}"
                 f"{df['peak_mem_gb'].max():>9,.1f}"
                 f"{df['temp_io_gb'].sum():>8,.1f}")
    lines.append("")
    bad = df[~(df["fits_wallclock"] & df["fits_memory"])]
    if bad.empty:
        lines.append("Every stage fits the cluster envelope.")
    else:
        lines.append("STAGES THAT DO NOT FIT:")
        for _, r in bad.iterrows():
            why = []
            if not r["fits_wallclock"]:
                why.append(f"{r['parallel_hours']:,.0f} h > {CLUSTER_MAX_HOURS:.0f} h")
            if not r["fits_memory"]:
                why.append(f"{r['peak_mem_gb']:,.0f} GB > {CLUSTER_MAX_MEM_GB:.0f} GB")
            lines.append(f"  {r['stage']:<20} {'; '.join(why)}   {r['note']}")
    lines.append("")
    lines.append("Notes:")
    for _, r in df.iterrows():
        if r["note"]:
            lines.append(f"  {r['stage']:<20} {r['note']}")
    return "\n".join(lines)
