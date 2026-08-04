#!/usr/bin/env python3
"""Pipeline driver — defense_analysis_v2.

Runs the full scientifically-defensible pipeline or an arbitrary subset of
stages. Stages are:

    tier1                 : Firth-weighted logistic + diagnostics
    phyloglm              : Tier 2 univariate phylogenetic logistic regression
    pagels                : Pagel's correlated-evolution test
    pglmm_mv              : Tier 2 multivariate PGLMM
    lasso                 : LASSO / Elastic Net on phylo-residualised data
    rf                    : clade-blocked Random Forest (binary + prevalence)
    burden                : phylo-corrected burden (PGLS + phyloglm on count)
    loco                  : leave-one-clade-out with Cochran's Q
    phylo_signal          : D-statistic
    clade_perm            : clade-restricted permutation
    prev_match            : prevalence-matched paired test
    misclass_mc           : misclassification Monte Carlo
    misclass_analytical   : analytical bias correction
    consensus             : rank-product across phyloglm + PGLMM + Pagel's
    phylo_vs_nonphylo     : side-by-side Tier 1 vs Tier 2 comparison
    figures               : plotting

Usage:
    python -m defense_analysis_v2.defense_plasmid_analysis \\
        --input data/species_data_binary.tsv \\
        --tree data/species_tree.nwk \\
        --output-dir results/ \\
        --stages phyloglm pglmm_mv burden loco misclass_mc consensus figures
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Support both "python -m defense_analysis_v2.defense_plasmid_analysis" and
# "python defense_analysis_v2/defense_plasmid_analysis.py"
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from defense_analysis_v2 import (
        config, logging_setup, io_utils, tree_utils, r_bridge,
        tier1, tier2_phylo_uni, tier2_pagels, tier2_multivariate,
        tier2_random_forest, tier3_burden, tier3_loco,
        tier3_entry_mode, tier3_sister_pairs, tier3_feature_control,
        tier3_misclassification, tier3_sensitivity, stats_utils,
        consensus as consensus_mod, reporting, plotting,
    )
else:
    from . import (config, logging_setup, io_utils, tree_utils, r_bridge,
                   tier1, tier2_phylo_uni, tier2_pagels, tier2_multivariate,
                   tier2_random_forest, tier3_burden, tier3_loco,
                   tier3_entry_mode, tier3_sister_pairs, tier3_feature_control,
                   tier3_misclassification, tier3_sensitivity, stats_utils)
    from . import consensus as consensus_mod
    from . import reporting, plotting


def _rank_systems_for_interactions(phyloglm_df, n_top: int):
    """Return the top-N defense systems by primary-direction phyloglm p-value
    against the legacy any_plasmid outcome. Used to pick pairwise interaction
    terms for the multivariate PGLMM without over-fitting the interaction
    space.
    """
    if phyloglm_df is None or phyloglm_df.empty:
        return None
    df = phyloglm_df
    if "outcome_label" in df.columns:
        df = df[df["outcome_label"] == "any_plasmid"]
    if "direction" in df.columns:
        df = df[df["direction"] == "plasmid_given_defense"]
    if df.empty or "phyloglm_p_value" not in df.columns:
        return None
    ordered = df.sort_values("phyloglm_p_value")["defense_system"].dropna().tolist()
    return ordered[:max(n_top, 0)]


ALL_STAGES = [
    # The negative control runs FIRST. It permutes the plasmid label within
    # (clade x sequencing-depth) strata and re-runs the primary sweep; if the
    # signal survives that, the pipeline is measuring sequencing effort and
    # nothing downstream is interpretable.
    "negative_control",
    "tier1", "phyloglm", "pagels", "pglmm_mv", "lasso", "rf", "burden",
    "loco", "within_clade_het", "phylo_signal", "clade_perm", "depth_match",
    "misclass_mc", "misclass_analytical", "defense_misclass",
    # Sampling-depth and feature-mode sensitivity reruns of the primary
    # phyloglm — these target the saturation shared by the max()-aggregated
    # defense call and the species-propagated plasmid label.
    "depth_sens", "prev_feature_sens",
    # Estimator and covariance-structure sensitivity.
    "phylo_model_sens",
    # A4 pre-registered entry-mode prediction; B1 matched sister pairs;
    # B3 matched-feature negative control.
    "entry_mode", "sister_pairs", "feature_control",
    "consensus", "phylo_vs_nonphylo", "figures",
]

# Stage aliases so existing --stages invocations keep working after renames.
STAGE_ALIASES = {
    "prev_match": "depth_match",
    "min_n_strains_sens": "depth_sens",
}


# ---- Checkpoint registry ------------------------------------------------
#
# Each entry maps a stage name to the list of ``outputs`` keys it produces.
# The pipeline saves these to disk as TSV immediately after the stage runs
# and reloads them on subsequent runs so partial-completion is safe. The
# first key in each list is treated as the stage's "primary" output for
# cache-hit detection — if it's present on disk the stage is considered
# done.
STAGE_OUTPUTS = {
    "tier1":               ["tier1"],
    "phyloglm":            ["tier2_phyloglm"],
    "pagels":              ["tier2_pagels"],
    "pglmm_mv":            ["tier2_pglmm_mv", "lasso", "elastic_net", "mv_stability"],
    "lasso":               ["lasso", "elastic_net"],
    "rf":                  ["rf_binary", "rf_prevalence", "rf_fold_aucs"],
    "burden":              ["burden_pgls", "burden_phyloglm"],
    "loco":                ["tier3_loco_detail", "tier3_loco_summary"],
    "phylo_signal":        ["tier3_phylo_signal"],
    "clade_perm":          ["tier3_perm"],
    "depth_match":         ["tier3_depth_matched"],
    "depth_sens":          ["tier3_depth_sens", "tier3_depth_band_concordance"],
    "negative_control":    ["negative_control"],
    "within_clade_het":    ["tier3_within_clade_detail",
                            "tier3_within_clade_summary"],
    "defense_misclass":    ["defense_misclass_analytical"],
    "entry_mode":          ["entry_mode_composition", "entry_mode_confirmatory",
                            "entry_mode_binary"],
    "sister_pairs":        ["sister_pair_summary", "sister_pair_detail",
                            "sister_vs_primary"],
    "feature_control":     ["feature_control_results",
                            "feature_control_comparison"],
    "prev_feature_sens":   ["tier3_prev_feature_sens"],
    "phylo_model_sens":    ["tier3_phylo_model_sens"],
    "misclass_mc":         ["misclass_mc_long", "misclass_summary"],
    "misclass_analytical": ["misclass_analytical", "misclass_analytical_summary"],
    "consensus":           ["consensus", "covariate_impact",
                            "binomial_concordance"],
    "phylo_vs_nonphylo":   ["phylo_vs_nonphylo"],
    # figures: no checkpoint — always regenerated from upstream tables
}


def _load_existing_checkpoints(out_dir: Path, logger) -> Dict[str, object]:
    """Scan ``out_dir`` for any TSV that matches a known stage output key
    and load it back into the ``outputs`` dict. Lets the pipeline pick up
    where a previous run left off.
    """
    outputs: Dict[str, object] = {}
    for stage_name, keys in STAGE_OUTPUTS.items():
        for k in keys:
            p = out_dir / f"{k}.tsv"
            if p.exists() and k not in outputs:
                try:
                    outputs[k] = pd.read_csv(p, sep="\t")
                except Exception as e:
                    logger.warning(f"Could not load cached {k} from {p}: {e}")
    if outputs:
        logger.info(
            f"Resumed from {len(outputs)} cached output(s) under {out_dir}: "
            f"{sorted(outputs.keys())}"
        )
    return outputs


def _fingerprint_path(out_dir: Path) -> Path:
    return out_dir / ".config_fingerprint"


def check_config_fingerprint(cfg, out_dir: Path, logger) -> bool:
    """Compare the current config against the one that produced the cached
    TSVs. Returns True if they match (cache is safe to reuse).

    Cache validity previously keyed on file existence ALONE — no hash of the
    config, the inputs, or the covariate list. Changing a threshold
    (min_n_strains_sensitivity 5 -> 20, say) and re-running silently reused
    the old TSV while logging a reassuring "cached" message. For a pipeline
    whose entire purpose is a sensitivity sweep over thresholds, that is a
    reproducibility trap: the reported sensitivity analysis could correspond
    to a threshold nobody ever ran.
    """
    fp_path = _fingerprint_path(out_dir)
    current = cfg.fingerprint()
    def _write_atomic(text: str) -> None:
        # Concurrent sibling jobs write the same value here; atomic replace
        # means a concurrent reader never sees a truncated hash.
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp = fp_path.with_suffix(".tmp")
            tmp.write_text(text)
            tmp.replace(fp_path)
        except Exception:
            pass

    if not fp_path.exists():
        _write_atomic(current)
        return True
    previous = fp_path.read_text().strip()
    if previous == current:
        return True
    logger.warning(
        f"Config fingerprint changed ({previous} -> {current}). Cached stage "
        f"outputs in {out_dir} were produced under different settings and will "
        f"NOT be reused. Delete the directory to reclaim the space.")
    _write_atomic(current)
    return False


def _is_stage_cached(stage_name: str, outputs: Dict[str, object],
                     force_rerun: set, config_matches: bool = True) -> bool:
    """A stage is cached if its primary output key is already in ``outputs``
    (loaded from disk), the user hasn't asked to rerun it, AND the config that
    produced the cache matches the current one.
    """
    if not config_matches:
        return False
    if "all" in force_rerun or stage_name in force_rerun:
        return False
    keys = STAGE_OUTPUTS.get(stage_name, [stage_name])
    return bool(keys) and keys[0] in outputs


def _save_stage_outputs(stage_name: str, outputs: Dict[str, object],
                        out_dir: Path, logger) -> None:
    """Persist a stage's outputs to ``out_dir`` as TSVs. Called immediately
    after each stage finishes so a job that's killed partway through a
    later stage doesn't lose the earlier stages' work.
    """
    keys = STAGE_OUTPUTS.get(stage_name, [stage_name])
    for k in keys:
        df = outputs.get(k)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue
        p = out_dir / f"{k}.tsv"
        try:
            # Write-then-rename. Jobs 2-4 of the cluster submission all start
            # when the core job finishes and each loads existing checkpoints at
            # startup, so one job can be reading this directory while another
            # is writing it. rename() is atomic within a filesystem, so a
            # reader sees either the old file or the complete new one, never a
            # half-written frame.
            tmp = p.with_suffix(".tsv.tmp")
            df.to_csv(tmp, sep="\t", index=False)
            tmp.replace(p)
            logger.info(
                f"  [{stage_name}] checkpointed {k} ({len(df)} rows) -> {p.name}"
            )
        except Exception as e:
            logger.warning(f"  [{stage_name}] failed to checkpoint {k}: {e}")

DEFAULT_STAGES = [
    # Statistical stages in dependency order. `figures` is run last.
    # `negative_control` is first on purpose — see ALL_STAGES.
    "negative_control",
    "tier1", "phyloglm", "pagels", "pglmm_mv", "lasso", "rf", "burden",
    "loco", "within_clade_het", "phylo_signal", "clade_perm", "depth_match",
    "misclass_mc", "misclass_analytical", "defense_misclass",
    "depth_sens", "prev_feature_sens", "phylo_model_sens",
    "entry_mode", "sister_pairs", "feature_control",
    "consensus", "phylo_vs_nonphylo", "figures",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phylogenetically-corrected defense-vs-plasmid analysis")
    p.add_argument("--input", help="species-level binary defense TSV")
    p.add_argument("--input-type", help="type-level binary defense TSV")
    p.add_argument("--tree", help="Newick phylogenetic tree of species")
    p.add_argument("--output-dir", required=False,
                   help="output directory (results + figures written here)")
    p.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=None,
                   help="stages to run (default: all)")
    p.add_argument("--granularity", choices=["subtype_level", "type_level", "both"],
                   default="both")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--fdr-method", default="fdr_bh")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--min-prev-mv", type=float, default=0.10)
    p.add_argument("--rscript", default="Rscript",
                   help="R executable (default: Rscript on PATH)")
    p.add_argument(
        "--subsample", type=int, default=None, metavar="N",
        help=("SMOKE TEST ONLY. Randomly subsample to N species (stratified "
              "by phylum) after aggregation, so the R side can be exercised "
              "in minutes. Results from a subsample are NOT the analysis and "
              "the run is tagged SUBSAMPLE in the log."))
    p.add_argument(
        "--estimate-cost", action="store_true",
        help=("Print a per-stage projection of R calls, model fits, "
              "wall-clock and memory for the configured stages, then exit "
              "WITHOUT running anything. Use this before submitting: four "
              "stages previously failed only after consuming most of a "
              "25-day budget, and every one of those failures was "
              "predictable from the data dimensions and the config."))
    p.add_argument(
        "--estimate-species", type=int, default=39681,
        help="Species count to assume for --estimate-cost (default: 39681)")
    p.add_argument(
        "--estimate-systems", type=int, default=435,
        help="Defense-system count for --estimate-cost (default: 435)")
    p.add_argument(
        "--estimate-outcomes", type=int, default=17,
        help="Outcome-stratum count for --estimate-cost (default: 17)")
    p.add_argument(
        "--force-rerun", nargs="+", default=[], metavar="STAGE",
        help=("Stage(s) to force-rerun even if a cached output is present "
              "on disk. Pass 'all' to ignore the cache entirely. By "
              "default, any stage whose primary output TSV already exists "
              "in the output directory is skipped on subsequent runs. "
              f"Valid stage names: {', '.join(ALL_STAGES)}."),
    )
    return p


def apply_cli_to_config(cfg: config.Config, ns: argparse.Namespace) -> config.Config:
    kw = {}
    if ns.input:       kw["input_file"] = ns.input
    if ns.input_type:  kw["input_file_type_level"] = ns.input_type
    if ns.tree:        kw["tree_file"] = ns.tree
    if ns.output_dir:  kw["output_dir"] = ns.output_dir
    if ns.stages:      kw["stages"] = tuple(ns.stages)
    kw["alpha"] = ns.alpha
    kw["fdr_method"] = ns.fdr_method
    kw["n_jobs"] = ns.n_jobs
    kw["n_permutations"] = ns.n_permutations
    kw["min_prevalence_multivariate"] = ns.min_prev_mv
    kw["r_executable"] = ns.rscript
    return replace(cfg, **kw)


def run_pipeline(input_path: str, cfg: config.Config,
                 granularity_label: str,
                 force_rerun: Optional[set] = None) -> Dict[str, object]:
    """Run the pipeline for a single granularity. Returns a dict of all
    result DataFrames keyed by the names used in reporting/plotting.

    Stages are checkpointed to ``cfg.output_dir/<granularity_label>/``;
    each stage's primary output is written immediately after the stage
    completes, and on a subsequent run any stage whose output is already
    on disk is skipped. Pass stage names (or "all") in ``force_rerun`` to
    bypass the cache.
    """
    out_dir = Path(cfg.output_dir) / granularity_label
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging_setup.setup_logging(str(out_dir),
                                         name=f"defense_analysis_v2.{granularity_label}")
    logger.info(f"=== Pipeline run: {granularity_label} ===")
    logger.info(f"Input     : {input_path}")
    logger.info(f"Tree      : {cfg.tree_file}")
    logger.info(f"Output    : {out_dir}")
    logger.info(f"Stages    : {cfg.stages or DEFAULT_STAGES}")
    if force_rerun:
        logger.info(f"Force-rerun: {sorted(force_rerun)}")
    t_start = time.time()

    stages = cfg.stages or tuple(DEFAULT_STAGES)
    # Resolve renamed stages so existing --stages invocations keep working.
    stages = tuple(STAGE_ALIASES.get(st, st) for st in stages)
    force_rerun = set(STAGE_ALIASES.get(st, st) for st in (force_rerun or []))

    # R availability check — only enforced if any stage that uses R is scheduled
    r_stages = {"negative_control", "within_clade_het", "depth_sens",
                "depth_match", "entry_mode", "feature_control",
                "phyloglm", "pagels", "pglmm_mv", "lasso", "burden", "loco",
                "phylo_signal", "misclass_mc",
                "prev_feature_sens", "phylo_model_sens"}
    if r_stages & set(stages):
        cfg.require_r()
        r_bridge.ensure_r_packages(
            cfg.r_executable,
            ["ape", "phylolm", "phytools", "caper", "phyr", "nlme", "jsonlite"],
            logger,
        )

    # Load + aggregate
    strain_df, defense_cols = io_utils.load_and_preprocess_data(cfg, logger, input_path)
    strain_df = io_utils.load_genome_covariates(cfg, strain_df, logger)
    plasmid_md = io_utils.load_plasmid_metadata(cfg, logger)
    prevalence_df, binary_df, outcome_spec = io_utils.aggregate_to_species_level(
        strain_df, defense_cols, logger, config=cfg, plasmid_md=plasmid_md)
    prevalence_df, binary_df = io_utils.add_defense_burden(
        prevalence_df, binary_df, defense_cols)

    # ---- smoke-test subsampling ----
    if cfg.subsample_species and cfg.subsample_species < len(prevalence_df):
        import numpy as _np
        rng = _np.random.default_rng(cfg.random_seed)
        n_target = int(cfg.subsample_species)
        rank = "gtdb_phylum" if "gtdb_phylum" in prevalence_df.columns else None
        if rank:
            # Proportional draw within each phylum, minimum 1, so the pruned
            # tree keeps a realistic shape rather than collapsing to one clade.
            frac = n_target / len(prevalence_df)
            keep = []
            for _, grp in prevalence_df.groupby(rank):
                k = max(1, int(round(len(grp) * frac)))
                keep.extend(grp.sample(min(k, len(grp)),
                                       random_state=int(cfg.random_seed)).index)
            idx = prevalence_df.loc[keep, "gtdb_species"]
        else:
            idx = prevalence_df.sample(n_target,
                                       random_state=int(cfg.random_seed))["gtdb_species"]
        keep_sp = set(idx)
        prevalence_df = prevalence_df[prevalence_df["gtdb_species"].isin(keep_sp)]
        binary_df = binary_df[binary_df["gtdb_species"].isin(keep_sp)]
        # The depth spline knots are quantiles of the rows being fit, so they
        # must be rebuilt on the subsample.
        prevalence_df = io_utils.add_depth_basis(prevalence_df, cfg, logger)
        binary_df = io_utils.add_depth_basis(binary_df, cfg, logger)
        logger.warning(
            f"*** SUBSAMPLE MODE: {len(prevalence_df):,} species "
            f"({100 * binary_df['has_plasmid_binary'].mean():.1f}% "
            f"plasmid-positive). This is a SMOKE TEST. Results are NOT the "
            f"analysis and must not be reported. ***")

    # ---- A4: entry-mode (conjugative / non-conjugative) plasmid features ----
    # Merged here so build_phylo_dataframe carries them through to R along
    # with every other species-level column.
    #
    # Built on COPIES and only committed once the whole block succeeds. The
    # previous version merged in place and then raised in the fillna loop; the
    # except clause logged and continued, but the merges had already happened,
    # so the pipeline carried on with a frame whose colliding columns had been
    # renamed to _x/_y. `any_plasmid_conjugative` ceased to exist and the
    # conjugative mobility stratum was silently dropped from every downstream
    # stage. A partial failure must not leave a half-modified frame behind.
    if "entry_mode" in (cfg.stages or DEFAULT_STAGES):
        try:
            em_table = tier3_entry_mode.load_entry_mode_table(
                cfg, logger, plasmid_md=plasmid_md)
            em_feats = tier3_entry_mode.build_entry_mode_features(
                em_table, prevalence_df["gtdb_species"].tolist(), cfg, logger)
            if not em_feats.empty:
                em_cols = [c for c in em_feats.columns if c != "gtdb_species"]
                # Fail loudly rather than let pandas silently rename on merge.
                clash = sorted(set(em_cols) & set(prevalence_df.columns))
                if clash:
                    raise ValueError(
                        f"entry-mode feature names collide with existing "
                        f"columns: {clash}. All entry-mode columns must carry "
                        f"the 'em_' prefix.")
                prev_new = prevalence_df.merge(em_feats, on="gtdb_species",
                                               how="left")
                bin_new = binary_df.merge(em_feats, on="gtdb_species",
                                          how="left")
                # Species with no plasmids are structural zeros, not missing.
                for dfref in (prev_new, bin_new):
                    for c in em_cols:
                        dfref[c] = dfref[c].fillna(0)
                # Commit only now.
                prevalence_df, binary_df = prev_new, bin_new
                logger.info(f"Entry-mode features merged: {em_cols}")
        except Exception as exc:
            logger.error(
                f"Entry-mode feature construction failed: {exc!r}. "
                f"The entry_mode stage will be skipped; the rest of the "
                f"pipeline continues on the UNMODIFIED frame.")

    # Tree setup
    workdir = Path(tempfile.mkdtemp(prefix=f"defense_v2_{granularity_label}_"))
    try:
        import dendropy
        safe_tree_path = tree_utils.dedupe_newick_file(cfg.tree_file, logger)
        tree = dendropy.Tree.get(path=str(safe_tree_path), schema="newick",
                                 preserve_underscores=True)
        tip_labels = [tip.label for tip in tree.taxon_namespace]
        matched_species, matched_tips, sp2tip = tree_utils.match_species_to_tree(
            binary_df["gtdb_species"].tolist(), tip_labels, logger)
    except Exception as e:
        logger.error(f"Tree load failed: {e}; phylogenetic stages will be skipped")
        matched_species, matched_tips, sp2tip = [], [], {}

    pruned_tree_path = None
    if matched_tips:
        pruned_tree_path = workdir / "pruned_tree.nwk"
        tree_utils.preprocess_newick_to_file(
            cfg.tree_file, matched_tips, pruned_tree_path, logger)
        phylo_data = tree_utils.build_phylo_dataframe(binary_df, defense_cols, sp2tip)
        logger.info(f"Tree-matched species: {len(phylo_data)}")
    else:
        phylo_data = None

    # Pre-load any cached outputs from prior runs so partial-completion
    # is recoverable. Each stage below checks _is_stage_cached() before
    # actually running and calls _save_stage_outputs() immediately after,
    # so a job killed mid-stage only loses that one stage's progress.
    outputs: Dict[str, object] = _load_existing_checkpoints(out_dir, logger)
    # Cached TSVs are only reusable if the config that produced them matches.
    config_matches = check_config_fingerprint(cfg, out_dir, logger)
    if not config_matches:
        outputs = {}

    def _run(stage_name: str, runner) -> None:
        """Run a stage if requested and not cached, then checkpoint."""
        if stage_name not in stages:
            return
        if _is_stage_cached(stage_name, outputs, force_rerun, config_matches):
            logger.info(
                f"[{stage_name}] cached output present; skipping. "
                f"Use --force-rerun {stage_name} to recompute."
            )
            return
        runner()
        _save_stage_outputs(stage_name, outputs, out_dir, logger)

    # --------------------------------------------------------------
    # Tier 1 (non-phylogenetic baseline, diagnostic only)
    # --------------------------------------------------------------
    def _stage_negative_control():
        outputs["negative_control"] = tier3_sensitivity.run_negative_control(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger,
            workdir)
    if phylo_data is not None and cfg.run_negative_control:
        _run("negative_control", _stage_negative_control)

    def _stage_tier1():
        outputs["tier1"] = tier1.run_tier1(binary_df, prevalence_df,
                                           defense_cols, cfg, logger,
                                           outcome_spec=outcome_spec)
    _run("tier1", _stage_tier1)

    # --------------------------------------------------------------
    # Tier 2
    # --------------------------------------------------------------
    def _stage_phyloglm():
        outputs["tier2_phyloglm"] = tier2_phylo_uni.run_tier2_phyloglm_univariate(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger, workdir,
            outcome_spec=outcome_spec)
    if phylo_data is not None:
        _run("phyloglm", _stage_phyloglm)

    def _stage_pagels():
        outputs["tier2_pagels"] = tier2_pagels.run_pagels_test(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger, workdir,
            outcome_spec=outcome_spec)
    if phylo_data is not None:
        _run("pagels", _stage_pagels)

    def _stage_pglmm_mv():
        # Interaction-term picking uses the primary-direction phyloglm ranks
        # against the legacy any_plasmid outcome. Filter if long-form.
        ranked = _rank_systems_for_interactions(outputs.get("tier2_phyloglm"),
                                                cfg.n_interaction_systems)
        mv_result = tier2_multivariate.run_tier2_multivariate(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger, workdir,
            outcome_spec=outcome_spec, ranked_systems=ranked)
        outputs["tier2_pglmm_mv"] = mv_result.pglmm
        outputs["lasso"] = mv_result.lasso
        outputs["elastic_net"] = mv_result.elastic_net
        outputs["mv_stability"] = mv_result.stability

    def _stage_lasso_only():
        lasso, enet = tier2_multivariate.run_regularised_on_residuals(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger, workdir)
        outputs["lasso"] = lasso
        outputs["elastic_net"] = enet

    if phylo_data is not None:
        if "pglmm_mv" in stages:
            _run("pglmm_mv", _stage_pglmm_mv)
        elif "lasso" in stages:
            _run("lasso", _stage_lasso_only)

    # --------------------------------------------------------------
    # Clade-blocked Random Forest (not R-dependent)
    # --------------------------------------------------------------
    def _stage_rf():
        try:
            rf_res = tier2_random_forest.run_clade_blocked_rf(
                binary_df, prevalence_df, defense_cols, cfg, logger,
                clade_rank="gtdb_class")
            outputs["rf_binary"] = rf_res.binary
            outputs["rf_prevalence"] = rf_res.prevalence
            outputs["rf_fold_aucs"] = rf_res.fold_aucs
        except Exception as e:
            logger.warning(f"Clade-blocked RF failed: {e}")
    _run("rf", _stage_rf)

    # --------------------------------------------------------------
    # Tier 3
    # --------------------------------------------------------------
    def _stage_burden():
        outputs["burden_pgls"] = tier3_burden.run_burden_pgls(
            phylo_data, str(pruned_tree_path), cfg, logger, workdir)
        outputs["burden_phyloglm"] = tier3_burden.run_burden_phyloglm(
            phylo_data, str(pruned_tree_path), cfg, logger, workdir)
    if phylo_data is not None:
        _run("burden", _stage_burden)

    def _stage_loco():
        # LOCO runs against the legacy any_plasmid outcome only — its purpose
        # is to check stability of the primary-outcome association, not to
        # multiply the analysis across strata.
        loco = tier3_loco.run_loco_with_cochran_q(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger,
            workdir, outputs["tier2_phyloglm"],
            outcome_label="any_plasmid",
            outcome_col="has_plasmid_binary")
        outputs["tier3_loco_detail"] = loco["details"]
        outputs["tier3_loco_summary"] = loco["summary"]
    if phylo_data is not None and "tier2_phyloglm" in outputs:
        _run("loco", _stage_loco)

    def _stage_within_clade_het():
        # The VALID heterogeneity test: within-clade fits are disjoint, so
        # Cochran's Q against chi2(k-1) holds. LOCO estimates share >90% of
        # their data and cannot support a heterogeneity p-value.
        het = tier3_loco.run_within_clade_heterogeneity(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger,
            workdir, outcome_label="any_plasmid",
            outcome_col="has_plasmid_binary")
        outputs["tier3_within_clade_detail"] = het["details"]
        outputs["tier3_within_clade_summary"] = het["summary"]
    if phylo_data is not None:
        _run("within_clade_het", _stage_within_clade_het)

    def _stage_phylo_signal():
        outputs["tier3_phylo_signal"] = tier3_sensitivity.run_phylogenetic_signal(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger, workdir)
    if phylo_data is not None:
        _run("phylo_signal", _stage_phylo_signal)

    def _stage_clade_perm():
        outputs["tier3_perm"] = tier3_sensitivity.run_clade_permutation(
            binary_df, defense_cols, cfg, logger)
    _run("clade_perm", _stage_clade_perm)

    def _stage_depth_match():
        # Match plasmid+ to plasmid- species on SAMPLING DEPTH, then test
        # defense presence with McNemar. The old "prevalence-matched" test
        # matched on deciles of defense prevalence and then tested the binary
        # derived from that same prevalence, so every paired difference was
        # structurally zero and every p-value was NaN.
        outputs["tier3_depth_matched"] = tier3_sensitivity.run_depth_matched(
            binary_df, prevalence_df, defense_cols, cfg, logger)
    _run("depth_match", _stage_depth_match)

    # --------------------------------------------------------------
    # Sampling-depth and feature-mode sensitivity reruns of the primary
    # phyloglm. These directly target the max()-saturation bias that
    # log_n_strains as a covariate can only partially correct for.
    # --------------------------------------------------------------
    def _stage_depth_sens():
        # Runs BOTH depth bands: the legacy high-depth filter (which enriches
        # for the artefact) and the low-depth complement (where saturation
        # cannot have operated). The concordance table is the deliverable.
        ds = tier3_sensitivity.run_min_n_strains_sensitivity(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger, workdir)
        outputs["tier3_depth_sens"] = ds
        conc = tier3_sensitivity.build_depth_band_concordance(ds, cfg)
        if conc is not None and not conc.empty:
            outputs["tier3_depth_band_concordance"] = conc
            n_artefact = int(
                (conc["depth_verdict"]
                 == "high_depth_only__possible_sampling_artefact").sum())
            n_robust = int((conc["depth_verdict"] == "robust_to_depth").sum())
            logger.info(
                f"Depth-band concordance: {n_robust} systems robust across "
                f"depth bands, {n_artefact} significant only in the deep tail "
                f"(possible sampling artefact)")
    if phylo_data is not None:
        _run("depth_sens", _stage_depth_sens)

    def _stage_prev_feature():
        outputs["tier3_prev_feature_sens"] = tier3_sensitivity.run_prevalence_feature_sensitivity(
            phylo_data, prevalence_df, defense_cols, str(pruned_tree_path),
            cfg, logger, workdir)
    if phylo_data is not None and cfg.run_prevalence_feature_sensitivity:
        _run("prev_feature_sens", _stage_prev_feature)

    def _stage_phylo_model_sens():
        outputs["tier3_phylo_model_sens"] = tier3_sensitivity.run_phylo_model_sensitivity(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger, workdir)
    if phylo_data is not None and cfg.phylo_model_sensitivity_models:
        _run("phylo_model_sens", _stage_phylo_model_sens)

    # --------------------------------------------------------------
    # Misclassification sensitivity
    # --------------------------------------------------------------
    def _stage_misclass_mc():
        mc_long = tier3_misclassification.run_misclassification_mc(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger,
            workdir, tier2_phyloglm=outputs.get("tier2_phyloglm"))
        outputs["misclass_mc_long"] = mc_long
        outputs["misclass_summary"] = tier3_misclassification.summarise_misclassification_mc(
            mc_long, cfg)
    # Depends on tier2_phyloglm so the sweep can be scoped to actual findings.
    if phylo_data is not None:
        _run("misclass_mc", _stage_misclass_mc)

    def _stage_misclass_analytical():
        outputs["misclass_analytical"] = tier3_misclassification.analytical_bias_correction(
            outputs["tier2_phyloglm"], outputs["tier1"],
            cfg.misclass_fnr_grid, cfg)
        outputs["misclass_analytical_summary"] = outputs["misclass_analytical"]
    if "tier2_phyloglm" in outputs and "tier1" in outputs:
        _run("misclass_analytical", _stage_misclass_analytical)

    def _stage_defense_misclass():
        # Symmetric FNR sensitivity on the DEFENSE side. DefenseFinder recall
        # varies by taxon; the pipeline previously modelled plasmid-detection
        # false negatives only, and there is no reason the 2x2 correction
        # should be applied to one variable and not the other.
        outputs["defense_misclass_analytical"] = \
            tier3_misclassification.defense_side_bias_correction(
                outputs.get("tier2_phyloglm", pd.DataFrame()),
                outputs.get("tier1", pd.DataFrame()), cfg)
    if "tier2_phyloglm" in outputs and cfg.run_defense_misclassification:
        _run("defense_misclass", _stage_defense_misclass)

    # --------------------------------------------------------------
    # A4 — pre-registered entry-mode (ssDNA) prediction
    # --------------------------------------------------------------
    def _stage_entry_mode():
        comp = tier3_entry_mode.run_entry_mode_composition(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger,
            workdir)
        outputs["entry_mode_composition"] = comp
        conf = tier3_entry_mode.run_entry_mode_confirmatory(comp, cfg, logger)
        if conf is not None and not conf.empty:
            outputs["entry_mode_confirmatory"] = conf
        sec = tier3_entry_mode.run_entry_mode_binary_secondary(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger,
            workdir)
        if sec is not None and not sec.empty:
            outputs["entry_mode_binary"] = sec
    if phylo_data is not None and "n_plasmids_entrymode" in phylo_data.columns:
        _run("entry_mode", _stage_entry_mode)

    # --------------------------------------------------------------
    # B1 — phylogenetically matched sister pairs
    # --------------------------------------------------------------
    def _stage_sister_pairs():
        sp = tier3_sister_pairs.run_sister_pairs(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger,
            workdir)
        outputs["sister_pair_summary"] = sp["summary"]
        if not sp["pairs"].empty:
            outputs["sister_pair_detail"] = sp["pairs"]
        if "tier2_phyloglm" in outputs:
            cmp_ = tier3_sister_pairs.compare_sister_to_primary(
                sp["summary"], outputs["tier2_phyloglm"], cfg)
            if cmp_ is not None and not cmp_.empty:
                outputs["sister_vs_primary"] = cmp_
    if phylo_data is not None and cfg.run_sister_pairs:
        _run("sister_pairs", _stage_sister_pairs)

    # --------------------------------------------------------------
    # B3 — matched-feature negative control
    # --------------------------------------------------------------
    def _stage_feature_control():
        fc = tier3_feature_control.run_feature_control(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger,
            workdir)
        outputs["feature_control_results"] = fc["results"]
        if "tier2_phyloglm" in outputs and not fc["results"].empty:
            cmp_ = tier3_feature_control.build_feature_control_comparison(
                fc["results"], outputs["tier2_phyloglm"], cfg, logger)
            if cmp_ is not None and not cmp_.empty:
                outputs["feature_control_comparison"] = cmp_
    if phylo_data is not None and cfg.run_feature_control:
        _run("feature_control", _stage_feature_control)

    # --------------------------------------------------------------
    # Consensus across phylogenetic methods — one consensus table per
    # outcome stratum, primary direction only.
    # --------------------------------------------------------------
    def _stage_consensus():
        outputs["consensus"] = consensus_mod.build_consensus_by_outcome(
            outputs.get("tier2_phyloglm", pd.DataFrame()),
            outputs.get("tier2_pagels", pd.DataFrame()),
            outputs.get("tier2_pglmm_mv", pd.DataFrame()),
            outcome_spec=outcome_spec,
            config=cfg,
        )
        # Global FDR across all primary tests. config.report_global_fdr has
        # always been True and stats_utils.apply_global_fdr has always
        # existed, but nothing called it — correction was per-stratum only.
        if "tier2_phyloglm" in outputs:
            outputs["tier2_phyloglm"] = reporting.attach_global_fdr(
                outputs["tier2_phyloglm"], cfg)
            # B4: E-values for unmeasured confounding. Uses the sqrt(OR)
            # common-outcome correction — plasmid carriage is common here, so
            # treating the OR as a risk ratio would inflate every E-value.
            if cfg.run_evalues:
                # Decide the OR-to-RR conversion from the OBSERVED outcome
                # prevalence rather than a constant. Hard-coding it is how this
                # went wrong before: the config asserted the outcome was common
                # when it is 5.7%.
                obs_prev = None
                if phylo_data is not None and \
                        "has_plasmid_binary" in phylo_data.columns:
                    obs_prev = float(phylo_data["has_plasmid_binary"].mean())
                rare = stats_utils.resolve_rare_outcome(
                    cfg.evalue_rare_outcome, obs_prev,
                    cfg.evalue_rare_outcome_threshold, logger)
                outputs["tier2_phyloglm"] = stats_utils.attach_evalues(
                    outputs["tier2_phyloglm"], rare_outcome=rare,
                    outcome_prevalence=obs_prev)
        # Binary-vs-binomial concordance for the stratified outcomes.
        bc = reporting.build_binomial_concordance(
            outputs.get("tier2_pglmm_mv", pd.DataFrame()), cfg)
        if bc is not None and not bc.empty:
            outputs["binomial_concordance"] = bc
        # Covariate-impact comparison rides alongside consensus.
        ci = consensus_mod.build_covariate_impact(
            outputs.get("tier2_phyloglm", pd.DataFrame()))
        if not ci.empty:
            outputs["covariate_impact"] = ci
    _run("consensus", _stage_consensus)

    def _stage_phylo_vs_nonphylo():
        outputs["phylo_vs_nonphylo"] = reporting.build_phylo_vs_nonphylo_comparison(
            outputs.get("tier1", pd.DataFrame()),
            outputs.get("tier2_phyloglm", pd.DataFrame()),
        )
    _run("phylo_vs_nonphylo", _stage_phylo_vs_nonphylo)

    # --------------------------------------------------------------
    # Persist (final sweep): individual stage outputs were already
    # checkpointed as they ran; save_all here is a belt-and-braces
    # rewrite of any output a stage forgot to checkpoint, plus the
    # combined / per-outcome aggregate tables that don't exist until
    # every stage has finished.
    # --------------------------------------------------------------
    reporting.save_all(outputs, out_dir)
    # Combined table (any_plasmid primary direction only, for backward
    # compatibility). Stash into outputs so key-findings plots can filter it.
    combined = reporting.build_combined_results(outputs)
    if combined is not None and not combined.empty:
        combined.to_csv(out_dir / "combined_all_results.tsv", sep="\t", index=False)
        outputs["combined"] = combined
    # Cross-stratum summary: (defense_system, outcome_label) with the
    # coefficients from every phylogenetic method that ran.
    per_outcome = reporting.build_per_outcome_summary(outputs)
    if per_outcome is not None and not per_outcome.empty:
        per_outcome.to_csv(out_dir / "per_outcome_summary.tsv", sep="\t", index=False)
        outputs["per_outcome_summary"] = per_outcome
    reporting.write_summary_report(outputs, out_dir, alpha=cfg.alpha)

    # Figures — pass the species-level tables so descriptive figures work.
    # Existing plot code expects single-outcome shapes for tier1 /
    # tier2_phyloglm / consensus; give it views filtered to the legacy
    # any_plasmid outcome + primary direction + binary outcome-mode.
    if "figures" in stages:
        plotting_outputs = dict(outputs)
        plotting_outputs["tier1"] = reporting._filter_primary_any_plasmid(
            outputs.get("tier1"))
        plotting_outputs["tier2_phyloglm"] = reporting._filter_primary_any_plasmid(
            outputs.get("tier2_phyloglm"))
        plotting_outputs["tier2_pagels"] = reporting._filter_primary_any_plasmid(
            outputs.get("tier2_pagels"))
        plotting_outputs["tier2_pglmm_mv"] = reporting._filter_primary_any_plasmid(
            outputs.get("tier2_pglmm_mv"))
        plotting_outputs["consensus"] = reporting._filter_primary_any_plasmid(
            outputs.get("consensus"))
        plotting.make_all_figures(plotting_outputs, out_dir / "figures", logger,
                                  binary_df=binary_df,
                                  prevalence_df=prevalence_df,
                                  defense_cols=defense_cols)

    logger.info(f"Pipeline completed in {(time.time() - t_start) / 60:.1f} min")
    return outputs


def main(argv=None):
    parser = build_parser()
    ns = parser.parse_args(argv)

    cfg = config.Config()
    cfg = apply_cli_to_config(cfg, ns)

    if getattr(ns, "subsample", None):
        cfg = replace(cfg, subsample_species=int(ns.subsample))

    if getattr(ns, "estimate_cost", False):
        from .cost_model import estimate_pipeline_cost, format_cost_report
        n_jobs = cfg.n_jobs if cfg.n_jobs and cfg.n_jobs > 0 else 20
        df = estimate_pipeline_cost(
            cfg, n_species=ns.estimate_species, n_defense=ns.estimate_systems,
            n_outcomes=ns.estimate_outcomes, n_jobs=n_jobs,
            stages=list(cfg.stages) or None)
        print(format_cost_report(df, n_jobs))
        return {"cost_estimate": df}

    force_rerun = set(ns.force_rerun or [])
    # Validate that every supplied force-rerun name is a real stage (or
    # the special 'all' sentinel); otherwise the user typically expects a
    # rerun and gets silent caching, which is a frustrating debugging trap.
    bad = [s for s in force_rerun if s != "all" and s not in ALL_STAGES]
    if bad:
        parser.error(
            f"--force-rerun: unknown stage(s) {bad}. "
            f"Valid choices: {', '.join(ALL_STAGES)}, all"
        )

    results: Dict[str, Dict[str, object]] = {}
    if ns.granularity in ("subtype_level", "both"):
        results["subtype_level"] = run_pipeline(
            cfg.input_file, cfg, "subtype_level", force_rerun=force_rerun)
    if ns.granularity in ("type_level", "both"):
        results["type_level"] = run_pipeline(
            cfg.input_file_type_level, cfg, "type_level", force_rerun=force_rerun)
    return results


if __name__ == "__main__":
    main()
