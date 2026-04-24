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
from typing import Dict

import pandas as pd

# Support both "python -m defense_analysis_v2.defense_plasmid_analysis" and
# "python defense_analysis_v2/defense_plasmid_analysis.py"
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from defense_analysis_v2 import (
        config, logging_setup, io_utils, tree_utils, r_bridge,
        tier1, tier2_phylo_uni, tier2_pagels, tier2_multivariate,
        tier2_random_forest, tier3_burden, tier3_loco,
        tier3_misclassification, tier3_sensitivity,
        consensus as consensus_mod, reporting, plotting,
    )
else:
    from . import (config, logging_setup, io_utils, tree_utils, r_bridge,
                   tier1, tier2_phylo_uni, tier2_pagels, tier2_multivariate,
                   tier2_random_forest, tier3_burden, tier3_loco,
                   tier3_misclassification, tier3_sensitivity)
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
    "tier1", "phyloglm", "pagels", "pglmm_mv", "lasso", "rf", "burden",
    "loco", "phylo_signal", "clade_perm", "prev_match", "misclass_mc",
    "misclass_analytical",
    # Sampling-depth and feature-mode sensitivity reruns of the primary
    # phyloglm — guard against max()-aggregation saturation of binary
    # defense features in heavily-sampled species.
    "min_n_strains_sens", "prev_feature_sens",
    # Phylogenetic-model sensitivity under alternative evolutionary models.
    "phylo_model_sens",
    "consensus", "phylo_vs_nonphylo", "figures",
]

DEFAULT_STAGES = [
    # Statistical stages in dependency order. `figures` is run last.
    "tier1", "phyloglm", "pagels", "pglmm_mv", "lasso", "rf", "burden",
    "loco", "phylo_signal", "clade_perm", "prev_match", "misclass_mc",
    "misclass_analytical",
    "min_n_strains_sens", "prev_feature_sens", "phylo_model_sens",
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
                 granularity_label: str) -> Dict[str, object]:
    """Run the pipeline for a single granularity. Returns a dict of all
    result DataFrames keyed by the names used in reporting/plotting.
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
    t_start = time.time()

    stages = cfg.stages or tuple(DEFAULT_STAGES)

    # R availability check — only enforced if any stage that uses R is scheduled
    r_stages = {"phyloglm", "pagels", "pglmm_mv", "lasso", "burden", "loco",
                "phylo_signal", "misclass_mc",
                "min_n_strains_sens", "prev_feature_sens", "phylo_model_sens"}
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

    outputs: Dict[str, object] = {}

    # --------------------------------------------------------------
    # Tier 1 (non-phylogenetic baseline, diagnostic only)
    # --------------------------------------------------------------
    if "tier1" in stages:
        outputs["tier1"] = tier1.run_tier1(binary_df, prevalence_df,
                                           defense_cols, cfg, logger,
                                           outcome_spec=outcome_spec)

    # --------------------------------------------------------------
    # Tier 2
    # --------------------------------------------------------------
    if "phyloglm" in stages and phylo_data is not None:
        outputs["tier2_phyloglm"] = tier2_phylo_uni.run_tier2_phyloglm_univariate(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger, workdir,
            outcome_spec=outcome_spec)
    if "pagels" in stages and phylo_data is not None:
        outputs["tier2_pagels"] = tier2_pagels.run_pagels_test(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger, workdir,
            outcome_spec=outcome_spec)
    if "pglmm_mv" in stages and phylo_data is not None:
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
    elif "lasso" in stages and phylo_data is not None:
        # Standalone LASSO path if user doesn't want PGLMM
        lasso, enet = tier2_multivariate.run_regularised_on_residuals(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger, workdir)
        outputs["lasso"] = lasso
        outputs["elastic_net"] = enet

    # --------------------------------------------------------------
    # Clade-blocked Random Forest (not R-dependent)
    # --------------------------------------------------------------
    if "rf" in stages:
        try:
            rf_res = tier2_random_forest.run_clade_blocked_rf(
                binary_df, prevalence_df, defense_cols, cfg, logger,
                clade_rank="gtdb_class")
            outputs["rf_binary"] = rf_res.binary
            outputs["rf_prevalence"] = rf_res.prevalence
            outputs["rf_fold_aucs"] = rf_res.fold_aucs
        except Exception as e:
            logger.warning(f"Clade-blocked RF failed: {e}")

    # --------------------------------------------------------------
    # Tier 3
    # --------------------------------------------------------------
    if "burden" in stages and phylo_data is not None:
        outputs["burden_pgls"] = tier3_burden.run_burden_pgls(
            phylo_data, str(pruned_tree_path), cfg, logger, workdir)
        outputs["burden_phyloglm"] = tier3_burden.run_burden_phyloglm(
            phylo_data, str(pruned_tree_path), cfg, logger, workdir)

    if "loco" in stages and phylo_data is not None and "tier2_phyloglm" in outputs:
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

    if "phylo_signal" in stages and phylo_data is not None:
        outputs["tier3_phylo_signal"] = tier3_sensitivity.run_phylogenetic_signal(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger, workdir)

    if "clade_perm" in stages:
        outputs["tier3_perm"] = tier3_sensitivity.run_clade_permutation(
            binary_df, defense_cols, cfg, logger)

    if "prev_match" in stages:
        outputs["tier3_prevalence_matched"] = tier3_sensitivity.run_prevalence_matched(
            binary_df, prevalence_df, defense_cols, cfg, logger)

    # --------------------------------------------------------------
    # Sampling-depth and feature-mode sensitivity reruns of the primary
    # phyloglm. These directly target the max()-saturation bias that
    # log_n_strains as a covariate can only partially correct for.
    # --------------------------------------------------------------
    if "min_n_strains_sens" in stages and phylo_data is not None:
        outputs["tier3_min_n_strains_sens"] = tier3_sensitivity.run_min_n_strains_sensitivity(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger, workdir)

    if ("prev_feature_sens" in stages and phylo_data is not None
            and cfg.run_prevalence_feature_sensitivity):
        outputs["tier3_prev_feature_sens"] = tier3_sensitivity.run_prevalence_feature_sensitivity(
            phylo_data, prevalence_df, defense_cols, str(pruned_tree_path),
            cfg, logger, workdir)

    if ("phylo_model_sens" in stages and phylo_data is not None
            and cfg.phylo_model_sensitivity_models):
        outputs["tier3_phylo_model_sens"] = tier3_sensitivity.run_phylo_model_sensitivity(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger, workdir)

    # --------------------------------------------------------------
    # Misclassification sensitivity
    # --------------------------------------------------------------
    if "misclass_mc" in stages and phylo_data is not None:
        mc_long = tier3_misclassification.run_misclassification_mc(
            phylo_data, defense_cols, str(pruned_tree_path), cfg, logger, workdir)
        outputs["misclass_mc_long"] = mc_long
        outputs["misclass_summary"] = tier3_misclassification.summarise_misclassification_mc(
            mc_long, cfg)

    if "misclass_analytical" in stages and \
            "tier2_phyloglm" in outputs and "tier1" in outputs:
        outputs["misclass_analytical"] = tier3_misclassification.analytical_bias_correction(
            outputs["tier2_phyloglm"], outputs["tier1"],
            cfg.misclass_fnr_grid, cfg)
        outputs["misclass_analytical_summary"] = outputs["misclass_analytical"]

    # --------------------------------------------------------------
    # Consensus across phylogenetic methods — one consensus table per
    # outcome stratum, primary direction only.
    # --------------------------------------------------------------
    if "consensus" in stages:
        outputs["consensus"] = consensus_mod.build_consensus_by_outcome(
            outputs.get("tier2_phyloglm", pd.DataFrame()),
            outputs.get("tier2_pagels", pd.DataFrame()),
            outputs.get("tier2_pglmm_mv", pd.DataFrame()),
            outcome_spec=outcome_spec,
        )

    if "phylo_vs_nonphylo" in stages:
        outputs["phylo_vs_nonphylo"] = reporting.build_phylo_vs_nonphylo_comparison(
            outputs.get("tier1", pd.DataFrame()),
            outputs.get("tier2_phyloglm", pd.DataFrame()),
        )

    # Covariate-impact comparison (with_cov vs without_cov, primary
    # direction). Inherits from the consensus stage so it runs whenever
    # consensus does.
    if "consensus" in stages:
        ci = consensus_mod.build_covariate_impact(
            outputs.get("tier2_phyloglm", pd.DataFrame()))
        if not ci.empty:
            outputs["covariate_impact"] = ci

    # --------------------------------------------------------------
    # Persist
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

    results: Dict[str, Dict[str, object]] = {}
    if ns.granularity in ("subtype_level", "both"):
        results["subtype_level"] = run_pipeline(cfg.input_file, cfg, "subtype_level")
    if ns.granularity in ("type_level", "both"):
        results["type_level"] = run_pipeline(cfg.input_file_type_level, cfg, "type_level")
    return results


if __name__ == "__main__":
    main()
