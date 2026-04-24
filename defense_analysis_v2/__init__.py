"""
defense_analysis_v2
===================

Phylogenetically-corrected comparative analysis of bacterial anti-phage
defense systems vs. plasmid carriage.

This package replaces the single-file pipeline (defense_plasmid_analysis.py)
plus the exploratory scripts under new_scripts/. The goal is a pipeline whose
every primary claim is defensible under peer review:

    - All association tests that appear in the main results are phylogenetically
      corrected. Plain (non-phylogenetic) Fisher / Mann-Whitney / logistic
      regression are retained only as diagnostics inside Tier 1 and never
      elevated as primary evidence.
    - Multivariate "independent effect" claims use a phylogenetic generalised
      linear mixed model (phyr::pglmm) with a Brownian-motion covariance
      structure declared explicitly, not plain multivariate logistic regression.
    - Regularised feature selection (LASSO / Elastic Net) operates on
      phylogenetically-decorrelated residuals, uses lambda chosen by the
      one-standard-error rule for stability, and is stability-checked by
      subsample replication.
    - Leave-one-clade-out robustness uses Cochran's Q test for heterogeneity
      across clades, at GTDB class level (finer than phylum) with a phylum-level
      fallback for robustness. No hand-picked CV threshold.
    - Defense-burden-vs-plasmid is tested with phylogenetic generalised least
      squares on the count, and phylogenetic logistic regression on presence,
      after estimating Pagel's lambda for the count. No raw Mann-Whitney.
    - Sparse-table tests use Firth's penalised logistic regression to control
      the small-sample and separation bias of ordinary logistic regression.
    - Misclassification sensitivity analysis (Monte Carlo + analytical bias
      correction) quantifies how much the primary phylogenetic result depends
      on the assumed plasmid-detection false-negative rate.
    - ``log(n_strains)`` is a default covariate on every phylogenetic fit so
      the species-level sampling-depth saturation of max()-aggregated binary
      defense features is partialled out, not left as a latent confounder.
    - Sampling-depth and feature-mode sensitivity reruns (species filtered to
      n_strains >= threshold; refit using mean-across-strains prevalence as
      the defense feature) confirm the primary conclusions aren't sampling
      artefacts.
    - Pagel's test draws multiple independent subsamples and reports the
      median p-value plus the fraction of subsamples significant, rather
      than a single noisy draw.
    - Rank-product consensus averages over methods that actually ranked each
      system, instead of silently penalising systems that one method skipped.
    - Evolutionary-model sensitivity reruns phyloglm under OU and BM+lambda
      so primary claims don't quietly depend on the Brownian assumption.
    - PGLMM fits return an explicit convergence code; non-converged fits are
      flagged at summary time rather than silently reported as results.

The driver is ``defense_plasmid_analysis.py`` inside this package.
"""

__version__ = "2.1.0"
__all__ = ["config", "logging_setup", "io_utils", "tree_utils", "stats_utils",
           "r_bridge", "tier1", "tier2_phylo_uni", "tier2_pagels",
           "tier2_multivariate", "tier3_burden", "tier3_loco",
           "tier3_misclassification", "tier3_sensitivity", "consensus",
           "reporting", "plotting"]
