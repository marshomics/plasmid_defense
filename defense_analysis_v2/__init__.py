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
    - A NATURAL-SPLINE BASIS on log(n_strains) is a default covariate on every
      phylogenetic fit. Sampling depth is a common cause of the predictor and
      the outcome: the plasmid label is species-propagated ("any strain carries
      one") and the defense call is max() across strains, so both behave as
      1-(1-p)^n. A single linear log(n) term is not sufficient because the
      logit of that saturation is not linear in log(n) and its curvature
      depends on p. See ``confound_sim.py``.
    - A NEGATIVE CONTROL permutes the plasmid label within joint
      (clade x depth-decile) strata and re-runs the primary sweep. If hits
      survive that, the pipeline is measuring sequencing effort. This is the
      first stage and the first section of the summary report.
    - Depth sensitivity runs in BOTH directions: the high-depth filter that
      reviewers expect (which enriches for the artefact) and the low-depth
      complement (where saturation cannot have operated). Their concordance is
      the deliverable, not either band's hit count.
    - The permutation null is stratified on clade AND depth. Stratifying on
      clade alone destroys the depth-outcome covariance while preserving the
      depth-predictor covariance, which makes the test anticonservative for
      exactly the systems most contaminated by sampling depth.
    - Pagel's test draws multiple subsamples and combines them with the CAUCHY
      combination. The median of k p-values is not a p-value: under H0 with
      k = 5 it is Beta(3,3), costing roughly 55x in power and yielding
      uninterpretable q-values.
    - Rank-product consensus is CALIBRATED against a permutation null drawn
      per number of contributing methods, so a system ranked #1 by one method
      no longer outranks one ranked #2 by three.
    - Model sensitivity sweeps the ESTIMATOR (MPLE vs IG10) and the COVARIANCE
      STRUCTURE (Pagel's-lambda-rescaled trees). ``phyloglm``'s ``method``
      argument selects the estimator, not an evolutionary process, so the
      previous "OU" arm was bit-identical to the primary fit.
    - Heterogeneity is tested on WITHIN-clade fits, which are disjoint and
      therefore independent. Leave-one-clade-out estimates share >90% of their
      data, so Cochran's Q against chi2(k-1) is invalid on them; LOCO is kept
      as an influence diagnostic with no p-value attached.
    - Degenerate fits FORFEIT their p-values. phyloglm boundary hits,
      separation, non-finite standard errors, and non-converged PGLMM fits all
      surrender the p-value so they cannot enter FDR, consensus, or a figure.
    - FDR is applied per stratum AND globally across all primary tests, with
      the global family restricted to the pre-declared primary outcomes.

Beyond the association screen, four analyses address mechanism and causal
ordering:

    - ENTRY-MODE PREDICTION (A4): conjugative plasmids enter as ssDNA and
      restriction-like systems cleave dsDNA, so dsDNA-restricting systems
      should deplete non-conjugative plasmids preferentially. Tested as a
      single pre-registered contrast on a partition declared in advance, using
      a within-species composition outcome that differences out every
      species-level property including sequencing depth.
    - SISTER PAIRS (B1): within-pair contrasts among depth-matched sister
      species. Controls phylogeny and depth by construction, not by model.
    - DIRECTIONALITY (B2): Pagel dependent-transition models under
      dep.var = x / y / xy, compared by AIC. Distinguishes "defense drives
      plasmid" from "plasmid drives defense".
    - MATCHED-FEATURE CONTROL (B3): simulated traits matched on prevalence and
      phylogenetic clustering calibrate the effect-size scale.

E-values quantify how strong an unmeasured confounder would have to be to
explain each primary association away.

The driver is ``defense_plasmid_analysis.py`` inside this package.
Regression tests live in ``tests/test_pipeline_fixes.py`` and
``tests/test_analysis_extensions.py``.
"""

__version__ = "3.2.0"
__all__ = ["config", "logging_setup", "io_utils", "tree_utils", "stats_utils",
           "r_bridge", "tier1", "tier2_phylo_uni", "tier2_pagels",
           "tier2_multivariate", "tier3_burden", "tier3_loco",
           "tier3_misclassification", "tier3_sensitivity",
           "tier3_entry_mode", "tier3_sister_pairs", "tier3_feature_control",
           "phylo_signal_fast", "cost_model", "consensus",
           "reporting", "plotting"]
