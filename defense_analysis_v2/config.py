"""Configuration dataclass for the pipeline.

Sampling-depth policy (read this before changing anything below)
----------------------------------------------------------------
The species-level plasmid label is propagated upstream ("any strain in the
species carries a plasmid") and the species-level defense call is max() across
strains. Both are therefore ``1 - (1-p)^n_strains`` in expectation, so
sequencing depth is a *common cause* of the predictor and the outcome. Left
uncorrected this manufactures an odds ratio of ~2.5 with a 100% false-positive
rate under a strict null (see ``confound_sim.py`` at the repo root).

A single linear ``log(n_strains)`` term is not sufficient: the logit of
``1 - (1-p)^n`` is not linear in ``log n``, and its curvature depends on p, so
a rare defense system and common plasmid carriage saturate on different
curves. Under clade-structured depth -- which is what GTDB actually looks
like -- one linear term leaves a 40% false-positive rate.

The pipeline therefore adjusts for depth with a *natural cubic spline basis*
on ``log(n_strains)`` (``depth_spline_df`` knots), included in every
phylogenetic model. The basis columns are built once in ``io_utils`` and
passed to R as ordinary covariates.

``covariate_modes`` no longer contains an unadjusted arm as a co-equal
alternative. The unadjusted fit is retained only as an explicitly-labelled
positive control for the confound (``unadjusted``), and
``primary_covariate_mode`` names the one mode that primary claims may cite.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional


@dataclass
class Config:
    """Pipeline configuration.

    All file paths default to the production locations used previously; override
    via CLI flags (see ``defense_plasmid_analysis.py``) or by constructing
    Config directly in notebooks.
    """

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    input_file: str = (
        "/ebio/abt3_projects2/Gut_genetics2/data/defensefinder/all_combined/"
        "defense_finder_human_animal_free_combined_reshaped_nodefenseincluded_hasplasmid.txt"
    )
    input_file_type_level: str = (
        "/ebio/abt3_projects2/Gut_genetics2/data/defensefinder/all_combined/"
        "defense_finder_human_animal_free_combined_reshaped_type_nodefenseincluded_hasplasmid.txt"
    )
    tree_file: str = (
        "/ebio/abt3_scratch/jmarsh/tract_score3/gtdb_custom_trees/"
        "human_animal_free_90percent_species_level/output/"
        "gtdbtk.rooted.speciesnames.tree"
    )

    # Plasmid metadata (per-plasmid rows). Joined to species via gtdb_species.
    # Used to stratify the plasmid outcome into mobility / replicon / size
    # classes and to compute per-species plasmid-count weights. Treats "-" and
    # blank as missing.
    plasmid_metadata_file: str = (
        "/ebio/abt3_projects2/Gut_genetics/data/plasmids/"
        "total_plasmid_metadata_duplicatesnoted_dupsdeleted_plasmidids.hostrange_"
        "merged_mobsuite_txsscan_conjscan_updated_flagellins_conj_pilv_rci_both_"
        "shufflons_nodups_withgut_large_removed_environs.txt"
    )

    # Genome-scale covariates per strain/assembly. Joined to the strain table
    # via the genome id column. Aggregated to species-level means before being
    # handed to the phylogenetic models.
    genome_covariates_file: str = (
        "/ebio/abt3_projects2/Gut_genetics2/data/total_metadata_qc_bbmap.txt"
    )
    genome_covariates_key: str = "genome"           # column in the covariates file keyed to our `genome` id
    genome_covariate_columns: Tuple[str, ...] = (
        "corrected_genome_size", "gc_avg", "cds_number",
    )

    output_dir: str = "/ebio/abt3_projects2/Gut_genetics2/data/defensefinder/plasmid_vs_defense_v2"

    granularities: Tuple[Tuple[str, str], ...] = (
        ("subtype_level", "input_file"),
        ("type_level", "input_file_type_level"),
    )

    # ------------------------------------------------------------------
    # Plasmid stratification
    # ------------------------------------------------------------------
    # Which mob_suite-style columns to parse. `-` and blanks are treated as
    # missing. A species is assigned `no_plasmids` if it appears in no rows of
    # the plasmid metadata table; otherwise its plasmid rows are tabulated.
    plasmid_mobility_column: str = "predicted_mobility_updated"
    plasmid_reptype_column: str = "rep_type(s)"
    plasmid_size_column: str = "size"

    # Top-N replicon categories to carry through as parallel outcomes. Replicon
    # labels come as semicolon-separated lists; we split and count each label
    # separately. Categories with fewer than `min_rep_type_species` species
    # having at least one plasmid of that category are dropped.
    # Raised from 25 to 50: replicon types supported by 25-50 species carry
    # very wide confidence intervals and consume FDR budget for strata that
    # cannot support a primary claim. Stratified replicon results are an
    # exploratory companion, not a headline claim (see
    # ``primary_outcome_labels``).
    top_n_rep_types: int = 10
    min_rep_type_species: int = 50

    # Size-class bin edges in bp. A plasmid is small if size < bins[0], medium
    # if bins[0] <= size < bins[1], large if size >= bins[1]. Defaults are the
    # conventional mob_suite bins.
    plasmid_size_bins_bp: Tuple[int, int] = (20_000, 100_000)

    # Primary outcome modelling mode per stratified class:
    #   "binary"   - species has at least one plasmid of class X, with a
    #                spline basis on log(n_plasmids) as covariate to defuse the
    #                "species with many plasmids has every type" saturation.
    #   "fraction" - fraction-of-species-plasmids that fall in class X, fit as
    #                cbind(k, n-k). Only ``phyr::pglmm`` can fit this;
    #                ``phylolm::phyloglm`` has no binomial mode, so Pagel's and
    #                the univariate tier cannot participate.
    #
    # "binary" is primary because it is the only mode all three consensus
    # methods can fit, so consensus is computed on comparable estimands. The
    # binomial fit runs alongside as a required concordance check and is
    # surfaced by ``reporting.build_binomial_concordance``; a stratified claim
    # is only reportable when the two modes agree in sign.
    plasmid_stratified_primary_mode: str = "binary"

    # Outcome labels a primary claim may be made about. Everything else runs
    # and is reported, but is tagged exploratory and is excluded from the
    # global FDR family (``stats_utils.apply_global_fdr``). Declare these
    # before looking at results.
    primary_outcome_labels: Tuple[str, ...] = (
        "any_plasmid", "conjugative", "mobilizable", "non-mobilizable",
    )

    # Also run the legacy "any plasmid vs none" outcome for backward
    # comparability. This is saturated at species level (most species end up
    # labelled as plasmid-carriers), but reviewers will ask for it.
    include_any_plasmid_outcome: bool = True

    # ------------------------------------------------------------------
    # Significance / multiple-testing
    # ------------------------------------------------------------------
    alpha: float = 0.05
    fdr_method: str = "fdr_bh"     # Benjamini-Hochberg

    # Apply a single global FDR across all primary tests, on top of the
    # per-stratum FDR. Implemented in ``stats_utils.apply_global_fdr`` and
    # wired in ``reporting.attach_global_fdr``; emits ``*_global_qvalue``.
    #
    # This matters: the univariate sweep is ~435 systems x |outcome strata| x
    # 2 directions x |covariate_modes| fits. Per-stratum BH controls the error
    # rate *within* a stratum only, which does not cover a narrative that
    # highlights whatever reached significance somewhere. The global family is
    # restricted to ``primary_outcome_labels`` x ``primary_covariate_mode`` so
    # exploratory strata do not dilute it.
    report_global_fdr: bool = True

    # Decision criterion for headline claims. "global" requires the global
    # q-value; "per_stratum" reverts to the legacy behaviour.
    primary_decision_qvalue: str = "global"

    # ------------------------------------------------------------------
    # Sample-size gates
    # ------------------------------------------------------------------
    # Minimum number of species carrying (or not carrying) a given defense
    # system to include that system in phylogenetic tests. Below this, the
    # phyloglm estimate is unstable regardless of significance.
    min_count_per_category: int = 10

    # Minimum prevalence for inclusion in the multivariate PGLMM. Rare systems
    # inflate variance without adding power; 10% is the same threshold the old
    # new_scripts/multivariate_analysis.py used, retained here for comparability.
    min_prevalence_multivariate: float = 0.10

    # Per-fit wall-clock timeout for the multivariate PGLMM, in hours.
    # phyr::pglmm with a phylogenetic random effect on N tips runs PQL
    # iterations over an O(N^2)-dense covariance structure; at ~40k tips
    # individual fits take many hours and the original 2h hard-coded
    # timeout was unrealistic. 48 covers the long tail at this scale.
    pglmm_timeout_hours: int = 48

    # Cap on tip count for PGLMM only (other phylogenetic stages use the full
    # tree). Each PGLMM fit gets a random subsample of this size, stratified by
    # ``permutation_clade_rank`` and by n_strains decile so the subsample does
    # not distort the depth distribution the spline has to fit.
    #
    # Default is 15_000, not None: docs/pglmm_step_recommendations.md records
    # that the full ~40k-tip fit needs 175-250 GB and does not complete on the
    # available hardware, so None shipped a default configuration that cannot
    # run. The random-effect cost drops as ~N^2.
    pglmm_max_species: Optional[int] = 15_000

    # Minimum species count in the LEFT-OUT clade for that clade to contribute
    # to the heterogeneity test. Clades below this are still fit and reported
    # (they are useful influence diagnostics) but are excluded from Cochran's
    # Q, because dropping 3 species from 40,000 returns a near-duplicate of the
    # full-data estimate and deflates Q.
    min_species_per_loco_clade: int = 50

    # Heterogeneity across clades is assessed on WITHIN-clade fits, not on
    # leave-one-clade-out fits. LOCO estimates share >90% of their data, so
    # they are not independent and Cochran's Q against chi2(k-1) is grossly
    # conservative. Within-clade fits are disjoint by construction, so Q is
    # valid. LOCO is retained as an influence diagnostic reporting the
    # coefficient shift, with no p-value attached.
    min_species_per_within_clade_fit: int = 200

    # ------------------------------------------------------------------
    # Clade / stratification choices
    # ------------------------------------------------------------------
    # GTDB rank for the clade-restricted permutation null.
    #
    # Class, not phylum. The permutation must preserve the structure that
    # generates the confound, and within-phylum exchangeability at ~40k tips is
    # far too coarse: n_strains varies by orders of magnitude within
    # Pseudomonadota, so shuffling the plasmid label within phylum destroys
    # Cov(plasmid, n_strains) while Cov(defense, n_strains) is preserved,
    # yielding an anticonservative null. See ``permutation_depth_bins``.
    permutation_clade_rank: str = "gtdb_class"

    # The permutation is stratified jointly on clade AND sampling depth. Within
    # each (clade, depth bin) cell the plasmid label is shuffled, so the null
    # preserves the depth-outcome relationship instead of destroying it. This
    # is the single most important correctness fix in the permutation null.
    permutation_depth_bins: int = 10

    # LOCO is run at both class and phylum level. Class is primary; phylum is
    # a fallback sensitivity for taxa where class-level samples are thin.
    loco_ranks: Tuple[str, ...] = ("gtdb_class", "gtdb_phylum")

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------
    n_permutations: int = 1000
    n_bootstrap: int = 100
    cv_folds: int = 10

    # LASSO lambda selection: one-SE rule picks the most-regularised lambda
    # whose CV error is within one SE of the minimum. More conservative than
    # CV-minimum; reduces overfitting with small samples.
    lasso_one_se_rule: bool = True

    # Subsample-stability check for LASSO selection: what fraction of bootstrap
    # subsamples selects each feature. Features selected in < stability_threshold
    # of subsamples are flagged as unstable.
    lasso_stability_threshold: float = 0.60
    lasso_stability_n_subsamples: int = 100
    lasso_stability_subsample_frac: float = 0.75

    # ------------------------------------------------------------------
    # Pagel's test — stability via multiple uniform subsamples
    # ------------------------------------------------------------------
    # Pagel's test is computationally prohibitive at full-tree scale, so each
    # call is subsampled. We take N near-disjoint subsamples and COMBINE the
    # per-subsample p-values with the Cauchy (ACAT) combination, which is a
    # valid p-value under arbitrary dependence.
    #
    # The previous behaviour reported the MEDIAN of the per-subsample p-values
    # and passed it to BH as if it were a p-value. It is not: under H0 with
    # k=5 the median is Beta(3,3), so P(median < 0.05) ~ 0.0012. That is
    # super-uniform, so it did not inflate false positives, but it destroyed
    # power and the resulting q-values had no interpretation. It also put rows
    # with different k (systems skipped in some subsamples) into one BH family
    # with different null distributions. Cauchy combination fixes both.
    #
    # n raised 5 -> 10 because the combination is now valid, so extra draws buy
    # real power rather than shrinking a mis-scaled statistic further.
    pagels_n_subsamples: int = 10
    # Per-subsample tip count. fitPagel scales roughly linearly in tip count
    # for the Felsenstein traversal and roughly linearly in the number of
    # optim() iterations needed to converge — combined, ~quadratic for dense
    # subsample sizes on hard fits. At 1500 tips with ~435 defense systems
    # iterated per R call, individual subsamples have been observed to
    # exceed 48-hour wall-clock and time out. 500 keeps a single subsample
    # tractable (~4-8 hours) while preserving enough power to detect
    # correlated trait evolution at the prevalences this dataset has. If
    # your tree or system list is dramatically smaller, bump this up.
    pagels_subsample_size: int = 500

    # Per-subsample wall-clock timeout in hours. fitPagel iterates over every
    # defense system within one R call and fits two continuous-time Markov
    # models per system, so total cost is roughly:
    #   pagels_subsample_size^2 * n_defense_systems * constant.
    # At pagels_subsample_size=1500 and ~435 systems the per-subsample
    # runtime is typically 4-12 hours; 48 covers the long tail with margin
    # so the test isn't silently dropped for outcomes whose subsample tree
    # happens to be slow to fit. Drop the subsample size to 800-1000 if you
    # need to keep this lower.
    pagels_timeout_hours: int = 48

    # ------------------------------------------------------------------
    # Misclassification sensitivity
    # ------------------------------------------------------------------
    # Assumed plasmid-detection false-negative rates to sweep. Lower is more
    # optimistic. The default range reflects the spread across modern plasmid
    # assemblers on Illumina-only data.
    misclass_fnr_grid: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    misclass_n_replicates: int = 200
    misclass_false_positive_rate: float = 0.0   # Assume no FP; flag in methods

    # Plasmid-detection FNR is NOT non-differential here. The species label is
    # "any strain carries a plasmid", so a species with n strains gets n
    # independent chances to detect one: the effective species-level FNR is
    # fnr^n, which falls steeply in sequencing depth. The Bross (1954)
    # correction assumes non-differential misclassification and is invalid
    # against a depth-differential FNR.
    #
    # When True, the Monte Carlo applies the depth-scaled species-level rate
    # fnr^n_strains instead of a flat fnr, and the analytical correction is
    # computed WITHIN depth strata and pooled, so the non-differential
    # assumption holds conditionally.
    misclass_depth_differential: bool = True
    misclass_depth_bins: int = 5

    # Defense-side false negatives. DefenseFinder recall varies by taxon: a
    # system well characterised in Pseudomonadota may be systematically missed
    # in deeper-branching clades. If that varies with clade-level plasmid
    # carriage the associations are biased, and the pipeline previously modelled
    # this on the plasmid side only. Symmetric analytical correction.
    run_defense_misclassification: bool = True
    defense_fnr_grid: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.15, 0.20)

    # ------------------------------------------------------------------
    # Covariates, bidirectional framing, interactions
    # ------------------------------------------------------------------
    # Covariate modes to iterate over inside every test stage. Each stage runs
    # once per mode and tags output rows with a ``covariate_mode`` column.
    #
    #   full        - genome-scale covariates + depth spline basis. PRIMARY.
    #   depth_only  - depth spline basis only. Isolates how much of an
    #                 association is attributable to genome capacity.
    #   unadjusted  - no covariates at all. This is NOT an alternative model.
    #                 It is a positive control for the sampling-depth
    #                 confound: under a strict null it returns OR ~ 2.5 with a
    #                 100% false-positive rate. It exists so the magnitude of
    #                 the confound is visible in the output, and is excluded
    #                 from consensus, from the global FDR family, and from
    #                 every primary claim.
    covariate_modes: Tuple[str, ...] = ("full", "depth_only", "unadjusted")

    # The one mode a primary claim may cite.
    primary_covariate_mode: str = "full"

    # Modes that are diagnostic only and must never reach a headline number.
    diagnostic_covariate_modes: Tuple[str, ...] = ("unadjusted",)

    # Legacy label aliases so older result tables and CLI invocations keep
    # resolving. Maps old -> new.
    covariate_mode_aliases: Tuple[Tuple[str, str], ...] = (
        ("with_cov", "full"),
        ("without_cov", "unadjusted"),
    )

    # Genome-capacity covariates. ``corrected_genome_size`` and ``cds_number``
    # are heavy-tailed and ARE log-transformed -- in ``io_utils`` on the Python
    # side, not in R. (The previous comment claimed the R layer did this; it
    # did not, it only centred and scaled, so heavy-tailed covariates entered
    # untransformed.) Centring/scaling still happens in R.
    use_genome_covariates: bool = True
    log_transform_covariates: Tuple[str, ...] = (
        "corrected_genome_size", "cds_number",
    )

    # For binary plasmid-class outcomes, include a spline basis on
    # log(n_plasmids_total) so species with thousands of plasmids don't
    # saturate to "has one of every class". Same reasoning as the depth
    # spline: a single linear term cannot span the saturation curve.
    use_plasmid_count_covariate_on_binary: bool = True

    # ------------------------------------------------------------------
    # Sampling-depth adjustment
    # ------------------------------------------------------------------
    # Master switch for the depth adjustment. Turning this off reproduces the
    # legacy no-correction behaviour and is only meaningful as a demonstration
    # of the confound.
    use_n_strains_covariate: bool = True

    # Degrees of freedom for the natural cubic spline basis on
    # log(n_strains). 5 df (4 interior knots at quantiles + boundary) removed
    # the residual bias in simulation where a single linear term left a 28%
    # false-positive rate. Raising this costs little; lowering it to 1
    # reproduces the old linear term.
    depth_spline_df: int = 5

    # Degrees of freedom for the spline on log(n_plasmids), used on binary
    # stratified outcomes. Fewer knots because n_plasmids has a much shorter
    # tail than n_strains.
    plasmid_count_spline_df: int = 3

    # Column-name prefixes for the generated basis columns.
    depth_spline_prefix: str = "depth_ns"
    plasmid_count_spline_prefix: str = "nplasmid_ns"

    # ------------------------------------------------------------------
    # Depth sensitivity reruns
    # ------------------------------------------------------------------
    # The n_strains sensitivity runs in BOTH directions, because filtering to
    # well-sampled species enriches for the artefact rather than removing it:
    # P(has_plasmid = 1) rises monotonically in n_strains, so restricting to
    # the deep tail drives outcome prevalence toward 1 and destroys the
    # contrast (38.5% -> 70.6% in simulation, discarding 71.5% of species).
    #
    #   high_depth_min - the legacy filter, retained because reviewers expect
    #                    it, but now gated on retained outcome variance and
    #                    reported as an attenuation check, not a control.
    #   low_depth_max  - the INFORMATIVE complement: species with at most this
    #                    many strains, where saturation cannot have operated.
    #                    Low power per system, but an interpretable null.
    min_n_strains_sensitivity: int = 5
    max_n_strains_sensitivity: int = 2

    # Refuse to report a depth-filtered rerun with too little outcome contrast
    # to fit.
    #
    # Judged on ABSOLUTE COUNTS, not on a proportion. A proportion band was the
    # original design and it is wrong for this dataset: overall prevalence is
    # 5.7%, and the LOW-depth band (n_strains <= 2, where saturation cannot
    # have operated) sits near 0.9% simply because a species with one strain
    # has one chance to carry a plasmid. That band is the informative half of
    # the depth sensitivity, and a 5% proportion floor would have discarded it
    # despite roughly 200 plasmid-positive species -- ample to fit.
    #
    # What actually breaks a fit is too few species in the MINORITY class, in
    # either direction. That is what these bound.
    depth_sens_min_outcome_count: int = 50
    depth_sens_min_outcome_fraction: float = 0.002   # guard against ~0 contrast
    depth_sens_max_outcome_fraction: float = 0.98    # and against saturation
    # Retained so existing configs and result tables keep resolving; no longer
    # the decision rule.
    depth_sens_outcome_prevalence_bounds: Tuple[float, float] = (0.05, 0.95)

    # Prevalence-feature sensitivity: refit with the per-species *prevalence*
    # (mean across strains) of the defense system instead of the max()-derived
    # binary. This de-saturates the PREDICTOR only; the outcome remains the
    # species-propagated plasmid label, so the depth covariates must stay in
    # the model. (They were previously dropped here on the grounds that
    # prevalence is "already strain-averaged", which misidentified what the
    # covariate was doing -- it stands in for depth as a common cause of both
    # variables, not as a predictor correction.)
    run_prevalence_feature_sensitivity: bool = True

    # Weight prevalence observations by strain count. A species with 1 strain
    # contributes prevalence in {0,1}; one with 500 strains contributes a
    # tightly estimated fraction. Unweighted, both enter identically.
    prevalence_feature_weight_by_depth: bool = True

    # If True, also run the symmetric direction:
    # defense_i_presence ~ plasmid_class + covariates. For each defense
    # system this answers "does carriage of plasmid class X predict having
    # defense system i?". Results are reported alongside the primary direction
    # with an "is_reverse" column so consumers can filter.
    run_bidirectional: bool = True

    # Interaction terms in the multivariate PGLMM. We add pairwise products
    # defense_A * defense_B for the top-K systems by primary-direction phyloglm
    # rank. Keep K small; each interaction burns a degree of freedom and adds
    # collinearity.
    add_multivariate_interactions: bool = True
    n_interaction_systems: int = 8

    # ------------------------------------------------------------------
    # Phylogenetic model choice
    # ------------------------------------------------------------------
    # ESTIMATOR for phyloglm, not an evolutionary model.
    #
    # ``phylolm::phyloglm`` does not offer a BM-vs-OU choice for binary traits:
    # its ``method`` argument selects the estimator (MPLE vs the Ives-Garland
    # penalised variant), and the latent process is fixed by the model. The
    # previous config exposed "BM" and "OUfixedRoot" as if they were
    # alternatives, and ``phyloglm_uni.R`` mapped BOTH to ``logistic_MPLE`` --
    # so the OU arm of the sensitivity analysis was bit-identical to the
    # primary fit and measured nothing.
    #
    #   "MPLE"      -> logistic_MPLE  (primary)
    #   "IG10"      -> logistic_IG10  (Ives & Garland penalised)
    phyloglm_estimator: str = "MPLE"

    # Estimator sensitivity. IG10 genuinely changes the estimator.
    phyloglm_estimator_sensitivity: Tuple[str, ...] = ("IG10",)

    # GENUINE covariance-structure sensitivity, done by rescaling the tree
    # under Pagel's lambda before the fit. lambda = 1 is the untransformed
    # tree; lambda < 1 pulls internal branches toward a star phylogeny, i.e.
    # weakens the assumed phylogenetic covariance. Defense systems and plasmids
    # move horizontally, so BM covariance is a simplifying assumption and this
    # is the axis reviewers will push on. Empty tuple disables.
    phylo_lambda_sensitivity: Tuple[float, ...] = (0.5, 0.25)

    # Retained under its old name so existing --stages invocations and result
    # tables keep resolving; now drives the estimator + lambda sensitivities.
    phylo_model_sensitivity_models: Tuple[str, ...] = ("IG10",)

    # ==================================================================
    # A4 — pre-registered entry-mode (ssDNA) prediction
    # ==================================================================
    # Conjugative plasmids enter the recipient as SINGLE-STRANDED DNA; Type II
    # restriction endonucleases and most restriction-like systems cleave
    # DOUBLE-stranded DNA. Non-conjugative plasmids arriving by transformation
    # enter as dsDNA. The mechanistic prediction is therefore:
    #
    #   dsDNA-restricting systems should exclude NON-CONJUGATIVE plasmids more
    #   strongly than conjugative ones; abortive-infection and nucleotide-
    #   signalling systems carry no such prediction.
    #
    # This is a CONFIRMATORY test on a pre-declared partition of the defense
    # systems, not an exploratory screen. Edit the partition below BEFORE
    # looking at any entry-mode result; the whole inferential value of the
    # analysis comes from having fixed it in advance.
    entry_mode_metadata_file: str = (
        "/ebio/abt3_scratch/atyakht_plasmid_db/plasmid_metadata.txt"
    )
    entry_mode_plasmid_id_column: str = "plasmid_id"
    entry_mode_conjugative_column: str = "conjugative"   # yes / no
    # Column carrying the host species in the entry-mode table. If absent, the
    # table is joined to the main plasmid metadata on plasmid_id to recover it.
    entry_mode_species_column: str = "gtdb_species"
    # Column holding plasmid ids in the MAIN plasmid metadata table, used for
    # the fallback join above.
    plasmid_id_column: str = "plasmid_id"

    # Engine for the within-species composition model.
    #   "pglmm" - binomial phyr::pglmm, one univariate fit per system. The
    #             defensible choice: an actual binomial likelihood.
    #   "pgls"  - empirical-logit PGLS with inverse-variance weights. Much
    #             cheaper; use only if the PGLMM sweep will not complete.
    entry_mode_engine: str = "pglmm"

    # Minimum plasmids per species to contribute to the composition model. A
    # species with a single plasmid carries almost no compositional
    # information and inflates the noise.
    entry_mode_min_plasmids_per_species: int = 3

    # Pre-declared mechanism partition, by defense-system CATEGORY as returned
    # by taxonomy.classify_defense_system.
    #
    # PREDICTED: systems that restrict incoming double-stranded DNA through
    # self/non-self discrimination, where the transient ssDNA intermediate of
    # conjugative transfer provides documented evasion.
    entry_mode_predicted_categories: Tuple[str, ...] = (
        "Restriction-Modification",   # incl. Type I/II/III
        "Type-IV-Restriction",        # modification-dependent (McrBC-like)
        "BREX",                       # methylation-based, acts pre-replication
        "DISARM",                     # methylation-based
        "Wadjet",                     # recognises circular dsDNA; plasmid-specific
        "Dnd",                        # phosphorothioation, RM-like logic
    )
    # NOT PREDICTED: abortive-infection and nucleotide-signalling systems,
    # which sense infection rather than cleave incoming dsDNA, so entry mode
    # should not modulate their effect.
    entry_mode_not_predicted_categories: Tuple[str, ...] = (
        "Abortive-Infection", "CBASS", "Thoeris", "Pycsar", "Retron",
        "Lamassu", "Viperin", "RADAR",
    )
    # Everything else is "unclassified": reported, but excluded from the
    # confirmatory contrast because its mechanism does not license a
    # directional prediction either way.

    # Permutations for the group-contrast null. Group labels are permuted
    # across systems, which preserves whatever dependence exists among the
    # per-system estimates (phylogeny, co-occurrence in defense islands).
    entry_mode_n_permutations: int = 20_000

    # ==================================================================
    # B1 — phylogenetic sister-pair (cherry) design
    # ==================================================================
    # Sister species share ancestry almost completely, so a within-pair
    # contrast removes phylogenetic confounding by construction rather than by
    # model. Requiring pair members to have similar sequencing depth removes
    # the depth confound by construction too -- the thing the depth spline can
    # only model away.
    run_sister_pairs: bool = True
    # Maximum |log1p(n_strains) difference| within a pair. Pairs exceeding this
    # are dropped: without depth matching the design reintroduces exactly the
    # confound it exists to eliminate.
    sister_pair_max_log_depth_diff: float = 0.5
    # Minimum discordant pairs required before a system is testable.
    sister_pair_min_discordant: int = 10
    # Also fit a conditional-logistic adjustment for the residual within-pair
    # depth difference, alongside the assumption-light exact McNemar.
    sister_pair_conditional_logistic: bool = True

    # ==================================================================
    # B2 — directionality from Pagel's dependent-transition models
    # ==================================================================
    # phytools::fitPagel accepts dep.var = "x" | "y" | "xy". Fitting all three
    # gives nested models comparable by AIC:
    #
    #   dep.var = "x"  transitions in PLASMID depend on defense state
    #                  -> "defense state drives plasmid gain/loss"
    #   dep.var = "y"  transitions in DEFENSE depend on plasmid state
    #                  -> "plasmid carriage drives defense gain/loss"
    #   dep.var = "xy" mutual dependence (the previous default)
    #
    # This is the only analysis in the design that speaks to evolutionary
    # ORDERING rather than association, and it answers the "or vice versa"
    # half of the research question directly.
    pagels_fit_directional_models: bool = True
    # Minimum AIC advantage before calling a direction. 2 is the conventional
    # "meaningfully better" threshold; below it, report "ambiguous".
    pagels_direction_min_delta_aic: float = 2.0

    # ==================================================================
    # B3 — matched-feature negative control
    # ==================================================================
    # Question: is there anything special about DEFENSE systems, or would any
    # accessory gene with the same prevalence and the same degree of
    # phylogenetic clustering show the same association?
    #
    # Synthetic traits are simulated on the actual tree by thresholding a
    # Brownian trait on a lambda-rescaled tree, which reproduces both the
    # target prevalence and a controlled amount of phylogenetic signal without
    # requiring any additional annotation. If a real gene-family table is
    # supplied it is used in addition.
    run_feature_control: bool = True
    feature_control_n_per_system: int = 5
    feature_control_lambda_grid: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    # Cap the number of real systems whose prevalence is matched, for runtime.
    # Systems are sampled across the prevalence range rather than taken from
    # the top, so the control spans the same prevalence spectrum as the data.
    feature_control_max_systems: int = 60
    # Optional TSV of non-defense gene families: rows = genome/species,
    # columns = gene families, 0/1. Empty string disables.
    feature_control_gene_family_file: str = ""

    # ==================================================================
    # B4 — E-values for unmeasured confounding
    # ==================================================================
    run_evalues: bool = True
    # OR-to-RR conversion for the E-value.
    #
    #   None  (DEFAULT)  decide from the OBSERVED outcome prevalence
    #   True             outcome is rare; OR ~ RR, use the OR directly
    #   False            outcome is common; RR ~ sqrt(OR)
    #
    # This was previously hard-coded False on the assumption that
    # species-level plasmid carriage is common "because the propagation step
    # drives prevalence high". That assumption is WRONG for this dataset:
    # measured prevalence is 5.7% (2,262 of 39,681 analysed species), which is
    # well inside the rare-outcome regime. Propagation raises prevalence
    # relative to the strain level, but from a per-strain rate of ~0.6% it
    # lands nowhere near common.
    #
    # Using sqrt(OR) on a 5.7% outcome UNDERSTATES the E-value, i.e. reports
    # the associations as less robust to unmeasured confounding than they are.
    # Conservative, but wrong, and a reviewer who checks the prevalence will
    # catch it.
    #
    # Deciding from the data rather than from a constant removes the failure
    # mode entirely: the chosen conversion is logged and recorded in the output
    # so it is never silent.
    evalue_rare_outcome: Optional[bool] = None
    # Prevalence below which the outcome is treated as rare. 0.15 is the
    # conventional cut for the OR ~ RR approximation.
    evalue_rare_outcome_threshold: float = 0.15

    # ------------------------------------------------------------------
    # Negative control
    # ------------------------------------------------------------------
    # Calibration check: permute the plasmid label within joint
    # (clade, n_strains decile) strata and run the primary univariate sweep.
    # Under a correctly specified model the FDR-significant count should be
    # near zero. If it is not, the pipeline is measuring sequencing effort and
    # no downstream result is interpretable. This is the single most
    # informative stage in the pipeline and should be run before any result is
    # believed.
    run_negative_control: bool = True
    negative_control_n_replicates: int = 20
    # Fail loudly if the mean FDR-significant count across replicates exceeds
    # this. alpha * n_systems is the expected count under correct calibration.
    negative_control_max_expected_hits_multiplier: float = 3.0

    # ==================================================================
    # Compute budget
    # ==================================================================
    # Every knob below trades scope for tractability. They exist because four
    # stages could not complete on the 128 GB / 25-day cluster: LOCO and the
    # misclassification Monte Carlo died with SIGBUS under concurrent I/O
    # pressure, phylo_signal hit the wall-clock ceiling, and PGLMM at full tree
    # size exceeded node memory. See docs/cluster_optimization_log.md.
    #
    # The guiding rule: reduce the NUMBER OF MODEL FITS by restricting each
    # stage to the comparisons it actually needs, rather than degrading the
    # statistics of the fits that remain.

    # --- LOCO ---------------------------------------------------------
    # LOCO is a stability check ON THE PRIMARY RESULT. Running it across all
    # covariate modes multiplies the cost with no added inference: the
    # unadjusted arm is a confound positive control and the depth_only arm is
    # a decomposition, neither of which has a "stability" claim attached.
    loco_covariate_modes_primary_only: bool = True
    # Only fit clades big enough to matter. Dropping 3 species out of 40,000
    # returns a near-duplicate of the full-data estimate: it is uninformative
    # as an influence diagnostic AND excluded from heterogeneity anyway, so
    # fitting it is pure waste. At GTDB class this typically leaves ~40-60
    # clades instead of ~200.
    loco_fit_only_gated_clades: bool = True
    # Primary rank only by default; set to both ranks for the fuller sweep.
    loco_ranks_primary_only: bool = True

    # --- Misclassification Monte Carlo --------------------------------
    # The MC answers "would this FINDING survive plasmid-detection false
    # negatives?". That question only applies to findings, so restricting the
    # sweep to systems that are FDR-significant in the primary analysis is not
    # a shortcut -- it is the correct scope. Typically 435 -> ~20-40 systems.
    misclass_restrict_to_significant: bool = True
    # Fall back to this many top-ranked systems if nothing reaches FDR.
    misclass_max_systems: int = 40
    # Primary covariate mode only, for the same reason as LOCO.
    misclass_primary_mode_only: bool = True
    # The reported quantity is a median coefficient per FNR level. 200 draws
    # is far past the point where the median stabilises; 40 gives a standard
    # error on the median about 1/6 of the coefficient's own SE, which is
    # negligible. Verified numerically in tests.
    misclass_n_replicates_effective: int = 40
    # 4 grid points span the plausible FNR range as well as 7 do for a
    # monotone attenuation curve, and the analytical Bross correction covers
    # the continuum anyway.
    misclass_fnr_grid_reduced: Tuple[float, ...] = (0.00, 0.10, 0.20, 0.30)
    misclass_use_reduced_grid: bool = True

    # --- Pagel directionality -----------------------------------------
    # Fitting the two restricted dependent models triples the per-system cost
    # of the most expensive stage. Direction is only MEANINGFUL where the
    # traits are actually dependent -- asking "which drives which?" for an
    # independent pair is not a question -- so the directional fits are gated
    # on the standard Pagel test rejecting independence. This is a scope
    # restriction that follows from the semantics, not a compromise.
    pagels_directional_only_if_dependent: bool = True
    pagels_directional_screen_alpha: float = 0.10
    # And only for outcomes a primary claim may be made about.
    pagels_directional_primary_outcomes_only: bool = True

    # --- phylo_signal --------------------------------------------------
    # Use the native vectorised D statistic instead of caper. caper was killed
    # at the 25-day ceiling; the native implementation computes the identical
    # statistic in well under an hour by flattening the tree once, traversing
    # level-wise, and batching all permutations as a matrix.
    phylo_signal_engine: str = "native"      # "native" | "caper"

    # --- entry mode -----------------------------------------------------
    # Iterate systems INSIDE one R process rather than spawning one R call per
    # system. Amortises interpreter start-up, tree parsing and data parsing
    # across all systems: 435 R invocations become 1.
    entry_mode_batch_in_r: bool = True
    # If the binomial PGLMM cannot complete, fall back automatically to the
    # empirical-logit PGLS rather than losing the stage. Recorded in the
    # output's `engine` column so the fallback is never invisible.
    entry_mode_auto_fallback_to_pgls: bool = True

    # --- R subprocess resilience ---------------------------------------
    # Retry calls killed by a SIGNAL (SIGBUS/SIGKILL), which at this scale are
    # transient resource contention. R-level errors are deterministic and are
    # never retried.
    r_max_retries: int = 2
    # Hard cap on concurrent R subprocesses, independent of n_jobs. The SIGBUS
    # crashes correlated with many workers writing large frames at once; with
    # the shared-frame bridge the per-call writes are tiny, but this remains as
    # a safety valve. 0 means "use n_jobs".
    max_concurrent_r_calls: int = 0

    # Smoke-testing. When set, the species table is randomly subsampled to
    # this many species (stratified by phylum so the tree keeps its shape)
    # immediately after aggregation. Exists so the R side can be exercised in
    # minutes rather than hours: without it there is no way to run a genuine
    # smoke test, and "run a few stages on the full data" is not a smoke test,
    # it is a short real run. NEVER set this for a production run -- results
    # from a subsample are not the analysis.
    subsample_species: Optional[int] = None

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------
    n_jobs: int = -1
    random_seed: int = 42

    # Which stages to run (used by the driver when a user wants a partial
    # rerun). An empty tuple means "run all".
    stages: Tuple[str, ...] = ()

    # R executable — override if R is not on $PATH.
    r_executable: str = "Rscript"

    def require_r(self) -> None:
        """Quick check that an R executable is callable. Raises RuntimeError."""
        import shutil
        if shutil.which(self.r_executable) is None:
            raise RuntimeError(
                f"R executable '{self.r_executable}' not found on PATH. "
                "Install R >= 4.0 plus packages: ape, phylolm, phytools, "
                "caper, phyr, nlme."
            )

    # ------------------------------------------------------------------
    # Covariate resolution
    # ------------------------------------------------------------------

    def depth_spline_columns(self) -> Tuple[str, ...]:
        """Candidate names of the spline basis columns on log(n_strains).

        Built in ``io_utils.add_depth_basis`` and passed to R as ordinary
        numeric covariates. ``depth_spline_df = 1`` degenerates to a single
        linear ``log(n_strains)`` term, reproducing legacy behaviour.

        FEWER columns than ``depth_spline_df`` may actually exist: n_strains is
        an integer with heavy ties at the low end, so requested knot quantiles
        can collapse and the basis builder drops the duplicates rather than
        emitting collinear columns. Always pass the result through
        ``resolve_covariates`` against the frame being fit.
        """
        if not self.use_n_strains_covariate:
            return ()
        return tuple(f"{self.depth_spline_prefix}{i + 1}"
                     for i in range(max(1, int(self.depth_spline_df))))

    @staticmethod
    def resolve_covariates(columns, frame) -> Tuple[str, ...]:
        """Drop covariate names absent from ``frame``, preserving order.

        Every R call site must funnel its covariate list through this. Spline
        bases can be shorter than requested (see ``depth_spline_columns``), and
        genome covariates can be missing entirely when the optional covariate
        table was not supplied. Passing a non-existent column name to the R
        scripts produces a ``column_missing`` skip for every defense system,
        i.e. an entire silently-empty stage.
        """
        available = set(getattr(frame, "columns", frame))
        return tuple(c for c in columns if c in available)

    def plasmid_count_spline_columns(self) -> Tuple[str, ...]:
        """Names of the natural-spline basis columns on log(n_plasmids)."""
        if not self.use_plasmid_count_covariate_on_binary:
            return ()
        return tuple(f"{self.plasmid_count_spline_prefix}{i + 1}"
                     for i in range(max(1, int(self.plasmid_count_spline_df))))

    def normalise_covariate_mode(self, mode: str) -> str:
        """Map a legacy covariate-mode label onto its current name."""
        for old, new in self.covariate_mode_aliases:
            if mode == old:
                return new
        return mode

    def covariate_columns(self, include_plasmid_count: bool = False,
                          include_n_strains: bool = True,
                          include_genome: bool = True) -> Tuple[str, ...]:
        """Species-level covariate column names to pass to the R scripts.

        ``include_n_strains`` adds the depth spline basis, which partials out
        the sampling-depth saturation shared by the max()-aggregated defense
        call and the species-propagated plasmid label. Callers should almost
        never set this False: depth is a common cause of BOTH variables, so
        removing it leaves the outcome arm of the confound unadjusted even
        when the predictor has been de-saturated by other means.

        ``include_plasmid_count`` adds the log(n_plasmids) spline basis, used
        on binary stratified outcomes so that species with thousands of
        plasmids do not trivially carry one of every class.
        """
        cov: Tuple[str, ...] = ()
        if include_genome and self.use_genome_covariates:
            cov = cov + tuple(self.genome_covariate_columns)
        if include_n_strains:
            cov = cov + self.depth_spline_columns()
        if include_plasmid_count:
            cov = cov + self.plasmid_count_spline_columns()
        return cov

    def covariate_columns_for_mode(self, mode: str,
                                   include_plasmid_count: bool = False,
                                   include_n_strains: bool = True
                                   ) -> Tuple[str, ...]:
        """Resolve covariates for a covariate_mode label.

            full        genome covariates + depth spline
            depth_only  depth spline only
            unadjusted  nothing (confound positive control)

        ``include_n_strains=False`` is honoured but is a footgun; it exists
        only for the deliberately-unadjusted arms.
        """
        mode = self.normalise_covariate_mode(mode)
        if mode == "unadjusted":
            return ()
        if mode == "depth_only":
            return self.covariate_columns(
                include_plasmid_count=include_plasmid_count,
                include_n_strains=include_n_strains,
                include_genome=False,
            )
        return self.covariate_columns(
            include_plasmid_count=include_plasmid_count,
            include_n_strains=include_n_strains,
            include_genome=True,
        )

    def is_diagnostic_mode(self, mode: str) -> bool:
        """True if results under this covariate mode are diagnostic only."""
        return self.normalise_covariate_mode(mode) in tuple(
            self.normalise_covariate_mode(m)
            for m in self.diagnostic_covariate_modes
        )

    def is_primary_slice(self, outcome_label: str, covariate_mode: str) -> bool:
        """True if (outcome, covariate mode) is eligible for a primary claim
        and therefore belongs in the global FDR family."""
        return (outcome_label in self.primary_outcome_labels
                and self.normalise_covariate_mode(covariate_mode)
                == self.normalise_covariate_mode(self.primary_covariate_mode))

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        """Stable hash of every field that can change a numeric result.

        Stage checkpointing keys on this, so changing a threshold and
        re-running no longer silently reuses a stale TSV. Paths and compute
        knobs (n_jobs, r_executable, output_dir, stages) are excluded because
        they cannot change the numbers.
        """
        import hashlib
        import json
        from dataclasses import asdict

        ignore = {"output_dir", "n_jobs", "stages", "r_executable",
                  "pglmm_timeout_hours", "pagels_timeout_hours"}
        payload = {k: v for k, v in sorted(asdict(self).items())
                   if k not in ignore}
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]
