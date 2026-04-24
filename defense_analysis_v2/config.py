"""Configuration dataclass for the pipeline."""

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
        "human_animal_free_90percent/output/gtdbtk.rooted.speciesnames.tree"
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
    top_n_rep_types: int = 10
    min_rep_type_species: int = 25

    # Size-class bin edges in bp. A plasmid is small if size < bins[0], medium
    # if bins[0] <= size < bins[1], large if size >= bins[1]. Defaults are the
    # conventional mob_suite bins.
    plasmid_size_bins_bp: Tuple[int, int] = (20_000, 100_000)

    # Primary outcome modelling mode per stratified class:
    #   "fraction" - fraction-of-species-plasmids that fall in class X, with
    #                plasmid-count as binomial weight. Uses cbind(k,n-k) in R.
    #   "binary"   - species has at least one plasmid of class X
    #                (with log(n_plasmids) included as covariate to defuse the
    #                "species with many plasmids has every type" saturation).
    # Both are computed; this flag chooses which is the *primary* test. The
    # other is carried through as a secondary / concordance check.
    plasmid_stratified_primary_mode: str = "fraction"

    # Also run the legacy "any plasmid vs none" outcome for backward
    # comparability. This is saturated at species level (most species end up
    # labelled as plasmid-carriers), but reviewers will ask for it.
    include_any_plasmid_outcome: bool = True

    # ------------------------------------------------------------------
    # Significance / multiple-testing
    # ------------------------------------------------------------------
    alpha: float = 0.05
    fdr_method: str = "fdr_bh"     # Benjamini-Hochberg

    # Whether to apply a single global FDR across all primary tests (on top of
    # per-tier FDR). Reported as an additional column; per-tier FDR remains the
    # default decision criterion.
    report_global_fdr: bool = True

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

    # Minimum species count per left-out clade to keep the LOCO result for that
    # clade. Clades below this are reported but excluded from Cochran's Q.
    min_species_per_loco_clade: int = 50

    # ------------------------------------------------------------------
    # Clade / stratification choices
    # ------------------------------------------------------------------
    # GTDB rank for clade-restricted permutation null (preserves clade-level
    # plasmid prevalence).
    permutation_clade_rank: str = "gtdb_phylum"

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
    # call is subsampled. Rather than a single draw, we take N independent
    # uniform subsamples and report the median logLR p-value plus the
    # fraction of subsamples significant at alpha. Consensus downstream uses
    # the median. Set pagels_n_subsamples = 1 to restore the legacy behaviour.
    pagels_n_subsamples: int = 5
    pagels_subsample_size: int = 1500

    # ------------------------------------------------------------------
    # Misclassification sensitivity
    # ------------------------------------------------------------------
    # Assumed plasmid-detection false-negative rates to sweep. Lower is more
    # optimistic. The default range reflects the spread across modern plasmid
    # assemblers on Illumina-only data.
    misclass_fnr_grid: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    misclass_n_replicates: int = 200
    misclass_false_positive_rate: float = 0.0   # Assume no FP; flag in methods

    # ------------------------------------------------------------------
    # Covariates, bidirectional framing, interactions
    # ------------------------------------------------------------------
    # Covariate modes to iterate over inside every test stage. Each stage
    # runs once per mode and tags output rows with a ``covariate_mode``
    # column, so consumers can compare with-vs-without-covariate fits side
    # by side. ``with_cov`` uses the genome-scale covariates; ``without_cov``
    # uses none (and is the classic uncontrolled fit).
    covariate_modes: Tuple[str, ...] = ("with_cov", "without_cov")

    # Extra covariates added to every phylogenetic model (univariate phyloglm
    # and multivariate PGLMM) to control for genome capacity. We log-transform
    # genome_size and cds_number (heavy-tailed) and centre everything to unit
    # variance; done at the R layer. This flag gates whether the covariates
    # exist at all; the per-run covariate_mode decides whether a given fit
    # uses them.
    use_genome_covariates: bool = True

    # For binary plasmid-class outcomes, also include log(n_plasmids_total) as
    # a covariate so species with thousands of plasmids don't saturate.
    use_plasmid_count_covariate_on_binary: bool = True

    # Sampling-depth correction. Species-level binary defense presence is
    # produced by max() across strains, which saturates for heavily sequenced
    # species. We add log(n_strains) as a covariate on every phylogenetic
    # model so this saturation is partialled out alongside genome capacity.
    # Gate: turn off to reproduce the legacy no-correction behaviour.
    use_n_strains_covariate: bool = True

    # Minimum n_strains a species must have to be kept for the n_strains-
    # sensitivity rerun. The primary run uses every species; the sensitivity
    # run filters to species with >= this many strains and refits phyloglm to
    # show that the associations aren't an artefact of poorly-sampled species.
    min_n_strains_sensitivity: int = 5

    # Additional feature-mode sensitivity. The primary Tier 2 features are the
    # species-level binary presence flags (max across strains). When this is
    # True, a second phyloglm fit is done using the per-species *prevalence*
    # (mean across strains) of the defense system as the predictor, which is
    # immune to the max() saturation bias. Runs against the primary any_plasmid
    # outcome only — its purpose is a pointed sensitivity, not a full replica
    # of the stratified analysis.
    run_prevalence_feature_sensitivity: bool = True

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
    # Evolutionary model for phyloglm. Brownian motion is standard for binary
    # traits on a fixed tree. Alternatives ("BM+lambda", "OUfixedRoot") are
    # available; we log which is used so methods can cite it.
    phylo_evolutionary_model: str = "BM"   # documented default

    # Additional evolutionary models to re-fit phyloglm under for a sensitivity
    # run against the legacy any_plasmid outcome. Defense systems and plasmids
    # move horizontally, so Brownian motion is not self-evidently the right
    # covariance structure — this lets reviewers see how much the primary
    # conclusions depend on that choice. Empty tuple disables the sensitivity.
    phylo_model_sensitivity_models: Tuple[str, ...] = ("OUfixedRoot", "BM_penalized")

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

    def covariate_columns(self, include_plasmid_count: bool = False,
                          include_n_strains: bool = True) -> Tuple[str, ...]:
        """Return the species-level covariate column names to pass to R
        scripts. Used by every phylogenetic model call.

        ``include_plasmid_count`` adds ``log_n_plasmids`` as a covariate; that
        column is built by the driver for binary plasmid-class outcomes so the
        "species with thousands of plasmids always has one of every class"
        saturation is explicitly controlled.

        ``include_n_strains`` adds ``log_n_strains`` to partial out the
        species-level sampling-depth bias introduced by max()-aggregating
        strain-level defense calls up to species. Species with many strains
        saturate to 1 for almost every defense system; without this covariate,
        the phyloglm fit mistakes sampling effort for biology. On by default
        when ``use_n_strains_covariate`` is True.
        """
        cov: Tuple[str, ...] = ()
        if self.use_genome_covariates:
            cov = cov + tuple(self.genome_covariate_columns)
        if include_n_strains and self.use_n_strains_covariate:
            cov = cov + ("log_n_strains",)
        if include_plasmid_count and self.use_plasmid_count_covariate_on_binary:
            cov = cov + ("log_n_plasmids",)
        return cov

    def covariate_columns_for_mode(self, mode: str,
                                    include_plasmid_count: bool = False,
                                    include_n_strains: bool = True) -> Tuple[str, ...]:
        """Resolve covariates given a covariate_mode label. ``with_cov``
        returns the standard list; ``without_cov`` returns an empty tuple.
        Used by each test stage when it iterates over ``covariate_modes``.

        The ``include_n_strains`` flag is a belt-and-braces switch for the
        rare case where a caller wants to explicitly exclude log_n_strains
        (e.g. the prevalence-feature sensitivity, where the feature itself is
        already strain-averaged and adding n_strains again is redundant).
        """
        if mode == "without_cov":
            return ()
        return self.covariate_columns(
            include_plasmid_count=include_plasmid_count,
            include_n_strains=include_n_strains,
        )
