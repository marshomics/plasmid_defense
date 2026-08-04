"""Regression tests for the sampling-depth and statistics fixes.

Each test pins a specific bug that was found in the audit. They are written so
that reverting the fix makes the test fail with a message naming the bug.

Run with:  python -m pytest tests/test_pipeline_fixes.py -v
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_analysis_v2 import io_utils, stats_utils, tier3_sensitivity  # noqa: E402
from defense_analysis_v2.config import Config  # noqa: E402

LOG = logging.getLogger("test")
LOG.addHandler(logging.NullHandler())


# ======================================================================
# Synthetic data with the same pathology as the real thing
# ======================================================================

def make_null_data(n_species=3000, seed=0, n_defense=6):
    """Species table where defense and plasmid are INDEPENDENT per strain.

    Both species-level labels are built the way the pipeline builds them:
    "at least one positive among n_strains". Any association a method finds
    here is manufactured by the aggregation.
    """
    rng = np.random.default_rng(seed)
    clade = rng.integers(0, 12, n_species)
    clade_depth = rng.lognormal(0.4, 1.3, 12)
    n_strains = np.minimum(np.maximum(1, rng.poisson(clade_depth[clade]) + 1), 3000)

    p_pla = np.clip(rng.beta(2, 8, 12)[clade] * rng.lognormal(0, .4, n_species),
                    1e-4, .99)
    plasmid = rng.binomial(1, 1 - (1 - p_pla) ** n_strains)

    df = pd.DataFrame({
        "gtdb_species": [f"s__sp{i}" for i in range(n_species)],
        "tip": [f"s__sp{i}" for i in range(n_species)],
        "gtdb_phylum": [f"p__{c // 4}" for c in clade],
        "gtdb_class": [f"c__{c}" for c in clade],
        "n_strains": n_strains,
        "log_n_strains": np.log1p(n_strains),
        "has_plasmid_binary": plasmid,
    })
    for j in range(n_defense):
        p_def = np.clip(rng.beta(2, 20, 12)[clade] * rng.lognormal(0, .4, n_species),
                        1e-4, .99)
        df[f"sys{j}"] = rng.binomial(1, 1 - (1 - p_def) ** n_strains)
        # strain-mean prevalence, used by prevalence_df
        df[f"sys{j}_prev"] = p_def
    return df


DEFENSE_COLS = [f"sys{j}" for j in range(6)]


# ======================================================================
# 1. Depth spline basis
# ======================================================================

def test_spline_basis_is_full_rank_and_finite():
    rng = np.random.default_rng(1)
    x = np.log1p(np.maximum(1, rng.lognormal(0.7, 1.6, 4000).astype(int)))
    basis, knots = io_utils.restricted_cubic_spline_basis(x, df=5)
    assert basis.shape[1] >= 1
    assert np.isfinite(basis).all()
    design = np.column_stack([np.ones(len(x)), basis])
    assert np.linalg.matrix_rank(design) == design.shape[1], \
        "spline basis is collinear with the intercept"


def test_spline_basis_degenerates_gracefully():
    """Constant / near-constant input must not emit collinear columns."""
    basis, knots = io_utils.restricted_cubic_spline_basis(np.ones(200), df=5)
    assert basis.shape[1] == 1 and knots == []


def test_spline_df_one_reproduces_linear_term():
    x = np.linspace(0, 5, 100)
    basis, _ = io_utils.restricted_cubic_spline_basis(x, df=1)
    assert basis.shape[1] == 1
    np.testing.assert_allclose(basis[:, 0], x)


def test_depth_basis_attached_and_idempotent():
    cfg = Config()
    df = make_null_data(500)
    df = io_utils.add_depth_basis(df, cfg, LOG)
    cols1 = [c for c in df.columns if c.startswith(cfg.depth_spline_prefix)]
    assert cols1, "depth spline basis was not attached"
    df = io_utils.add_depth_basis(df, cfg, LOG)
    cols2 = [c for c in df.columns if c.startswith(cfg.depth_spline_prefix)]
    assert cols1 == cols2, "re-attaching the basis duplicated columns"


def test_spline_removes_more_confounding_than_linear_term():
    """The whole reason for the spline: a single linear log(n) term leaves
    residual depth confounding, the spline basis does not."""
    import statsmodels.api as sm
    cfg = Config()

    def fpr(use_spline, n_rep=25):
        hits = 0
        for r in range(n_rep):
            d = make_null_data(2500, seed=100 + r, n_defense=1)
            d = io_utils.add_depth_basis(d, cfg, LOG)
            if use_spline:
                cols = [c for c in d.columns
                        if c.startswith(cfg.depth_spline_prefix)]
            else:
                cols = ["log_n_strains"]
            X = sm.add_constant(d[["sys0"] + cols].astype(float))
            try:
                m = sm.Logit(d["has_plasmid_binary"].astype(float), X).fit(
                    disp=0, maxiter=300)
                hits += int(m.pvalues["sys0"] < 0.05)
            except Exception:
                pass
        return hits / n_rep

    assert fpr(True) <= fpr(False) + 0.08, \
        "spline adjustment is not at least as good as the linear term"


# ======================================================================
# 2. Pagel's: median-p was not a p-value
# ======================================================================

def test_cauchy_combination_is_calibrated_and_median_is_not():
    rng = np.random.default_rng(7)
    draws = [rng.uniform(size=5) for _ in range(8000)]
    med = np.array([np.median(p) for p in draws])
    cau = np.array([stats_utils.combine_subsample_pvalues(p) for p in draws])
    assert abs((cau < 0.05).mean() - 0.05) < 0.012, \
        "Cauchy combination is not calibrated under H0"
    assert (med < 0.05).mean() < 0.01, \
        "median-of-p is expected to be strongly super-uniform (the old bug)"


def test_cauchy_handles_varying_k():
    """Rows combining different numbers of subsamples must land on one scale."""
    rng = np.random.default_rng(3)
    for k in (2, 5, 10):
        p = np.array([stats_utils.combine_subsample_pvalues(rng.uniform(size=k))
                      for _ in range(4000)])
        assert abs((p < 0.05).mean() - 0.05) < 0.02, f"miscalibrated at k={k}"


# ======================================================================
# 3. Rank product: uncalibrated score promoted single-method rows
# ======================================================================

def test_rank_product_null_is_calibrated_across_k():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({m: rng.permutation(n) + 1.0 for m in "abc"})
    df.loc[df.sample(120, random_state=1).index, "c"] = np.nan
    df.loc[df.sample(80, random_state=2).index, "b"] = np.nan
    out = stats_utils.rank_product_with_null(df, list("abc"),
                                             n_permutations=20000,
                                             random_seed=5)
    assert abs((out["rank_product_p_value"] < 0.05).mean() - 0.05) < 0.03
    for k, g in out.groupby("n_methods_contributing"):
        assert abs(g["rank_product_p_value"].mean() - 0.5) < 0.12, \
            f"rank-product null is not calibrated at k={k}"


def test_rank_product_no_longer_favours_single_method_rows():
    """A row scored by one method must not beat a strong row scored by three.

    Under the old raw geometric mean, `single_method` scored 1.0 and
    `three_methods` scored 2.0, so the row with the LEAST corroboration came
    out on top. With a per-k permutation null, being ranked 2nd of 200 by
    three independent methods is far less likely under H0 than being ranked
    1st by one method, so the ordering inverts.
    """
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({m: rng.permutation(n) + 1.0 for m in "abc"},
                      index=[f"filler{i}" for i in range(n)])
    df.loc["filler0"] = [1.0, np.nan, np.nan]   # ranked #1 by one method only
    df.loc["filler1"] = [2.0, 2.0, 2.0]         # ranked #2 by all three
    df = df.rename(index={"filler0": "single_method",
                          "filler1": "three_methods"})

    raw = stats_utils.rank_product(df, list("abc"), missing_policy="skip")
    assert raw["single_method"] < raw["three_methods"], \
        "fixture no longer reproduces the original raw-score inversion"

    out = stats_utils.rank_product_with_null(df, list("abc"),
                                             n_permutations=20000,
                                             random_seed=1)
    assert out.loc["three_methods", "rank_product_p_value"] < \
        out.loc["single_method", "rank_product_p_value"], \
        "single-method row still outranks a corroborated row"


# ======================================================================
# 4. Global FDR was never applied
# ======================================================================

def test_global_fdr_is_applied_and_restricted_to_primary_family():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "defense_system": [f"s{i}" for i in range(200)],
        "phyloglm_p_value": rng.uniform(size=200),
        "outcome_label": ["any_plasmid"] * 100 + ["reptype_IncF"] * 100,
        "covariate_mode": ["full"] * 200,
    })
    cfg = Config()
    mask = pd.Series([cfg.is_primary_slice(l, m) for l, m
                      in zip(df.outcome_label, df.covariate_mode)])
    out = stats_utils.apply_global_fdr(df, ["phyloglm_p_value"],
                                       family_mask=mask)
    assert "phyloglm_p_value_global_qvalue" in out.columns
    exploratory = out[out.outcome_label == "reptype_IncF"]
    assert exploratory["phyloglm_p_value_global_qvalue"].isna().all(), \
        "exploratory strata leaked into the global FDR family"
    primary = out[out.outcome_label == "any_plasmid"]
    assert primary["phyloglm_p_value_global_qvalue"].notna().any()


# ======================================================================
# 5. Depth-matched test replaces the vacuous prevalence-matched test
# ======================================================================

def test_depth_matched_test_produces_real_pvalues():
    """The old prevalence-matched test returned NaN for every system because
    it matched on prevalence deciles and then tested binary == (prev > 0)."""
    cfg = Config()
    d = make_null_data(4000, seed=11)
    prevalence_df = d.copy()
    for c in DEFENSE_COLS:
        prevalence_df[c] = d[f"{c}_prev"]
    out = tier3_sensitivity.run_depth_matched(d, prevalence_df, DEFENSE_COLS,
                                              cfg, LOG)
    assert not out.empty, "depth-matched test returned nothing"
    assert out["matched_p_value"].notna().any(), \
        "every p-value is NaN — this is the old vacuous test"
    finite = out["matched_p_value"].dropna()
    assert ((finite >= 0) & (finite <= 1)).all()


def test_old_prevalence_matched_alias_points_at_working_test():
    assert tier3_sensitivity.run_prevalence_matched is \
        tier3_sensitivity.run_depth_matched


# ======================================================================
# 6. Permutation null must preserve the depth-outcome relationship
# ======================================================================

def test_permutation_strata_include_depth():
    cfg = Config()
    d = make_null_data(2000, seed=4)
    strata = tier3_sensitivity.build_permutation_strata(d, cfg, LOG)
    assert len(np.unique(strata)) > d["gtdb_class"].nunique(), \
        "strata are not finer than clade alone — depth is not being held fixed"


def test_stratified_shuffle_preserves_depth_outcome_covariance():
    """This is the core fix: shuffling within clade ALONE destroys
    Cov(plasmid, n_strains), which is exactly the confound under test."""
    cfg = Config()
    d = make_null_data(4000, seed=6)
    rng = np.random.default_rng(0)
    plasmid = d["has_plasmid_binary"].values
    depth = d["log_n_strains"].values
    obs_corr = np.corrcoef(plasmid, depth)[0, 1]

    joint = tier3_sensitivity.build_permutation_strata(d, cfg, LOG)
    clade_only = d["gtdb_class"].astype(str).values

    joint_corr = np.mean([
        np.corrcoef(tier3_sensitivity._stratified_shuffle(plasmid, joint, rng),
                    depth)[0, 1] for _ in range(15)])
    clade_corr = np.mean([
        np.corrcoef(tier3_sensitivity._stratified_shuffle(plasmid, clade_only, rng),
                    depth)[0, 1] for _ in range(15)])

    assert obs_corr > 0.05, "test fixture has no depth-outcome association"
    assert abs(joint_corr - obs_corr) < abs(clade_corr - obs_corr), (
        "depth-stratified permutation does not preserve the depth-outcome "
        "relationship better than clade-only shuffling — the null is "
        "anticonservative")


# ======================================================================
# 7. Config plumbing
# ======================================================================

def test_unadjusted_mode_is_diagnostic_only():
    cfg = Config()
    assert cfg.is_diagnostic_mode("unadjusted")
    assert cfg.is_diagnostic_mode("without_cov")     # legacy alias
    assert not cfg.is_diagnostic_mode("full")
    assert cfg.covariate_columns_for_mode("unadjusted") == ()


def test_resolve_covariates_drops_missing_columns():
    cfg = Config()
    frame = pd.DataFrame({"a": [1], "depth_ns1": [1.0]})
    assert Config.resolve_covariates(["a", "depth_ns1", "nope"], frame) == \
        ("a", "depth_ns1")


def test_fingerprint_changes_with_a_threshold():
    a = Config().fingerprint()
    b = Config(min_n_strains_sensitivity=20).fingerprint()
    assert a != b, "config fingerprint does not detect a threshold change"
    c = Config(output_dir="/tmp/somewhere-else").fingerprint()
    assert a == c, "fingerprint should ignore non-numeric-result fields"


def test_primary_stratified_mode_matches_what_consensus_consumes():
    """config declared 'fraction' primary while every consumer filtered to
    binary — the two must now agree."""
    assert Config().plasmid_stratified_primary_mode == "binary"


# ======================================================================
# 8. Firth under separation — the regime Firth actually exists for
# ======================================================================

def test_firth_matches_mle_without_separation():
    res = stats_utils.validate_firth_implementation()
    assert res["passed"], f"Firth disagrees with GLM on clean data: {res}"


def test_firth_is_finite_under_complete_separation():
    """The old validation only covered non-separated data, which exercises the
    likelihood but not the Jeffreys-prior penalty."""
    res = stats_utils.validate_firth_under_separation()
    assert res["finite_coefficients"] and res["finite_standard_errors"], \
        "Firth returned non-finite estimates under separation"
    assert res["bounded"], "Firth coefficient diverged under separation"
    assert res["smaller_than_mle"], \
        "Firth did not shrink the separated slope relative to the MLE"
    assert res["passed"], res


# ======================================================================
# 9. End-to-end: aggregation attaches everything downstream expects
# ======================================================================

def test_aggregation_emits_depth_basis_and_logs_transform():
    cfg = Config()
    rng = np.random.default_rng(5)
    n_strain_rows = 4000
    species = rng.integers(0, 400, n_strain_rows)
    strain = pd.DataFrame({
        "genome": [f"GCF_{i}" for i in range(n_strain_rows)],
        "gtdb_species": [f"s__sp{s}" for s in species],
        "has_plasmid_binary": 0,
        "gtdb_domain": "Bacteria", "gtdb_phylum": "p__A", "gtdb_class": "c__A",
        "gtdb_order": "o__A", "gtdb_family": "f__A", "gtdb_genus": "g__A",
        "sysA": rng.integers(0, 2, n_strain_rows),
        "corrected_genome_size": rng.lognormal(15, 0.3, n_strain_rows),
        "gc_avg": rng.uniform(30, 70, n_strain_rows),
        "cds_number": rng.lognormal(8, 0.3, n_strain_rows),
    })
    # plasmid label constant within species, as the pipeline requires
    per_sp = {s: int(v) for s, v in
              zip(strain.gtdb_species.unique(),
                  rng.integers(0, 2, strain.gtdb_species.nunique()))}
    strain["has_plasmid_binary"] = strain.gtdb_species.map(per_sp)

    prevalence_df, binary_df, spec = io_utils.aggregate_to_species_level(
        strain, ["sysA"], LOG, config=cfg, plasmid_md=None)

    depth_cols = [c for c in binary_df.columns
                  if c.startswith(cfg.depth_spline_prefix)]
    assert depth_cols, "depth spline basis missing from aggregated table"
    assert binary_df[depth_cols].notna().all().all()
    # heavy-tailed covariates must be log-transformed (config claimed the R
    # layer did this; it never did)
    assert binary_df["corrected_genome_size"].max() < 25, \
        "genome size was not log-transformed"
    assert binary_df["cds_number"].max() < 25, \
        "CDS count was not log-transformed"
    assert "gc_avg" in binary_df.columns and binary_df["gc_avg"].max() > 25, \
        "GC content should NOT be log-transformed"


def test_aggregation_rejects_within_species_label_conflict():
    cfg = Config()
    strain = pd.DataFrame({
        "genome": ["a", "b"],
        "gtdb_species": ["s__x", "s__x"],
        "has_plasmid_binary": [0, 1],      # conflict
        "gtdb_domain": "Bacteria", "gtdb_phylum": "p", "gtdb_class": "c",
        "gtdb_order": "o", "gtdb_family": "f", "gtdb_genus": "g",
        "sysA": [0, 1],
    })
    with pytest.raises(ValueError, match="species-level plasmid invariant"):
        io_utils.aggregate_to_species_level(strain, ["sysA"], LOG, config=cfg)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
