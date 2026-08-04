"""Regression tests for A4, B1, B2, B3, B4.

Run with:  python -m pytest tests/test_analysis_extensions.py -v
"""
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_analysis_v2 import (stats_utils, tier2_pagels, tier3_entry_mode,  # noqa: E402
                                 tier3_feature_control, tier3_sister_pairs)
from defense_analysis_v2.config import Config  # noqa: E402
from defense_analysis_v2.taxonomy import classify_defense_system  # noqa: E402

LOG = logging.getLogger("test")
LOG.addHandler(logging.NullHandler())


def _balanced_newick(depth: int, prefix: str = "t") -> str:
    if depth == 0:
        return f"{prefix}:1"
    return (f"({_balanced_newick(depth - 1, prefix + 'L')},"
            f"{_balanced_newick(depth - 1, prefix + 'R')}):1")


@pytest.fixture(scope="module")
def tree_file():
    p = Path(tempfile.mkdtemp()) / "t.nwk"
    p.write_text(_balanced_newick(7) + ";")   # 128 tips
    return str(p)


# ======================================================================
# B4 — E-values
# ======================================================================

def test_evalue_matches_published_values():
    """VanderWeele & Ding 2017 worked example: RR = 3.9 gives E = 7.26."""
    assert round(stats_utils._evalue_from_rr(3.9), 2) == 7.26
    assert round(stats_utils._evalue_from_rr(2.0), 3) == 3.414
    # A null association needs no confounding at all.
    assert stats_utils._evalue_from_rr(1.0) == 1.0


def test_evalue_is_symmetric_under_inversion():
    """A protective effect of 0.5 is as hard to explain away as a harmful 2.0."""
    assert (stats_utils._evalue_from_rr(0.5)
            == pytest.approx(stats_utils._evalue_from_rr(2.0)))


def test_evalue_uses_sqrt_or_for_common_outcomes():
    """The two conversions must differ in the documented direction. Which one
    applies to this dataset is decided from the observed prevalence (5.7%,
    i.e. rare) — see test_evalue_conversion_is_derived_from_observed_prevalence."""
    common = stats_utils.evalue_from_odds_ratio(4.0)
    rare = stats_utils.evalue_from_odds_ratio(4.0, rare_outcome=True)
    assert common["risk_ratio_approx"] == pytest.approx(2.0)
    assert rare["risk_ratio_approx"] == pytest.approx(4.0)
    assert common["evalue_point"] < rare["evalue_point"]
    assert common["evalue_conversion"] == "sqrt_odds_ratio"


def test_evalue_ci_is_one_when_interval_spans_null():
    r = stats_utils.evalue_from_odds_ratio(1.3, 0.7, 2.4)
    assert r["evalue_ci"] == 1.0


def test_evalue_ci_uses_bound_nearest_the_null():
    r = stats_utils.evalue_from_odds_ratio(4.0, 2.0, 8.0)
    # The lower bound (2.0) is nearer the null than the upper (8.0).
    expected = stats_utils._evalue_from_rr(np.sqrt(2.0))
    assert r["evalue_ci"] == pytest.approx(expected)
    assert r["evalue_ci"] < r["evalue_point"]


def test_attach_evalues_over_a_frame():
    df = pd.DataFrame({"phyloglm_odds_ratio": [2.0, 0.5, np.nan],
                       "phyloglm_ci_low": [1.4, 0.3, np.nan],
                       "phyloglm_ci_high": [2.9, 0.8, np.nan]})
    out = stats_utils.attach_evalues(df)
    assert "evalue_point" in out.columns and "evalue_ci" in out.columns
    assert out["evalue_point"].notna().sum() == 2
    assert np.isnan(out["evalue_point"].iloc[2])


# ======================================================================
# A4 — entry mode
# ======================================================================

def test_conjugative_flag_parsing_is_strict():
    p = tier3_entry_mode._parse_conjugative
    assert p("yes") == 1 and p("YES") == 1 and p(" Yes ") == 1 and p("1") == 1
    assert p("no") == 0 and p("No") == 0 and p("0") == 0
    # Anything unrecognised must be None so the plasmid is DROPPED rather than
    # silently coerced — a mis-parsed entry-mode label inverts the contrast.
    assert p("unknown") is None and p("") is None and p(None) is None
    assert p(np.nan) is None


def test_entry_mode_features_count_correctly():
    em = pd.DataFrame({
        "plasmid_id": [f"p{i}" for i in range(6)],
        "gtdb_species": ["s__A"] * 4 + ["s__B"] * 2,
        "conjugative": [1, 1, 0, 0, 1, 1],
    })
    feats = tier3_entry_mode.build_entry_mode_features(
        em, ["s__A", "s__B"], Config(), LOG).set_index("gtdb_species")
    assert feats.loc["s__A", "n_plasmids_entrymode"] == 4
    assert feats.loc["s__A", "n_plasmid_conjugative"] == 2
    assert feats.loc["s__A", "n_plasmid_nonconjugative"] == 2
    assert feats.loc["s__A", "frac_plasmid_nonconjugative"] == pytest.approx(0.5)
    assert feats.loc["s__B", "n_plasmid_nonconjugative"] == 0
    assert feats.loc["s__B", "any_plasmid_nonconjugative"] == 0
    assert feats.loc["s__B", "any_plasmid_conjugative"] == 1


def test_mechanism_partition_is_as_preregistered():
    cfg = Config()
    cols = ["RM_Type_II", "McrBC", "Wadjet_I", "BREX_I", "DISARM_1",
            "CBASS_I", "Thoeris_I", "AbiEii", "Retron_I_A",
            "Gabija", "CRISPR_Cas_I_E"]
    g = tier3_entry_mode.assign_mechanism_groups(cols, cfg).set_index(
        "defense_system")["mechanism_group"].to_dict()
    for s in ("RM_Type_II", "McrBC", "Wadjet_I", "BREX_I", "DISARM_1"):
        assert g[s] == "predicted_dsDNA_restricting", s
    for s in ("CBASS_I", "Thoeris_I", "AbiEii", "Retron_I_A"):
        assert g[s] == "not_predicted", s
    # Mechanism ambiguous -> excluded from the confirmatory contrast.
    for s in ("Gabija", "CRISPR_Cas_I_E"):
        assert g[s] == "unclassified", s


def test_drt_is_not_classified_as_restriction_modification():
    """DRTs are retroelement-based. Classifying them as RM would put them in
    the predicted dsDNA-restricting group and corrupt the confirmatory test."""
    assert classify_defense_system("DRT_1") == "Retron"
    assert classify_defense_system("McrBC") == "Type-IV-Restriction"


def test_entry_mode_confirmatory_detects_a_planted_effect():
    cfg = Config(entry_mode_n_permutations=4000, random_seed=1)
    rng = np.random.default_rng(0)
    pred = [f"RM_Type_II_{i}" for i in range(12)]
    notp = [f"CBASS_{i}" for i in range(12)]
    # Predicted group planted with a negative effect (depletes non-conjugative).
    comp = pd.DataFrame({
        "defense_system": pred + notp,
        "entry_mode_coefficient": np.concatenate([
            rng.normal(-0.8, 0.2, len(pred)), rng.normal(0.0, 0.2, len(notp))]),
        "entry_mode_std_err": 0.15,
    })
    comp = comp.merge(tier3_entry_mode.assign_mechanism_groups(
        list(comp.defense_system), cfg), on="defense_system")
    res = tier3_entry_mode.run_entry_mode_confirmatory(comp, cfg, LOG)
    assert not res.empty
    r = res.iloc[0]
    assert r["observed_difference"] < 0
    assert r["p_one_sided_preregistered"] < 0.01
    assert bool(r["prediction_supported"])


def test_entry_mode_confirmatory_is_calibrated_under_the_null():
    """No planted effect -> the pre-registered one-sided p must be uniform."""
    cfg = Config(entry_mode_n_permutations=2000)
    hits = 0
    n_rep = 60
    for rep in range(n_rep):
        rng = np.random.default_rng(500 + rep)
        names = ([f"RM_Type_II_{i}" for i in range(10)]
                 + [f"CBASS_{i}" for i in range(10)])
        comp = pd.DataFrame({
            "defense_system": names,
            "entry_mode_coefficient": rng.normal(0, 0.5, len(names)),
            "entry_mode_std_err": 0.2,
        })
        comp = comp.merge(tier3_entry_mode.assign_mechanism_groups(
            list(comp.defense_system), cfg), on="defense_system")
        res = tier3_entry_mode.run_entry_mode_confirmatory(
            comp, Config(entry_mode_n_permutations=2000, random_seed=rep), LOG)
        if not res.empty and res.iloc[0]["p_one_sided_preregistered"] < 0.05:
            hits += 1
    assert hits / n_rep < 0.20, (
        f"entry-mode confirmatory test fired on {hits}/{n_rep} null datasets")


def test_entry_mode_confirmatory_needs_both_groups():
    cfg = Config()
    comp = pd.DataFrame({
        "defense_system": ["RM_Type_II_1", "RM_Type_II_2"],
        "entry_mode_coefficient": [-0.5, -0.4],
        "entry_mode_std_err": [0.1, 0.1],
        "mechanism_group": ["predicted_dsDNA_restricting"] * 2,
    })
    assert tier3_entry_mode.run_entry_mode_confirmatory(comp, cfg, LOG).empty


# ======================================================================
# B1 — sister pairs
# ======================================================================

def test_sister_groups_capture_cherries_and_polytomies():
    p = Path(tempfile.mkdtemp()) / "poly.nwk"
    p.write_text("(((A:1,B:1):1,(C:1,D:1,E:1):1):1,(F:1,G:1):1);")
    groups = tier3_sister_pairs.extract_sister_groups(str(p), LOG)
    sets = [sorted(v) for v in groups.values()]
    assert ["A", "B"] in sets
    # Polytomy handled as a group, not silently reduced to an arbitrary cherry.
    assert ["C", "D", "E"] in sets
    assert ["F", "G"] in sets


def test_depth_matching_drops_unmatched_pairs():
    depth = np.array([0.0, 0.4, 0.0, 5.0])
    assert len(tier3_sister_pairs._match_within_group([0, 2], [1, 3], depth, 0.5)) == 1
    assert tier3_sister_pairs._match_within_group([0, 2], [1, 3], depth, 0.0) == []


def test_each_tip_used_at_most_once_per_pairing():
    depth = np.zeros(6)
    pairs = tier3_sister_pairs._match_within_group([0, 1, 2], [3, 4], depth, 1.0)
    used = [i for pr in pairs for i in pr]
    assert len(used) == len(set(used)), "a tip was reused, breaking independence"
    assert len(pairs) == 2


def test_mcnemar_direction_and_null():
    cfg = Config()
    pairs = pd.DataFrame({
        "plasmid_in_defense_pos": [1] * 30 + [0] * 5,
        "plasmid_in_defense_neg": [0] * 30 + [1] * 5,
        "log_depth_diff": [0.0] * 35,
    })
    r = tier3_sister_pairs._test_one_system("sys", pairs, cfg)
    assert r["n_plasmid_with_defense_only"] == 30
    assert r["sister_odds_ratio"] == pytest.approx(6.0)
    assert r["sister_p_value"] < 1e-4

    null = pd.DataFrame({
        "plasmid_in_defense_pos": [1] * 25 + [0] * 25,
        "plasmid_in_defense_neg": [0] * 25 + [1] * 25,
        "log_depth_diff": [0.0] * 50,
    })
    assert tier3_sister_pairs._test_one_system("sys", null, cfg)["sister_p_value"] == 1.0


def test_sister_pairs_gate_on_too_few_discordant():
    cfg = Config(sister_pair_min_discordant=20)
    pairs = pd.DataFrame({
        "plasmid_in_defense_pos": [1] * 5,
        "plasmid_in_defense_neg": [0] * 5,
        "log_depth_diff": [0.0] * 5,
    })
    r = tier3_sister_pairs._test_one_system("sys", pairs, cfg)
    assert r["skip_reason"] == "too_few_discordant_pairs"
    assert np.isnan(r["sister_p_value"])


# ======================================================================
# B2 — Pagel directionality
# ======================================================================

def _aic_frame(ind, pdd, ddp, mut, n=5):
    return pd.DataFrame({
        "pagel_aic_independent": [ind] * n,
        "pagel_aic_plasmid_drives_defense": [pdd] * n,
        "pagel_aic_defense_drives_plasmid": [ddp] * n,
        "pagel_aic_mutual": [mut] * n,
    })


@pytest.mark.parametrize("ind,pdd,ddp,mut,expected", [
    (120, 118, 108, 110, "defense_drives_plasmid"),
    (120, 108, 118, 110, "plasmid_drives_defense"),
    (120, 110, 109, 111, "ambiguous"),
    (100, 110, 111, 115, "independent_no_dependence"),
    (130, 120, 120, 110, "mutual_or_ambiguous"),
])
def test_pagel_direction_verdicts(ind, pdd, ddp, mut, expected):
    r = tier2_pagels._summarise_direction(_aic_frame(ind, pdd, ddp, mut), Config())
    assert r["pagel_direction"] == expected


def test_pagel_direction_sign_convention():
    """delta AIC > 0 must mean 'defense drives plasmid'. Getting this backwards
    would invert the paper's directional conclusion."""
    r = tier2_pagels._summarise_direction(_aic_frame(120, 118, 108, 110), Config())
    assert r["pagel_direction_delta_aic"] > 0
    assert r["pagel_direction"] == "defense_drives_plasmid"


def test_pagel_akaike_weights_sum_to_one():
    r = tier2_pagels._summarise_direction(_aic_frame(120, 118, 108, 110), Config())
    total = sum(r[f"pagel_weight_{m}"] for m in
                ("independent", "plasmid_drives_defense",
                 "defense_drives_plasmid", "mutual"))
    assert total == pytest.approx(1.0)


def test_pagel_direction_absent_columns_is_graceful():
    r = tier2_pagels._summarise_direction(pd.DataFrame({"x": [1]}), Config())
    assert r["pagel_direction"] == "not_fitted"


# ======================================================================
# B3 — matched-feature control
# ======================================================================

def test_bm_simulation_returns_all_tips(tree_file):
    rng = np.random.default_rng(0)
    bm, tips = tier3_feature_control.simulate_bm_on_tree(tree_file, rng)
    assert len(tips) == 128 and len(bm) == 128
    assert np.isfinite(bm).all()


def test_synthetic_trait_hits_target_prevalence(tree_file):
    rng = np.random.default_rng(1)
    bm, _ = tier3_feature_control.simulate_bm_on_tree(tree_file, rng)
    for target in (0.05, 0.2, 0.5, 0.85):
        t = tier3_feature_control.make_matched_binary_trait(bm, target, 1.0, rng)
        assert abs(t.mean() - target) < 0.03, target
        assert set(np.unique(t)) <= {0, 1}


def test_lambda_controls_phylogenetic_clustering(tree_file):
    """lambda is the proportion of trait variance that is phylogenetic, so
    sister-tip concordance must increase monotonically with it."""
    rng = np.random.default_rng(2)
    pairs = [(i, i + 1) for i in range(0, 128, 2)]  # sisters are adjacent
    conc = {}
    for lam in (0.0, 0.5, 1.0):
        vals = []
        for _ in range(120):
            bm, _ = tier3_feature_control.simulate_bm_on_tree(tree_file, rng)
            t = tier3_feature_control.make_matched_binary_trait(bm, 0.5, lam, rng)
            vals.append(np.mean([t[i] == t[j] for i, j in pairs]))
        conc[lam] = float(np.mean(vals))
    assert conc[0.0] < conc[0.5] < conc[1.0]
    assert conc[0.0] == pytest.approx(0.5, abs=0.06)


def test_reference_systems_span_the_prevalence_range():
    cfg = Config(feature_control_max_systems=5)
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({f"sys{i}": rng.binomial(1, p, n)
                       for i, p in enumerate(np.linspace(0.02, 0.9, 20))})
    picked = tier3_feature_control._pick_reference_systems(
        df, list(df.columns), cfg)
    prevs = sorted(df[c].mean() for c in picked)
    assert len(picked) == 5
    assert prevs[0] < 0.15 and prevs[-1] > 0.7, (
        "reference systems must span the prevalence spectrum, not cluster")


def test_feature_control_comparison_flags_ordinary_effects():
    cfg = Config()
    rng = np.random.default_rng(4)
    ctrl = pd.DataFrame({
        "synthetic_feature": [f"__synth_{i}" for i in range(200)],
        "phyloglm_coefficient": rng.normal(0, 0.3, 200),
        "phyloglm_p_value": rng.uniform(size=200),
        "realised_prevalence": rng.uniform(0.1, 0.6, 200),
    })
    prim = pd.DataFrame({
        "defense_system": ["big_effect", "ordinary_effect"],
        "phyloglm_coefficient": [3.0, 0.05],
        "phyloglm_p_value": [1e-9, 1e-9],
        "phyloglm_fdr_qvalue": [1e-8, 1e-8],
        "outcome_label": "any_plasmid",
        "direction": "plasmid_given_defense",
        "covariate_mode": "full",
        "defense_prevalence": [0.3, 0.3],
    })
    out = tier3_feature_control.build_feature_control_comparison(
        ctrl, prim, cfg, LOG).set_index("defense_system")
    assert out.loc["big_effect", "control_percentile"] > 95
    assert out.loc["ordinary_effect", "control_percentile"] < 60
    assert bool(out.loc["big_effect", "exceeds_matched_null"])
    assert not bool(out.loc["ordinary_effect", "exceeds_matched_null"])


# ======================================================================
# Config plumbing for the new stages
# ======================================================================

def test_new_stages_are_registered():
    from defense_analysis_v2 import defense_plasmid_analysis as dpa
    for stage in ("entry_mode", "sister_pairs", "feature_control"):
        assert stage in dpa.ALL_STAGES
        assert stage in dpa.DEFAULT_STAGES
        assert stage in dpa.STAGE_OUTPUTS


def test_entry_mode_defaults_point_at_the_supplied_table():
    cfg = Config()
    assert cfg.entry_mode_metadata_file.endswith("plasmid_metadata.txt")
    assert cfg.entry_mode_plasmid_id_column == "plasmid_id"
    assert cfg.entry_mode_conjugative_column == "conjugative"
    # The defensible engine is the actual binomial likelihood.
    assert cfg.entry_mode_engine == "pglmm"


def test_evalue_conversion_is_derived_from_observed_prevalence():
    """Hard-coding this was wrong: the config asserted the outcome was common
    when the measured species-level plasmid prevalence is 5.7%."""
    from defense_analysis_v2.stats_utils import resolve_rare_outcome
    assert Config().evalue_rare_outcome is None, "must be derived, not fixed"
    # 5.7% -> rare -> OR ~ RR
    assert resolve_rare_outcome(None, 0.057, 0.15) is True
    # a genuinely common outcome -> sqrt(OR)
    assert resolve_rare_outcome(None, 0.45, 0.15) is False
    # explicit override still wins
    assert resolve_rare_outcome(True, 0.45, 0.15) is True
    # unknown prevalence errs toward understating the E-value
    assert resolve_rare_outcome(None, None, 0.15) is False


def test_rare_outcome_gives_a_larger_evalue_than_common():
    from defense_analysis_v2.stats_utils import evalue_from_odds_ratio
    rare = evalue_from_odds_ratio(2.5, 1.6, 3.9, rare_outcome=True)
    common = evalue_from_odds_ratio(2.5, 1.6, 3.9, rare_outcome=False)
    assert rare["evalue_point"] > common["evalue_point"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
