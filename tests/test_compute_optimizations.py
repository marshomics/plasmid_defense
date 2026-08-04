"""Regression tests for the compute optimisations.

Each test pins a property that keeps a previously-blocked stage inside the
cluster envelope, so a future config change that reintroduces the blocker fails
here rather than 25 days into a job.
"""
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defense_analysis_v2 import cost_model, phylo_signal_fast, r_bridge  # noqa: E402
from defense_analysis_v2.config import Config  # noqa: E402

LOG = logging.getLogger("test")
LOG.addHandler(logging.NullHandler())


def _balanced(depth: int, prefix: str = "t") -> str:
    if depth == 0:
        return f"{prefix}:1"
    return (f"({_balanced(depth - 1, prefix + 'L')},"
            f"{_balanced(depth - 1, prefix + 'R')}):1")


@pytest.fixture(scope="module")
def tree_file():
    p = Path(tempfile.mkdtemp()) / "t.nwk"
    p.write_text(_balanced(9) + ";")   # 512 tips
    return str(p)


# ======================================================================
# Native D statistic
# ======================================================================

def test_d_is_one_for_a_random_trait(tree_file):
    ft = phylo_signal_fast.flatten_tree(tree_file, LOG)
    rng = np.random.default_rng(0)
    trait = np.zeros(ft.n_tips)
    trait[rng.choice(ft.n_tips, ft.n_tips // 4, replace=False)] = 1
    r = phylo_signal_fast.phylo_d(ft, trait, 300, rng)
    assert 0.7 < r["D"] < 1.3, f"random trait should give D ~ 1, got {r['D']}"


def test_d_is_zero_for_a_brownian_trait(tree_file):
    ft = phylo_signal_fast.flatten_tree(tree_file, LOG)
    rng = np.random.default_rng(1)
    bm = phylo_signal_fast.simulate_bm_tips(ft, 1, rng)
    trait = phylo_signal_fast._threshold_to_prevalence(bm, ft.n_tips // 4).ravel()
    r = phylo_signal_fast.phylo_d(ft, trait, 300, rng)
    assert -0.4 < r["D"] < 0.4, f"Brownian trait should give D ~ 0, got {r['D']}"


def test_d_is_negative_for_a_perfectly_clumped_trait(tree_file):
    ft = phylo_signal_fast.flatten_tree(tree_file, LOG)
    rng = np.random.default_rng(2)
    trait = np.zeros(ft.n_tips)
    trait[:ft.n_tips // 4] = 1          # one contiguous clade
    r = phylo_signal_fast.phylo_d(ft, trait, 300, rng)
    assert r["D"] < 0.0


def test_d_handles_constant_traits_without_crashing(tree_file):
    ft = phylo_signal_fast.flatten_tree(tree_file, LOG)
    rng = np.random.default_rng(3)
    r = phylo_signal_fast.phylo_d(ft, np.zeros(ft.n_tips), 50, rng)
    assert np.isnan(r["D"]) and r["error"] == "trait_is_constant"


def test_threshold_hits_the_exact_prevalence(tree_file):
    """Nulls must have EXACTLY the observed prevalence — a null at a different
    prevalence is not comparable, because D compares sums of differences."""
    ft = phylo_signal_fast.flatten_tree(tree_file, LOG)
    rng = np.random.default_rng(4)
    cont = rng.normal(size=(ft.n_tips, 20))
    for k in (1, 17, ft.n_tips // 3, ft.n_tips - 1):
        b = phylo_signal_fast._threshold_to_prevalence(cont, k)
        assert (b.sum(axis=0) == k).all(), k


def test_sum_of_changes_reduces_to_sister_difference(tree_file):
    """On a bifurcating node the edge-sum formulation must equal |left-right|,
    which is the textbook sister-clade difference."""
    p = Path(tempfile.mkdtemp()) / "cherry.nwk"
    p.write_text("(a:1,b:1);")
    ft = phylo_signal_fast.flatten_tree(str(p), LOG)
    s = phylo_signal_fast.sum_of_changes(ft, np.array([1.0, 0.0]))
    assert float(s[0]) == pytest.approx(1.0)
    s0 = phylo_signal_fast.sum_of_changes(ft, np.array([1.0, 1.0]))
    assert float(s0[0]) == pytest.approx(0.0)


def test_flatten_tree_levels_cover_every_node(tree_file):
    ft = phylo_signal_fast.flatten_tree(tree_file, LOG)
    covered = sorted(int(i) for lvl in ft.levels for i in lvl)
    assert covered == list(range(ft.n_nodes))
    # Children must be strictly deeper than parents, which is what makes the
    # level-wise post-order sweep correct.
    nonroot = ft.parent >= 0
    assert (ft.depth[nonroot] > ft.depth[ft.parent[nonroot]]).all()


def test_native_d_is_the_default_engine():
    assert Config().phylo_signal_engine == "native"


# ======================================================================
# Shared-frame bridge
# ======================================================================

def test_shared_frame_written_once():
    d = Path(tempfile.mkdtemp())
    frame = pd.DataFrame({"tip": [f"t{i}" for i in range(50)],
                          "x": range(50)})
    sf = r_bridge.write_shared_frame(frame, d, "s", LOG)
    assert sf.path.exists() and sf.n_rows == 50
    mtime = sf.path.stat().st_mtime
    r_bridge.SharedFrame(frame, sf.path)      # must not rewrite
    assert sf.path.stat().st_mtime == mtime


def test_call_r_script_requires_data_or_shared():
    with pytest.raises(ValueError, match="data= or shared="):
        r_bridge.call_r_script("phyloglm_uni.R", tree_path="t.nwk",
                               args={}, logger=LOG)


def test_shared_frame_call_writes_only_small_side_files(monkeypatch):
    """The whole point: a shared call must not re-serialise the big frame."""
    d = Path(tempfile.mkdtemp())
    big = pd.DataFrame({"tip": [f"t{i}" for i in range(2000)],
                        **{f"c{j}": np.zeros(2000) for j in range(50)}})
    sf = r_bridge.write_shared_frame(big, d, "big", LOG)

    captured = {}

    class _P:
        returncode, stdout, stderr = 1, "", "stub"

    def fake_run(cmd, **kw):
        captured["data_arg"] = cmd[4]
        return _P()

    monkeypatch.setattr(r_bridge.subprocess, "run", fake_run)
    wd = d / "call1"
    r_bridge.call_r_script(
        "phyloglm_uni.R", tree_path="t.nwk", shared=sf,
        overrides=pd.DataFrame({"tip": big["tip"], "y": np.ones(2000)}),
        args={}, logger=LOG, workdir=wd)

    # R was pointed at the SHARED path, not a per-call copy.
    assert captured["data_arg"] == str(sf.path)
    assert not (wd / "data.tsv").exists()
    assert (wd / "overrides.tsv").exists()
    assert (wd / "overrides.tsv").stat().st_size < sf.path.stat().st_size / 5


def test_signal_deaths_are_retried_but_r_errors_are_not(monkeypatch):
    calls = {"n": 0}

    class _P:
        def __init__(self, rc):
            self.returncode, self.stdout, self.stderr = rc, "", ""

    def fake_sigbus(cmd, **kw):
        calls["n"] += 1
        return _P(-7)                     # SIGBUS

    monkeypatch.setattr(r_bridge.subprocess, "run", fake_sigbus)
    monkeypatch.setattr(r_bridge.time, "sleep", lambda *_: None)
    r_bridge.call_r_script("phyloglm_uni.R", tree_path="t",
                           data=pd.DataFrame({"tip": ["a"]}), args={},
                           logger=LOG, max_retries=2)
    assert calls["n"] == 3, "SIGBUS should be retried max_retries times"

    calls["n"] = 0

    def fake_error(cmd, **kw):
        calls["n"] += 1
        return _P(1)                      # deterministic R error

    monkeypatch.setattr(r_bridge.subprocess, "run", fake_error)
    r_bridge.call_r_script("phyloglm_uni.R", tree_path="t",
                           data=pd.DataFrame({"tip": ["a"]}), args={},
                           logger=LOG, max_retries=2)
    assert calls["n"] == 1, "R-level errors are deterministic; never retry"


# ======================================================================
# Scope reductions
# ======================================================================

def test_scope_defaults_are_on():
    c = Config()
    assert c.loco_covariate_modes_primary_only
    assert c.loco_fit_only_gated_clades
    assert c.misclass_restrict_to_significant
    assert c.misclass_primary_mode_only
    assert c.misclass_use_reduced_grid
    assert c.pagels_directional_only_if_dependent
    assert c.entry_mode_batch_in_r


def test_misclass_reduced_settings_cut_the_fit_count():
    c = Config()
    full = len(c.misclass_fnr_grid) * c.misclass_n_replicates * len(c.covariate_modes)
    reduced = (len(c.misclass_fnr_grid_reduced)
               * c.misclass_n_replicates_effective * 1)
    assert reduced < full / 10, (full, reduced)


def test_median_is_stable_at_the_reduced_replicate_count():
    """The MC reports a MEDIAN coefficient per FNR level. 40 draws must leave
    Monte Carlo error small relative to the coefficient's own SE."""
    rng = np.random.default_rng(0)
    coef_sd = 0.20                     # a typical phyloglm SE
    spread = []
    for _ in range(400):
        draws = rng.normal(0.5, coef_sd, 40)
        spread.append(np.median(draws))
    mc_se = float(np.std(spread))
    assert mc_se < coef_sd / 4, (
        f"median over 40 draws has MC SE {mc_se:.3f} vs coefficient SE "
        f"{coef_sd}; reduce is not safe")


# ======================================================================
# Cost model
# ======================================================================

def test_every_stage_fits_the_cluster_envelope_at_defaults():
    df = cost_model.estimate_pipeline_cost(
        Config(), n_species=39681, n_defense=435, n_outcomes=17, n_jobs=20)
    bad = df[~(df["fits_wallclock"] & df["fits_memory"])]
    assert bad.empty, f"stages exceed the envelope: {bad['stage'].tolist()}"


def test_cost_model_flags_caper_as_infeasible():
    df = cost_model.estimate_pipeline_cost(
        Config(phylo_signal_engine="caper"), 39681, 435, 17, 20,
        stages=["phylo_signal"])
    assert not bool(df.iloc[0]["fits_wallclock"])


def test_cost_model_flags_full_tree_pglmm_as_out_of_memory():
    df = cost_model.estimate_pipeline_cost(
        Config(pglmm_max_species=None), 39681, 435, 17, 20, stages=["pglmm_mv"])
    assert not bool(df.iloc[0]["fits_memory"]), (
        "full-tree PGLMM needs ~175 GB and must be flagged against the "
        "128 GB node ceiling")


def test_optimisations_materially_reduce_projected_cost():
    optimised = cost_model.estimate_pipeline_cost(
        Config(), 39681, 435, 17, 20)
    legacy = cost_model.estimate_pipeline_cost(
        Config(loco_covariate_modes_primary_only=False,
               loco_ranks_primary_only=False,
               loco_fit_only_gated_clades=False,
               misclass_restrict_to_significant=False,
               misclass_primary_mode_only=False,
               misclass_use_reduced_grid=False,
               pagels_directional_only_if_dependent=False,
               pagels_directional_primary_outcomes_only=False,
               entry_mode_batch_in_r=False,
               phylo_signal_engine="caper"),
        39681, 435, 17, 20, n_clades_gated=200)
    assert optimised["parallel_hours"].sum() < legacy["parallel_hours"].sum() / 3


def test_cost_report_renders():
    df = cost_model.estimate_pipeline_cost(Config(), 39681, 435, 17, 20)
    txt = cost_model.format_cost_report(df, 20)
    assert "PROJECTED COST" in txt and "TOTAL" in txt


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ======================================================================
# Depth-sensitivity contrast gate (count-based, not proportion-based)
# ======================================================================

def test_depth_gate_keeps_the_low_depth_band_at_realistic_prevalence():
    """Overall plasmid prevalence is 5.7%, so the LOW-depth band sits near 1%
    purely because a one-strain species has one chance to carry a plasmid.
    That band is the only place saturation cannot have operated, and a
    proportion floor of 5% would have discarded it despite ~200 positives."""
    c = Config()
    n_species, prev = 21_824, 0.009          # low-depth band
    n_pos = int(n_species * prev)
    minority = min(n_pos, n_species - n_pos)
    assert minority >= c.depth_sens_min_outcome_count
    assert (c.depth_sens_min_outcome_fraction <= prev
            <= c.depth_sens_max_outcome_fraction), "low band must survive"
    # The old proportion band would have thrown it away.
    lo, _ = c.depth_sens_outcome_prevalence_bounds
    assert prev < lo, "fixture no longer reproduces the original failure"


def test_depth_gate_still_rejects_a_saturated_subset():
    """The gate must still catch its original target: a subset driven to
    ~99% positive, which has no contrast left to fit."""
    c = Config()
    n_species, prev = 5_000, 0.995
    n_pos = int(n_species * prev)
    minority = min(n_pos, n_species - n_pos)
    assert not (minority >= c.depth_sens_min_outcome_count
                and c.depth_sens_min_outcome_fraction <= prev
                <= c.depth_sens_max_outcome_fraction)


def test_depth_gate_rejects_too_few_positives():
    c = Config()
    n_species, n_pos = 10_000, 12
    minority = min(n_pos, n_species - n_pos)
    assert minority < c.depth_sens_min_outcome_count
