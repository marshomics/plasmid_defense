"""B3 — matched-feature negative control.

The question
------------
Is there anything special about DEFENSE systems, or would any accessory gene
with the same prevalence and the same degree of phylogenetic clustering show
the same association with plasmid carriage?

This is different from the outcome-permutation negative control in
``tier3_sensitivity.run_negative_control``. That one asks "is the model
calibrated?" by scrambling the outcome. This one leaves the outcome alone and
replaces the PREDICTOR with a biologically arbitrary trait matched on the two
properties that could drive a spurious association: how common it is, and how
phylogenetically clumped it is. It calibrates the effect-size scale — if
arbitrary traits produce the same distribution of odds ratios, the "defense"
signal is not about defense, it is about being a variably-present accessory
trait in lineages that also vary in plasmid carriage.

How the synthetic traits are built
----------------------------------
A continuous trait is simulated under Brownian motion on the actual tree, then
mixed with independent noise:

    z = sqrt(lambda) * z_BM  +  sqrt(1 - lambda) * z_iid

which is exactly Pagel's lambda covariance ``lambda * C + (1 - lambda) * I``,
so lambda is interpretable as the proportion of trait variance that is
phylogenetic. Thresholding z at the quantile matching a target prevalence gives
a binary trait with both properties controlled. Sweeping lambda over a grid
spans the range from "no phylogenetic structure" to "pure Brownian".

Simulation is done by tree traversal rather than by drawing from the
phylogenetic VCV: a 40,000-tip covariance matrix is ~13 GB, and the traversal
is O(n) and exact.

If a real non-defense gene-family table is supplied
(``config.feature_control_gene_family_file``) it is used in addition, since
real gene families carry realistic co-occurrence structure that a simulated
trait does not.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .r_bridge import call_r_script
from .stats_utils import apply_fdr


# ======================================================================
# Simulating phylogenetically-structured binary traits
# ======================================================================

def simulate_bm_on_tree(tree_path: str, rng: np.random.Generator
                        ) -> Tuple[np.ndarray, List[str]]:
    """One Brownian-motion realisation at the tips. Returns (values, tips).

    Traversal-based: each edge contributes ``N(0, edge_length)`` to the running
    value. Exact, and O(n) in both time and memory.
    """
    import dendropy

    tree = dendropy.Tree.get(path=tree_path, schema="newick",
                             preserve_underscores=True,
                             suppress_internal_node_taxa=True)
    values: Dict[int, float] = {}
    tips, vals = [], []
    for node in tree.preorder_node_iter():
        parent = node.parent_node
        base = 0.0 if parent is None else values[id(parent)]
        length = node.edge_length if node.edge_length else 0.0
        # Guard against negative/NaN branch lengths from epsilon patching.
        length = max(0.0, float(length)) if np.isfinite(length) else 0.0
        values[id(node)] = base + rng.normal(0.0, np.sqrt(length)) if length > 0 else base
        if node.is_leaf() and node.taxon is not None:
            tips.append(node.taxon.label.strip().replace(" ", "_"))
            vals.append(values[id(node)])
    return np.asarray(vals, dtype=float), tips


def make_matched_binary_trait(bm: np.ndarray, prevalence: float,
                              lam: float, rng: np.random.Generator
                              ) -> np.ndarray:
    """Binary trait with the given prevalence and Pagel-lambda phylogenetic
    signal, from a pre-simulated BM realisation."""
    z = np.asarray(bm, dtype=float)
    sd = z.std()
    z = (z - z.mean()) / sd if sd > 0 else np.zeros_like(z)
    noise = rng.normal(size=z.size)
    lam = float(np.clip(lam, 0.0, 1.0))
    mixed = np.sqrt(lam) * z + np.sqrt(1.0 - lam) * noise
    # Threshold at the quantile that reproduces the target prevalence exactly.
    prevalence = float(np.clip(prevalence, 1e-6, 1 - 1e-6))
    cut = np.quantile(mixed, 1.0 - prevalence)
    return (mixed > cut).astype(int)


def _pick_reference_systems(phylo_data: pd.DataFrame, defense_cols: List[str],
                            config: Config) -> List[str]:
    """Sample real systems ACROSS the prevalence spectrum, not from the top.

    Taking the most prevalent systems would produce a control matched only to
    the common ones, and the rare systems -- where separation and instability
    actually bite -- would have no matched null at all.
    """
    prev = {c: float(phylo_data[c].mean()) for c in defense_cols
            if c in phylo_data.columns}
    prev = {c: p for c, p in prev.items() if 0.0 < p < 1.0}
    if not prev:
        return []
    n = min(int(config.feature_control_max_systems), len(prev))
    ordered = sorted(prev, key=lambda c: prev[c])
    idx = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    return [ordered[i] for i in sorted(set(idx.tolist()))]


# ======================================================================
# Driver
# ======================================================================

def run_feature_control(phylo_data: pd.DataFrame,
                        defense_cols: List[str],
                        tree_path: str,
                        config: Config,
                        logger: logging.Logger,
                        workdir: Path) -> Dict[str, pd.DataFrame]:
    """Run the primary sweep on matched synthetic (and optionally real
    non-defense) features.

    Returns ``{"results": per-synthetic-feature fits,
               "comparison": per-real-system calibration}``.
    """
    if not config.run_feature_control:
        return {"results": pd.DataFrame(), "comparison": pd.DataFrame()}

    refs = _pick_reference_systems(phylo_data, defense_cols, config)
    if not refs:
        logger.warning("Feature control skipped — no usable reference systems")
        return {"results": pd.DataFrame(), "comparison": pd.DataFrame()}

    rng = np.random.default_rng(int(config.random_seed) + 777)
    n_rep = max(1, int(config.feature_control_n_per_system))
    lam_grid = list(config.feature_control_lambda_grid) or [1.0]

    logger.info(
        f"Feature control: simulating {len(refs)} x {n_rep} = "
        f"{len(refs) * n_rep} matched traits across lambda {lam_grid}")

    # One BM realisation per replicate index, reused across reference systems.
    # Reusing is legitimate: each synthetic trait thresholds the realisation at
    # a different prevalence, and reusing keeps the tree traversal count down
    # from (systems x reps) to (reps).
    bm_by_rep, tip_order = {}, None
    for rep in range(n_rep):
        bm, tips = simulate_bm_on_tree(tree_path, rng)
        bm_by_rep[rep] = pd.Series(bm, index=tips)
        tip_order = tips

    if tip_order is None:
        return {"results": pd.DataFrame(), "comparison": pd.DataFrame()}

    d = phylo_data.copy()
    synth_cols, meta = [], []
    for si, ref in enumerate(refs):
        target_prev = float(d[ref].mean())
        for rep in range(n_rep):
            lam = lam_grid[(si + rep) % len(lam_grid)]
            bm_aligned = bm_by_rep[rep].reindex(d["tip"]).values
            if not np.isfinite(bm_aligned).all():
                bm_aligned = np.nan_to_num(bm_aligned, nan=0.0)
            name = f"__synth_{si:03d}_{rep}"
            d[name] = make_matched_binary_trait(bm_aligned, target_prev, lam,
                                                rng)
            synth_cols.append(name)
            meta.append({"synthetic_feature": name,
                         "matched_to_system": ref,
                         "target_prevalence": target_prev,
                         "realised_prevalence": float(d[name].mean()),
                         "lambda": lam})
    meta_df = pd.DataFrame(meta)

    # Optional real non-defense gene families.
    gf_cols: List[str] = []
    gf_path = str(config.feature_control_gene_family_file or "").strip()
    if gf_path:
        try:
            gf = pd.read_csv(gf_path, sep="\t", low_memory=False)
            key = "gtdb_species" if "gtdb_species" in gf.columns else gf.columns[0]
            gf = gf.drop_duplicates(subset=[key]).set_index(key)
            join_key = ("gtdb_species" if "gtdb_species" in d.columns else "tip")
            aligned = gf.reindex(d[join_key]).reset_index(drop=True)
            for c in aligned.columns:
                col = pd.to_numeric(aligned[c], errors="coerce")
                if col.notna().sum() < len(d) * 0.5:
                    continue
                binary = (col.fillna(0) > 0).astype(int)
                if 0 < binary.mean() < 1:
                    name = f"__genefam_{c}"
                    d[name] = binary.values
                    gf_cols.append(name)
            logger.info(f"Feature control: {len(gf_cols)} real non-defense "
                        f"gene families loaded from {gf_path}")
        except Exception as exc:
            logger.warning(f"Feature control: could not load gene-family "
                           f"table {gf_path}: {exc}")

    all_cols = synth_cols + gf_cols
    covariates = list(config.resolve_covariates(
        config.covariate_columns_for_mode(config.primary_covariate_mode,
                                          include_plasmid_count=False), d))

    r = call_r_script(
        "phyloglm_uni.R",
        tree_path=tree_path,
        data=d,
        args={"response": "has_plasmid_binary",
              "predictors": all_cols,
              "mode": "predictor",
              "defense_side": "predictor",
              "covariates": covariates,
              "tip_column": "tip",
              "evolutionary_model": config.phyloglm_estimator,
              "btol": 20, "boot": 0,
              "min_count": config.min_count_per_category,
              "min_count_response": config.min_count_per_category},
        logger=logger,
        r_executable=config.r_executable,
        workdir=workdir / "feature_control",
    )
    if not r.ok:
        logger.error(f"feature control sweep failed: {r.error}")
        return {"results": pd.DataFrame(), "comparison": pd.DataFrame()}

    res = r.dataframe.rename(columns={"test_label": "synthetic_feature"})
    res["phyloglm_fdr_qvalue"] = apply_fdr(res["phyloglm_p_value"],
                                           method=config.fdr_method).values
    res["feature_type"] = np.where(
        res["synthetic_feature"].astype(str).str.startswith("__genefam_"),
        "real_gene_family", "simulated_matched_trait")
    res = res.merge(meta_df, on="synthetic_feature", how="left")
    if "realised_prevalence" not in res.columns or res["realised_prevalence"].isna().all():
        res["realised_prevalence"] = np.nan
    for c in all_cols:
        if c in d.columns:
            res.loc[res["synthetic_feature"] == c, "realised_prevalence"] = \
                res.loc[res["synthetic_feature"] == c,
                        "realised_prevalence"].fillna(float(d[c].mean()))

    n_fit = int(res["phyloglm_p_value"].notna().sum())
    n_sig = int((res["phyloglm_fdr_qvalue"] < config.alpha).sum())
    logger.info(
        f"Feature control: {n_fit}/{len(res)} control features fit; "
        f"{n_sig} ({100 * n_sig / max(n_fit, 1):.1f}%) reach FDR q < "
        f"{config.alpha} despite being biologically arbitrary")
    return {"results": res, "comparison": pd.DataFrame()}


def build_feature_control_comparison(control_results: pd.DataFrame,
                                     tier2_phyloglm: pd.DataFrame,
                                     config: Config,
                                     logger: logging.Logger) -> pd.DataFrame:
    """Calibrate each real system's effect against the matched null.

    For every real defense system, the comparison set is the control features
    whose realised prevalence is closest to that system's. The reported
    percentile is where the real |coefficient| falls in that null, and the
    empirical p-value is the fraction of matched control features with an equal
    or larger |coefficient|.

    A system at the 99th percentile of its matched null is doing something an
    arbitrary trait of the same prevalence and clustering does not do. A system
    at the 60th percentile is not, whatever its q-value says.
    """
    if control_results is None or control_results.empty:
        return pd.DataFrame()
    if tier2_phyloglm is None or tier2_phyloglm.empty:
        return pd.DataFrame()

    ctrl = control_results.dropna(subset=["phyloglm_coefficient",
                                          "phyloglm_p_value"]).copy()
    if ctrl.empty:
        return pd.DataFrame()
    ctrl["abs_coef"] = ctrl["phyloglm_coefficient"].abs()

    prim = tier2_phyloglm
    if "outcome_label" in prim.columns:
        prim = prim[prim["outcome_label"] == "any_plasmid"]
    if "direction" in prim.columns:
        prim = prim[prim["direction"] == "plasmid_given_defense"]
    if "covariate_mode" in prim.columns:
        prim = prim[prim["covariate_mode"].map(config.normalise_covariate_mode)
                    == config.normalise_covariate_mode(
                        config.primary_covariate_mode)]
    prim = prim.dropna(subset=["phyloglm_coefficient", "phyloglm_p_value"])
    if prim.empty:
        return pd.DataFrame()

    ctrl_prev = ctrl["realised_prevalence"].values.astype(float)
    ctrl_abs = ctrl["abs_coef"].values.astype(float)

    rows = []
    for _, r in prim.iterrows():
        obs = abs(float(r["phyloglm_coefficient"]))
        # Match on prevalence where available; otherwise use the whole null.
        prev = np.nan
        for c in ("defense_prevalence", "n_predictor_present"):
            if c in r.index and pd.notna(r[c]):
                if c == "n_predictor_present" and pd.notna(r.get("n_species")):
                    prev = float(r[c]) / float(r["n_species"])
                else:
                    prev = float(r[c])
                break
        if np.isfinite(prev) and np.isfinite(ctrl_prev).any():
            order = np.argsort(np.abs(ctrl_prev - prev))
            k = min(len(order), max(20, len(order) // 5))
            null = ctrl_abs[order[:k]]
        else:
            null = ctrl_abs
        if null.size == 0:
            continue
        pct = float((null < obs).mean() * 100.0)
        emp_p = float(((null >= obs).sum() + 1) / (null.size + 1))
        rows.append({
            "defense_system": r.get("defense_system"),
            "phyloglm_coefficient": r["phyloglm_coefficient"],
            "phyloglm_fdr_qvalue": r.get("phyloglm_fdr_qvalue"),
            "matched_prevalence": prev,
            "n_matched_controls": int(null.size),
            "control_percentile": pct,
            "control_empirical_p": emp_p,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["control_empirical_fdr_qvalue"] = apply_fdr(
        out["control_empirical_p"], method=config.fdr_method).values
    out["exceeds_matched_null"] = out["control_empirical_fdr_qvalue"] < config.alpha

    n_exceed = int(out["exceeds_matched_null"].sum())
    logger.info(
        f"Feature control: {n_exceed}/{len(out)} defense systems have an "
        f"effect larger than arbitrary traits matched on prevalence and "
        f"phylogenetic clustering")
    return out
