"""B1 — phylogenetically matched sister-pair (cherry) design.

The idea
--------
Sister species share ancestry almost completely. A within-pair contrast
therefore removes phylogenetic confounding BY CONSTRUCTION rather than by
model, and because pair members can additionally be required to have similar
sequencing depth, it removes the sampling-depth confound by construction too --
the thing the depth spline can only model away. This is the comparative-
genomics equivalent of a discordant-twin study, and it is the most defensible
result obtainable from the species-level tables as they stand.

The trade is power. Only systems with many discordant, depth-matched pairs are
testable, and the effective sample size is the number of *doubly discordant*
pairs, not the number of species. That is the correct trade: a system that
survives here is a headline finding, and one that does not survive is not
thereby refuted -- it may simply lack pairs.

Pair construction
-----------------
Pairs are formed from the DIRECT LEAF CHILDREN of each internal node, not from
strict bifurcating cherries. This matters because ``tree_utils`` resolves
polytomies arbitrarily with epsilon-length branches: under arbitrary
resolution, which tips end up as a bifurcating "cherry" is an artefact of the
resolution order, and restricting to cherries would silently discard most of a
polytomy's tips while keeping an arbitrary subset. Grouping by parent node
recovers the biologically meaningful set -- tips that are equally closely
related -- and reduces to the cherry case on a genuinely bifurcating node.

Each tip is used in at most one pair per system, so pairs are independent and
an exact binomial test is valid without further correction.

Tests
-----
Primary: exact McNemar (binomial test on the doubly-discordant pairs). No
distributional assumptions beyond pair independence.

Secondary: conditional logistic regression adjusting for the residual
within-pair depth difference, for the case where depth matching leaves a small
imbalance. Reported alongside; if the two disagree, the depth matching was not
tight enough and ``sister_pair_max_log_depth_diff`` should be reduced.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .config import Config
from .stats_utils import apply_fdr


# ======================================================================
# Pair extraction from the tree
# ======================================================================

def extract_sister_groups(tree_path: str,
                          logger: logging.Logger) -> Dict[str, List[str]]:
    """Return ``{internal_node_id: [direct leaf children]}`` for every internal
    node with at least two direct leaf children.

    Uses dendropy, which is already a pipeline dependency. Tip labels are
    normalised to underscore form to match the rest of the pipeline.
    """
    import dendropy

    tree = dendropy.Tree.get(path=tree_path, schema="newick",
                             preserve_underscores=True,
                             suppress_internal_node_taxa=True)
    groups: Dict[str, List[str]] = {}
    for i, node in enumerate(tree.preorder_node_iter()):
        if node.is_leaf():
            continue
        leaf_children = [c.taxon.label.strip().replace(" ", "_")
                         for c in node.child_node_iter()
                         if c.is_leaf() and c.taxon is not None]
        if len(leaf_children) >= 2:
            groups[f"node{i}"] = leaf_children

    n_tips = sum(len(v) for v in groups.values())
    logger.info(
        f"Sister groups: {len(groups):,} internal nodes with >= 2 direct leaf "
        f"children, covering {n_tips:,} tips")
    return groups


def _match_within_group(pos_idx: List[int], neg_idx: List[int],
                        depth: np.ndarray,
                        max_diff: float) -> List[Tuple[int, int]]:
    """Greedy depth-matched pairing of defense-positive to defense-negative
    tips within one sister group.

    Both sides are sorted by depth and matched in order, which is the optimal
    1:1 assignment for minimising total absolute difference on a single
    continuous covariate. Pairs exceeding ``max_diff`` are dropped rather than
    kept, because an unmatched pair reintroduces exactly the confound the
    design exists to eliminate.
    """
    pos = sorted(pos_idx, key=lambda i: depth[i])
    neg = sorted(neg_idx, key=lambda i: depth[i])
    pairs: List[Tuple[int, int]] = []
    i = j = 0
    while i < len(pos) and j < len(neg):
        d = abs(depth[pos[i]] - depth[neg[j]])
        if d <= max_diff:
            pairs.append((pos[i], neg[j]))
            i += 1
            j += 1
        elif depth[pos[i]] < depth[neg[j]]:
            i += 1
        else:
            j += 1
    return pairs


def build_pairs_for_system(system: str,
                           frame: pd.DataFrame,
                           groups: Dict[str, List[str]],
                           tip_to_row: Dict[str, int],
                           config: Config) -> pd.DataFrame:
    """Depth-matched pairs discordant for ``system``. One row per pair."""
    defense = frame[system].values
    plasmid = frame["has_plasmid_binary"].values
    depth = (frame["log_n_strains"].values
             if "log_n_strains" in frame.columns
             else np.zeros(len(frame)))

    recs = []
    for node_id, tips in groups.items():
        rows = [tip_to_row[t] for t in tips if t in tip_to_row]
        if len(rows) < 2:
            continue
        pos = [r for r in rows if defense[r] == 1]
        neg = [r for r in rows if defense[r] == 0]
        if not pos or not neg:
            continue
        for p, n in _match_within_group(
                pos, neg, depth, config.sister_pair_max_log_depth_diff):
            recs.append({
                "node": node_id,
                "tip_defense_pos": frame.iloc[p]["tip"],
                "tip_defense_neg": frame.iloc[n]["tip"],
                "plasmid_in_defense_pos": int(plasmid[p]),
                "plasmid_in_defense_neg": int(plasmid[n]),
                "log_depth_defense_pos": float(depth[p]),
                "log_depth_defense_neg": float(depth[n]),
                "log_depth_diff": float(depth[p] - depth[n]),
            })
    return pd.DataFrame(recs)


# ======================================================================
# Tests
# ======================================================================

def _conditional_logistic(pairs: pd.DataFrame) -> dict:
    """Conditional logistic on plasmid-discordant pairs, adjusting for the
    residual within-pair depth difference.

    For 1:1 matched pairs the conditional likelihood reduces to a logistic
    regression on the outcome-discordant pairs with all responses set to 1, the
    covariates entered as within-pair differences, and NO intercept. The
    coefficient on the defense difference is the conditional log odds ratio.
    """
    import statsmodels.api as sm

    disc = pairs[pairs["plasmid_in_defense_pos"]
                 != pairs["plasmid_in_defense_neg"]].copy()
    if len(disc) < 10:
        return {}
    # +1 when the defense-positive member is the plasmid-carrying one.
    disc["delta_defense"] = np.where(disc["plasmid_in_defense_pos"] == 1, 1.0, -1.0)
    # Depth difference oriented the same way (case minus control).
    disc["delta_depth"] = np.where(disc["plasmid_in_defense_pos"] == 1,
                                   disc["log_depth_diff"],
                                   -disc["log_depth_diff"])
    X = disc[["delta_defense", "delta_depth"]].values.astype(float)
    y = np.ones(len(disc))
    try:
        m = sm.Logit(y, X).fit(disp=0, maxiter=200)
        return {
            "clogit_coefficient": float(m.params[0]),
            "clogit_std_err": float(m.bse[0]),
            "clogit_p_value": float(m.pvalues[0]),
            "clogit_odds_ratio": float(np.exp(m.params[0])),
            "clogit_depth_coefficient": float(m.params[1]),
        }
    except Exception:
        return {}


def _test_one_system(system: str, pairs: pd.DataFrame,
                     config: Config) -> dict:
    n_pairs = len(pairs)
    if n_pairs == 0:
        return {"defense_system": system, "n_pairs": 0,
                "skip_reason": "no_discordant_depth_matched_pairs"}

    a = int(((pairs["plasmid_in_defense_pos"] == 1)
             & (pairs["plasmid_in_defense_neg"] == 0)).sum())
    b = int(((pairs["plasmid_in_defense_pos"] == 0)
             & (pairs["plasmid_in_defense_neg"] == 1)).sum())
    n_disc = a + b

    rec = {
        "defense_system": system,
        "n_pairs": n_pairs,
        "n_discordant_pairs": n_disc,
        "n_plasmid_with_defense_only": a,
        "n_plasmid_without_defense_only": b,
        "mean_abs_log_depth_diff": float(pairs["log_depth_diff"].abs().mean()),
    }
    if n_disc < config.sister_pair_min_discordant:
        rec["skip_reason"] = "too_few_discordant_pairs"
        rec["sister_p_value"] = np.nan
        return rec

    # Exact McNemar. Under H0 each discordant pair is equally likely to fall
    # either way, so a ~ Binomial(n_disc, 0.5).
    rec["sister_p_value"] = float(stats.binomtest(a, n_disc, 0.5).pvalue)
    rec["sister_odds_ratio"] = float(a / b) if b > 0 else np.inf
    rec["sister_log_odds_ratio"] = (float(np.log(a / b))
                                    if a > 0 and b > 0 else np.nan)
    # Wilson interval on the discordant proportion, mapped to the OR scale.
    lo, hi = _wilson(a, n_disc)
    rec["sister_or_ci_low"] = float(lo / (1 - lo)) if 0 < lo < 1 else np.nan
    rec["sister_or_ci_high"] = float(hi / (1 - hi)) if 0 < hi < 1 else np.nan
    rec["skip_reason"] = np.nan

    if config.sister_pair_conditional_logistic:
        rec.update(_conditional_logistic(pairs))
    return rec


def _wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ======================================================================
# Driver
# ======================================================================

def run_sister_pairs(phylo_data: pd.DataFrame,
                     defense_cols: List[str],
                     tree_path: str,
                     config: Config,
                     logger: logging.Logger,
                     workdir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Run the sister-pair analysis for every defense system.

    Returns ``{"summary": per-system results, "pairs": the pair table for the
    systems that were testable}``.
    """
    if not config.run_sister_pairs:
        return {"summary": pd.DataFrame(), "pairs": pd.DataFrame()}
    if "tip" not in phylo_data.columns:
        logger.warning("Sister pairs skipped — no 'tip' column")
        return {"summary": pd.DataFrame(), "pairs": pd.DataFrame()}

    groups = extract_sister_groups(tree_path, logger)
    if not groups:
        logger.warning("Sister pairs skipped — no sister groups found in tree")
        return {"summary": pd.DataFrame(), "pairs": pd.DataFrame()}

    frame = phylo_data.reset_index(drop=True)
    tip_to_row = {t: i for i, t in enumerate(frame["tip"])}

    logger.info(
        f"Sister-pair analysis: {len(defense_cols)} systems, depth-match "
        f"tolerance |dlog(n_strains)| <= {config.sister_pair_max_log_depth_diff}, "
        f"minimum {config.sister_pair_min_discordant} discordant pairs")

    summaries, pair_frames = [], []
    for system in defense_cols:
        if system not in frame.columns:
            continue
        pairs = build_pairs_for_system(system, frame, groups, tip_to_row, config)
        rec = _test_one_system(system, pairs, config)
        summaries.append(rec)
        if not pairs.empty and pd.isna(rec.get("skip_reason", np.nan)):
            pairs = pairs.copy()
            pairs["defense_system"] = system
            pair_frames.append(pairs)

    summary = pd.DataFrame(summaries)
    if summary.empty:
        return {"summary": summary, "pairs": pd.DataFrame()}

    summary["sister_fdr_qvalue"] = apply_fdr(
        summary["sister_p_value"], method=config.fdr_method).values

    n_testable = int(summary["sister_p_value"].notna().sum())
    n_sig = int((summary["sister_fdr_qvalue"] < config.alpha).sum())
    logger.info(
        f"Sister pairs: {n_testable}/{len(summary)} systems had enough "
        f"depth-matched discordant pairs; {n_sig} significant at FDR q < "
        f"{config.alpha}")
    if n_testable:
        logger.info(
            f"  median discordant pairs per testable system: "
            f"{summary['n_discordant_pairs'].median():.0f}; "
            f"median |depth difference| within pairs: "
            f"{summary['mean_abs_log_depth_diff'].median():.3f}")

    pairs_out = (pd.concat(pair_frames, ignore_index=True)
                 if pair_frames else pd.DataFrame())
    return {"summary": summary, "pairs": pairs_out}


def compare_sister_to_primary(sister_summary: pd.DataFrame,
                              tier2_phyloglm: pd.DataFrame,
                              config: Config) -> pd.DataFrame:
    """Concordance between the sister-pair result and the primary regression.

    This is the interesting comparison. The sister-pair design controls
    phylogeny and depth by construction; the regression controls them by model.
    Where they agree, the model is doing its job. Where the regression is
    significant and the sister-pair test is not *despite adequate pairs*, the
    regression's control is suspect.
    """
    if sister_summary is None or sister_summary.empty:
        return pd.DataFrame()
    if tier2_phyloglm is None or tier2_phyloglm.empty:
        return pd.DataFrame()

    prim = tier2_phyloglm
    if "outcome_label" in prim.columns:
        prim = prim[prim["outcome_label"] == "any_plasmid"]
    if "direction" in prim.columns:
        prim = prim[prim["direction"] == "plasmid_given_defense"]
    if "covariate_mode" in prim.columns:
        prim = prim[prim["covariate_mode"].map(
            config.normalise_covariate_mode)
            == config.normalise_covariate_mode(config.primary_covariate_mode)]
    keep = [c for c in ("defense_system", "phyloglm_coefficient",
                        "phyloglm_fdr_qvalue") if c in prim.columns]
    if len(keep) < 3:
        return pd.DataFrame()

    m = sister_summary.merge(prim[keep], on="defense_system", how="inner")
    if m.empty:
        return m
    a = config.alpha
    sig_s = m["sister_fdr_qvalue"] < a
    sig_p = m["phyloglm_fdr_qvalue"] < a
    testable = m["sister_p_value"].notna()
    same_sign = (np.sign(m["sister_log_odds_ratio"])
                 == np.sign(m["phyloglm_coefficient"]))

    m["sister_verdict"] = np.select(
        [sig_s & sig_p & same_sign,
         sig_s & sig_p & ~same_sign,
         ~sig_s & sig_p & testable,
         sig_s & ~sig_p,
         ~testable],
        ["confirmed_by_matched_pairs",
         "direction_conflict",
         "regression_only__matched_pairs_do_not_confirm",
         "matched_pairs_only",
         "untestable_insufficient_pairs"],
        default="ns_both")
    return m
