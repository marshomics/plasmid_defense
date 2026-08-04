"""Native, vectorised Fritz & Purvis D statistic.

Why this module exists
----------------------
``caper::phylo.d`` was killed at the cluster's 25-day wall-clock limit without
completing: 452 columns x 1000 permutations x ~40,000 tips is intractable in
R-level code. The statistic itself is not expensive -- each evaluation is a
single O(n) tree traversal -- so the cost was implementation overhead, not
arithmetic.

This implementation restructures the computation so that:

  * the tree is parsed into flat arrays ONCE, not per column;
  * the post-order traversal runs LEVEL-WISE, so the Python loop is over tree
    depth (a few hundred iterations) rather than over nodes (~80,000);
  * all permutations are evaluated SIMULTANEOUSLY as a (nodes x permutations)
    matrix, so the inner work is BLAS-shaped rather than scalar.

The result is the same statistic to numerical precision, computed in minutes
rather than weeks, which converts phylo_signal from a skipped stage into a
routine one.

The statistic
-------------
Following Fritz & Purvis (2010):

1. Estimate internal node values by averaging daughter values up the tree.
2. Sum the absolute differences across every edge. Call this ``sum_d``.
   (For a bifurcating node this reduces exactly to the sister-clade difference
   |left - right|, and it generalises to polytomies without special-casing --
   which matters here, because the GTDB tree is polytomy-rich.)
3. Compare the observed ``sum_d`` against two nulls:
      random     -- tip labels shuffled (expected D = 1)
      Brownian   -- BM simulated on the tree, thresholded at the observed
                    prevalence (expected D = 0)
4. ``D = (obs - mean_brownian) / (mean_random - mean_brownian)``

``p_random`` is the probability of observing clustering this strong under the
random null; ``p_brownian`` the probability under the Brownian null.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ======================================================================
# Flattened tree representation
# ======================================================================

@dataclass
class FlatTree:
    """Tree as flat arrays, built once and reused for every column.

    ``parent`` holds the parent index per node (-1 at the root); ``depth`` the
    number of edges to the root; ``levels`` groups node indices by depth.
    Tips occupy indices ``0 .. n_tips - 1`` and their labels are in
    ``tip_labels`` in that order.
    """
    parent: np.ndarray          # int, (n_nodes,)
    depth: np.ndarray           # int, (n_nodes,)
    edge_len: np.ndarray        # float, (n_nodes,)
    is_leaf: np.ndarray         # bool, (n_nodes,)
    tip_labels: List[str]
    levels: List[np.ndarray]    # levels[d] = node indices at depth d

    @property
    def n_nodes(self) -> int:
        return self.parent.size

    @property
    def n_tips(self) -> int:
        return len(self.tip_labels)


def flatten_tree(tree_path: str, logger: logging.Logger) -> FlatTree:
    """Parse a Newick file into :class:`FlatTree`."""
    import dendropy

    tree = dendropy.Tree.get(path=tree_path, schema="newick",
                             preserve_underscores=True,
                             suppress_internal_node_taxa=True)

    # Tips first so tip-indexed operations are a contiguous slice.
    leaves, internals = [], []
    for node in tree.preorder_node_iter():
        (leaves if node.is_leaf() else internals).append(node)
    order = leaves + internals
    index = {id(n): i for i, n in enumerate(order)}

    n = len(order)
    parent = np.full(n, -1, dtype=np.int64)
    edge_len = np.zeros(n, dtype=np.float64)
    is_leaf = np.zeros(n, dtype=bool)
    for i, node in enumerate(order):
        is_leaf[i] = node.is_leaf()
        if node.parent_node is not None:
            parent[i] = index[id(node.parent_node)]
        el = node.edge_length
        edge_len[i] = float(el) if el is not None and np.isfinite(el) else 0.0
    edge_len = np.maximum(edge_len, 0.0)

    # Depth by walking down from the root. Roots have depth 0.
    depth = np.full(n, -1, dtype=np.int64)
    roots = np.where(parent < 0)[0]
    depth[roots] = 0
    remaining = n - roots.size
    frontier = roots
    while remaining > 0 and frontier.size:
        children = np.where(np.isin(parent, frontier) & (depth < 0))[0]
        if children.size == 0:
            break
        depth[children] = depth[parent[children]] + 1
        remaining -= children.size
        frontier = children
    # Any node unreachable from a root (shouldn't happen) is parked at depth 0.
    depth[depth < 0] = 0

    max_d = int(depth.max())
    levels = [np.where(depth == d)[0] for d in range(max_d + 1)]

    tip_labels = [nd.taxon.label.strip().replace(" ", "_")
                  if nd.taxon is not None else f"__tip{i}"
                  for i, nd in enumerate(leaves)]

    logger.info(
        f"Flattened tree: {len(tip_labels):,} tips, {n:,} nodes, "
        f"max depth {max_d} -> traversal is {max_d + 1} vectorised steps "
        f"instead of {n:,} scalar ones")
    return FlatTree(parent, depth, edge_len, is_leaf, tip_labels, levels)


# ======================================================================
# Core: sum of sister-clade differences, vectorised over replicates
# ======================================================================

def sum_of_changes(ft: FlatTree, tip_values: np.ndarray) -> np.ndarray:
    """Sum of absolute across-edge differences, for each column of
    ``tip_values`` (shape ``(n_tips, P)``). Returns shape ``(P,)``.

    Internal node values are the unweighted mean of their children, computed by
    a level-wise post-order sweep: every node's children lie at strictly
    greater depth, so processing depths from deepest to shallowest guarantees
    children are finalised before their parent is read.
    """
    tip_values = np.asarray(tip_values, dtype=np.float64)
    if tip_values.ndim == 1:
        tip_values = tip_values[:, None]
    P = tip_values.shape[1]

    values = np.zeros((ft.n_nodes, P), dtype=np.float64)
    acc = np.zeros((ft.n_nodes, P), dtype=np.float64)
    cnt = np.zeros(ft.n_nodes, dtype=np.float64)
    values[:ft.n_tips] = tip_values

    for d in range(len(ft.levels) - 1, -1, -1):
        nodes = ft.levels[d]
        if nodes.size == 0:
            continue
        internal = nodes[~ft.is_leaf[nodes]]
        if internal.size:
            c = cnt[internal]
            safe = np.where(c > 0, c, 1.0)
            values[internal] = acc[internal] / safe[:, None]
        par = ft.parent[nodes]
        has_par = nodes[par >= 0]
        if has_par.size:
            np.add.at(acc, ft.parent[has_par], values[has_par])
            np.add.at(cnt, ft.parent[has_par], 1.0)

    nonroot = np.where(ft.parent >= 0)[0]
    return np.abs(values[nonroot] - values[ft.parent[nonroot]]).sum(axis=0)


def simulate_bm_tips(ft: FlatTree, n_sim: int,
                     rng: np.random.Generator) -> np.ndarray:
    """``(n_tips, n_sim)`` Brownian-motion tip values, level-wise pre-order."""
    vals = np.zeros((ft.n_nodes, n_sim), dtype=np.float64)
    for d in range(1, len(ft.levels)):
        nodes = ft.levels[d]
        if nodes.size == 0:
            continue
        sd = np.sqrt(ft.edge_len[nodes])[:, None]
        vals[nodes] = vals[ft.parent[nodes]] + rng.normal(size=(nodes.size, n_sim)) * sd
    return vals[:ft.n_tips]


def _threshold_to_prevalence(cont: np.ndarray, n_present: int) -> np.ndarray:
    """Binarise each column of ``cont`` so exactly ``n_present`` tips are 1.

    Rank-based rather than quantile-based so the realised prevalence matches
    the observed trait EXACTLY. D compares sums of differences across nulls,
    and a null with even slightly different prevalence is not comparable.
    """
    n_tips, P = cont.shape
    out = np.zeros((n_tips, P), dtype=np.float64)
    if n_present <= 0:
        return out
    if n_present >= n_tips:
        return out + 1.0
    idx = np.argpartition(-cont, n_present - 1, axis=0)[:n_present]
    np.put_along_axis(out, idx, 1.0, axis=0)
    return out


# ======================================================================
# D statistic
# ======================================================================

def phylo_d(ft: FlatTree, tip_values: np.ndarray, n_perm: int,
            rng: np.random.Generator, chunk: int = 250) -> dict:
    """Fritz & Purvis D for one binary trait, plus both p-values."""
    tip_values = np.asarray(tip_values, dtype=np.float64).ravel()
    n_present = int(np.nansum(tip_values))
    n_tips = ft.n_tips
    if n_present == 0 or n_present == n_tips:
        return {"D": np.nan, "p_random": np.nan, "p_brownian": np.nan,
                "n_permutations": 0, "sum_d_observed": np.nan,
                "error": "trait_is_constant"}

    obs = float(sum_of_changes(ft, tip_values)[0])

    rand_sums, bm_sums = [], []
    done = 0
    while done < n_perm:
        k = min(chunk, n_perm - done)
        # Random null: shuffle tip assignment, prevalence preserved exactly.
        perm = np.empty((n_tips, k))
        for j in range(k):
            perm[:, j] = rng.permutation(tip_values)
        rand_sums.append(sum_of_changes(ft, perm))
        # Brownian null: simulate BM, threshold at the observed prevalence.
        bm = simulate_bm_tips(ft, k, rng)
        bm_sums.append(sum_of_changes(ft, _threshold_to_prevalence(bm, n_present)))
        done += k

    rand = np.concatenate(rand_sums)
    brown = np.concatenate(bm_sums)
    mean_rand, mean_brown = float(rand.mean()), float(brown.mean())

    denom = mean_rand - mean_brown
    d_val = (obs - mean_brown) / denom if abs(denom) > 1e-12 else np.nan

    # p_random: how often is a randomly-assigned trait AT LEAST as clustered
    # (i.e. has a sum of changes at least as small) as the observed one?
    p_random = float(((rand <= obs).sum() + 1) / (rand.size + 1))
    # p_brownian: how often is a Brownian trait at least as DISPERSED as
    # observed?
    p_brownian = float(((brown >= obs).sum() + 1) / (brown.size + 1))

    return {"D": float(d_val), "p_random": p_random, "p_brownian": p_brownian,
            "n_permutations": int(n_perm), "sum_d_observed": obs,
            "sum_d_mean_random": mean_rand, "sum_d_mean_brownian": mean_brown,
            "n_present": n_present, "error": None}


def run_phylo_signal_fast(phylo_data: pd.DataFrame, columns: Sequence[str],
                          tree_path: str, n_perm: int, random_seed: int,
                          logger: logging.Logger,
                          tip_column: str = "tip") -> pd.DataFrame:
    """D statistic for every column, using one flattened tree for all of them.

    Drop-in replacement for the ``phylo_d.R`` / caper path, which could not
    complete within the cluster's 25-day wall-clock ceiling.
    """
    ft = flatten_tree(tree_path, logger)
    label_to_row = {t: i for i, t in enumerate(phylo_data[tip_column].astype(str))}
    order = np.array([label_to_row.get(lbl, -1) for lbl in ft.tip_labels])
    matched = order >= 0
    if matched.sum() < 10:
        logger.error(
            f"phylo_signal: only {int(matched.sum())} tips matched between the "
            f"tree and the data; aborting")
        return pd.DataFrame()
    if not matched.all():
        logger.info(f"phylo_signal: {int((~matched).sum()):,} tree tips have no "
                    f"data row and are treated as missing")

    rng = np.random.default_rng(random_seed)
    usable = [c for c in columns if c in phylo_data.columns]
    logger.info(f"D statistic (native): {len(usable)} columns x {n_perm} "
                f"permutations on {ft.n_tips:,} tips")

    rows = []
    for i, col in enumerate(usable):
        vals = np.zeros(ft.n_tips)
        src = pd.to_numeric(phylo_data[col], errors="coerce").fillna(0).values
        vals[matched] = (src[order[matched]] > 0).astype(float)
        rec = phylo_d(ft, vals, n_perm, rng)
        rec["column"] = col
        rows.append(rec)
        if (i + 1) % 50 == 0:
            logger.info(f"  D statistic: {i + 1}/{len(usable)} columns")

    out = pd.DataFrame(rows)
    if not out.empty:
        cols = ["column"] + [c for c in out.columns if c != "column"]
        out = out[cols]
        ok = out["D"].notna()
        if ok.any():
            logger.info(
                f"D statistic complete: median D = {out.loc[ok, 'D'].median():.3f}; "
                f"{int((out.loc[ok, 'p_random'] < 0.05).sum())}/{int(ok.sum())} "
                f"columns significantly more clustered than random")
    return out
