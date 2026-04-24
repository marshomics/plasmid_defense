"""Phylogenetic tree loading, tip matching, and preprocessing.

The raw GTDB species tree uses tip labels with various formatting conventions
(spaces vs. underscores, quoted vs. unquoted). ``match_species_to_tree`` tries
several normalisations and picks whichever matches the most species to avoid
silently dropping half the dataset.

Tree preprocessing (zero-length-branch fix, polytomy resolution) happens once,
on the Python side using dendropy, so that every downstream R call receives
a well-conditioned tree.
"""

import logging
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import dendropy
    from dendropy import Tree
    DENDROPY_AVAILABLE = True
except ImportError:
    dendropy = None  # type: ignore[assignment]
    DENDROPY_AVAILABLE = False


# Matches a Newick tip token: a (possibly quoted, possibly underscore-
# containing) label that appears immediately before a comma, closing
# parenthesis, colon (branch length), or semicolon. Internal node labels
# come right after a closing parenthesis, so we exclude those by requiring
# the character immediately before the label to be either '(' or ','.
_TIP_LABEL_RE = re.compile(
    r"(?<=[(,])"                          # preceded by ( or ,
    r"('(?:[^']|'')+'|[^,():;\s][^,():;]*?)"  # quoted OR unquoted label
    r"(?=[:,);])"                         # followed by : , ) ;
)


def _extract_tip_labels(newick: str) -> List[str]:
    """Pull tip labels out of a raw Newick string without a full parser.
    Used solely to detect / dedupe duplicates before handing the string to
    dendropy, which rejects duplicates under its default TaxonNamespace.
    """
    return [m.group(1) for m in _TIP_LABEL_RE.finditer(newick)]


def dedupe_newick_file(tree_path: str, logger: logging.Logger,
                       out_path: Optional[Path] = None) -> Path:
    """Return a path to a Newick file with unique leaf labels.

    Uses dendropy's Newick parser with ``suppress_internal_node_taxa=True``
    and ``suppress_leaf_node_taxa=True`` so labels are stored on
    ``node.label`` directly and duplicates don't collide in a shared
    TaxonNamespace. This is a far more robust parser than regex over the
    raw file — handles BEAST / NHX annotations, bootstrap values,
    internal node labels, and multi-tree files correctly.

    If duplicate leaf labels are detected, the 2nd, 3rd, ... occurrences
    are renamed to ``<label>__dupN`` and a warning is logged. The rewrite
    is written to ``out_path`` (or a tempfile) and the new path is
    returned. Input with no duplicates is left untouched and the original
    path is returned.
    """
    if not DENDROPY_AVAILABLE:
        raise RuntimeError("dendropy is required for tree loading")

    # suppress_*_node_taxa=True moves labels to .label instead of into a
    # shared TaxonNamespace where duplicates raise. Use TreeList.get so
    # a file with multiple trees doesn't silently lose trees past the
    # first (we still only analyse the first below, but with a warning).
    try:
        tree_list = dendropy.TreeList.get(  # type: ignore[attr-defined]
            path=str(tree_path),
            schema="newick",
            preserve_underscores=True,
            suppress_internal_node_taxa=True,
            suppress_leaf_node_taxa=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse tree file '{tree_path}' with dendropy: {e}"
        ) from e

    if len(tree_list) == 0:
        raise RuntimeError(f"No trees found in {tree_path}")
    if len(tree_list) > 1:
        logger.warning(
            f"Tree file '{tree_path}' contains {len(tree_list)} trees; "
            "analysing only the first. Downstream phylogenetic models "
            "take a single tree."
        )
    tree = tree_list[0]

    # Normalise leaf labels. Under ``suppress_leaf_node_taxa=True`` (which
    # we need so duplicates don't collide in a shared TaxonNamespace),
    # dendropy preserves the raw token exactly — including any surrounding
    # single quotes from the original Newick. Writing them back out would
    # then re-quote the already-quoted string, and ape would read a label
    # that still contains literal quote characters. Strip a single outer
    # pair here so the downstream intersect isn't sabotaged by stray
    # quote chars.
    leaves = [n for n in tree.leaf_node_iter()]
    n_unquoted = 0
    for leaf in leaves:
        lab = leaf.label
        if lab and len(lab) >= 2 and lab[0] == "'" and lab[-1] == "'":
            leaf.label = lab[1:-1]
            n_unquoted += 1
    if n_unquoted:
        logger.info(
            f"Stripped literal outer quotes from {n_unquoted} leaf labels "
            "(artefact of suppress_leaf_node_taxa parsing)."
        )

    # Collect leaf labels, detect duplicates, rename in-place.
    from collections import Counter as _Counter
    seen: _Counter = _Counter()
    for leaf in leaves:
        seen[leaf.label] += 1
    total_leaves = len(leaves)
    unique_labels = len(seen)
    dups = {lab: n for lab, n in seen.items() if n > 1 and lab is not None}
    logger.info(
        f"Tree parse: {total_leaves} leaves, {unique_labels} unique labels, "
        f"{len(dups)} duplicated."
    )

    multi_tree = len(tree_list) > 1
    if not dups and not multi_tree:
        return Path(tree_path)

    if dups:
        example = next(iter(dups))
        logger.warning(
            f"Tree has {len(dups)} duplicate leaf label(s); renaming "
            f"extras with '__dupN' suffix. Example: {example!r} "
            f"x{dups[example]}"
        )
        counter: _Counter = _Counter()
        for leaf in leaves:
            lab = leaf.label
            if lab is None:
                continue
            counter[lab] += 1
            if counter[lab] > 1:
                leaf.label = f"{lab}__dup{counter[lab] - 1}"

    out_path = out_path or Path(tempfile.mkstemp(
        prefix="defense_v2_clean_tree_", suffix=".nwk")[1])
    tree.write(
        path=str(out_path),
        schema="newick",
        unquoted_underscores=True,
        suppress_internal_node_labels=False,
    )
    logger.info(f"Wrote cleaned tree to {out_path}")
    return out_path




def normalize_species_name(name: str) -> str:
    """Strip quotes and collapse whitespace. Preserves case."""
    s = str(name).strip()
    while s and s[0] in "\"'":
        s = s[1:]
    while s and s[-1] in "\"'":
        s = s[:-1]
    return " ".join(s.split())


def species_name_to_underscore(name: str) -> str:
    return normalize_species_name(name).replace(" ", "_")


def match_species_to_tree(species_list: List[str], tip_labels: List[str],
                          logger: logging.Logger) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Match data species names to tree tips.

    Returns ``(matched_species, matched_tips, species_to_tip_map)``. The
    matching strategy that yields the most matches wins; we log which one.
    """
    logger.info(f"Sample species from data: {species_list[:3]}")
    logger.info(f"Sample tree tips: {tip_labels[:3]}")

    data_norm = {normalize_species_name(s): s for s in species_list}
    tree_norm = {normalize_species_name(t): t for t in tip_labels}
    data_us = {species_name_to_underscore(s): s for s in species_list}
    tree_us = {species_name_to_underscore(t): t for t in tip_labels}

    strategies = [
        ("direct",                              set(data_norm) & set(tree_norm),  data_norm, tree_norm),
        ("underscore",                          set(data_us)   & set(tree_us),    data_us,   tree_us),
        ("data_underscore_to_tree_normalized",  set(data_us)   & set(tree_norm),  data_us,   tree_norm),
        ("data_normalized_to_tree_underscore",  set(data_norm) & set(tree_us),    data_norm, tree_us),
    ]
    name, keys, d_map, t_map = max(strategies, key=lambda x: len(x[1]))
    logger.info(f"Best name-matching strategy: '{name}' -> {len(keys)} species matched")

    if not keys:
        return [], [], {}

    matched_species = [d_map[k] for k in keys]
    matched_tips = [t_map[k] for k in keys]
    return matched_species, matched_tips, dict(zip(matched_species, matched_tips))


def preprocess_newick_to_file(tree_path: str, kept_tips: List[str],
                              out_path: Path, logger: logging.Logger,
                              epsilon: float = 1e-4) -> Path:
    """Load a newick tree, prune to ``kept_tips``, repair zero-length and
    non-finite branches, resolve polytomies, and write to ``out_path``.

    The output path is what R scripts then read via ``ape::read.tree``. This
    keeps all the tree surgery on the Python side where it's inspectable,
    rather than duplicating it in every R script.
    """
    if not DENDROPY_AVAILABLE:
        raise RuntimeError("dendropy is required for tree preprocessing")

    safe_path = dedupe_newick_file(tree_path, logger)
    tree = Tree.get(path=str(safe_path), schema="newick", preserve_underscores=True)

    # Retain only the tips we want, renaming via the tip-label matching already
    # done by the caller.
    kept = set(kept_tips)
    tree.retain_taxa_with_labels(kept)

    # Resolve polytomies (dendropy: resolve_polytomies converts them into a
    # sequence of arbitrary bifurcations with zero-length branches, which we
    # then fix below).
    tree.resolve_polytomies(update_bipartitions=True)

    bad_branch_count = 0
    for edge in tree.postorder_edge_iter():
        if edge.length is None or not np.isfinite(edge.length) or edge.length <= 0:
            edge.length = epsilon
            bad_branch_count += 1

    logger.info(
        f"Tree prepared: {len(tree.leaf_nodes()):,} tips; "
        f"{bad_branch_count} branches replaced with epsilon={epsilon}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # dendropy's writer auto-quotes labels that contain whitespace, which
    # is what we need for ape::read.tree to preserve them verbatim.
    # Labels of pure alphanumeric + underscore form are written unquoted;
    # for those ape applies its underscore->space conversion, which the
    # R scripts cancel out by doing a matching gsub on the `tip` column.
    tree.write(path=str(out_path), schema="newick", unquoted_underscores=True)
    return out_path


def build_phylo_dataframe(binary_df: pd.DataFrame, defense_cols: List[str],
                          species_to_tip: Dict[str, str]) -> pd.DataFrame:
    """Subset binary_df to species that are on the tree, and rename the
    species column to match the tree tip labels. The resulting frame can be
    handed directly to R — its ``tip`` column lines up with tree tip labels.

    All non-defense columns from ``binary_df`` are carried through so that
    covariates, stratified plasmid outcomes, and taxonomy are available to
    downstream R scripts. Defense columns are preserved in their original
    order so that predictor lists remain stable.
    """
    sub = binary_df[binary_df["gtdb_species"].isin(species_to_tip)].copy()
    sub["tip"] = sub["gtdb_species"].map(species_to_tip)
    # Tip labels in the pruned Newick are force-quoted by
    # preprocess_newick_to_file, so ape reads them verbatim with whatever
    # spaces / underscores / case the original tree used. The `tip` column
    # in the data TSV must match the ORIGINAL label form (i.e. whatever is
    # in species_to_tip), not a normalised version — else intersect() will
    # miss. No substitution here on purpose.
    non_defense = [c for c in sub.columns if c not in defense_cols]
    # Put tip first, then the other metadata / outcome / covariate columns,
    # then the defense columns.
    ordered = ["tip"] + [c for c in non_defense if c != "tip"] + defense_cols
    # Drop duplicates while preserving order
    seen = set()
    final = [c for c in ordered if not (c in seen or seen.add(c))]
    return sub[final]
