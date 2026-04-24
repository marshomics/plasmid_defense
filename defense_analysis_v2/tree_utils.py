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
    from dendropy import Tree
    DENDROPY_AVAILABLE = True
except ImportError:
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
    """Return a path to a Newick file with unique tip labels.

    GTDB species labels can collide when the same SpeciesCluster is
    represented by multiple MAGs (e.g. 's__UBA12225 sp002347925' appearing
    on two tips). dendropy's default loader raises on duplicates. We rename
    the 2nd, 3rd, ... occurrences of each duplicate to
    '<label>__dup<N>' and write a sanitized Newick file.

    Downstream species-name matching (``match_species_to_tree``) only sees
    the original labels for the first occurrence, so the duplicates are
    effectively excluded from phylogenetic analyses. A warning is logged
    listing every collision so the user can decide whether to regenerate
    the tree.
    """
    src = Path(tree_path).read_text()
    labels = _extract_tip_labels(src)
    counts = Counter(labels)
    dups = {lab: n for lab, n in counts.items() if n > 1}
    if not dups:
        return Path(tree_path)

    logger.warning(
        f"Tree has {len(dups)} duplicate tip label(s); renaming extras "
        f"with '__dupN' suffix so dendropy can load the tree. Example: "
        f"{next(iter(dups))} x{dups[next(iter(dups))]}"
    )

    # Walk the string left-to-right, assigning suffixes the second time we
    # see a given label.
    seen: Counter = Counter()
    out_parts: List[str] = []
    pos = 0
    for m in _TIP_LABEL_RE.finditer(src):
        lab = m.group(1)
        start, end = m.span(1)
        out_parts.append(src[pos:start])
        seen[lab] += 1
        if seen[lab] == 1 or counts[lab] == 1:
            out_parts.append(lab)
        else:
            # Preserve quoting if the original was quoted
            if lab.startswith("'") and lab.endswith("'"):
                inner = lab[1:-1]
                out_parts.append(f"'{inner}__dup{seen[lab] - 1}'")
            else:
                out_parts.append(f"{lab}__dup{seen[lab] - 1}")
        pos = end
    out_parts.append(src[pos:])

    out_path = out_path or Path(tempfile.mkstemp(
        prefix="defense_v2_dedup_tree_", suffix=".nwk")[1])
    out_path.write_text("".join(out_parts))
    logger.info(f"Wrote deduplicated tree to {out_path}")
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
    non_defense = [c for c in sub.columns if c not in defense_cols]
    # Put tip first, then the other metadata / outcome / covariate columns,
    # then the defense columns.
    ordered = ["tip"] + [c for c in non_defense if c != "tip"] + defense_cols
    # Drop duplicates while preserving order
    seen = set()
    final = [c for c in ordered if not (c in seen or seen.add(c))]
    return sub[final]
