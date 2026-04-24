"""Defense-system taxonomy helpers.

Groups DefenseFinder subtype/type names into mechanism categories
(Restriction-Modification, CRISPR-Cas, Abortive infection, Gabija, etc.).

The mapping is heuristic-by-name-prefix — consistent with the old script's
``classify_defense_system`` — and kept deliberately flat. Used for:
    - category rollups in descriptive figures (stacked bar by burden)
    - colouring points in takehome figures
    - the RM/Gabija subset selections in the key-findings plots
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

import pandas as pd


# Canonical categories, ordered so the first match wins.
_CATEGORY_RULES = [
    # (category label, list of lowercase substrings / regex patterns)
    ("Restriction-Modification", [r"^rm[_\-]?", r"^type_i+_", r"type_iii", r"^dnd",
                                   r"^drt", r"^mrr", r"^mcrbc", r"^bregab"]),
    ("CRISPR-Cas",                [r"^crispr", r"^cas\b", r"^cas[_\-]?[0-9]"]),
    ("Abortive-Infection",        [r"^abi[a-z]?", r"^rex", r"^bst", r"^prr"]),
    ("Toxin-Antitoxin",           [r"^ta\b", r"^mazef", r"^reli?be?", r"^hipba?",
                                   r"^ccdab", r"^vapbc"]),
    ("CBASS",                     [r"^cbass"]),
    ("Gabija",                    [r"^gabija"]),
    ("Thoeris",                   [r"^thoeris"]),
    ("Septu",                     [r"^septu"]),
    ("Druantia",                  [r"^druantia"]),
    ("Lamassu",                   [r"^lamassu"]),
    ("Wadjet",                    [r"^wadjet"]),
    ("Hachiman",                  [r"^hachiman"]),
    ("Shedu",                     [r"^shedu"]),
    ("Zorya",                     [r"^zorya"]),
    ("Pycsar",                    [r"^pycsar"]),
    ("RADAR",                     [r"^radar"]),
    ("Retron",                    [r"^retron"]),
    ("DISARM",                    [r"^disarm"]),
    ("BREX",                      [r"^brex"]),
    ("PDC-S",                     [r"^pdc[_\-]"]),
    ("Viperin",                   [r"^viperin"]),
    ("Dnd",                       [r"^dnd"]),
    ("Paris",                     [r"^paris"]),
    ("Kiwa",                      [r"^kiwa"]),
    ("Dodola",                    [r"^dodola"]),
]


def classify_defense_system(name: str) -> str:
    """Return the canonical category for a defense system name. Falls back to
    ``"Other"`` if no rule matches; callers can reshape this to include the
    raw name when they want fine-grained counts.
    """
    if not isinstance(name, str):
        return "Other"
    low = name.lower()
    for cat, patterns in _CATEGORY_RULES:
        for pat in patterns:
            if re.search(pat, low):
                return cat
    return "Other"


def classify_all(names: Iterable[str]) -> Dict[str, str]:
    """Vectorised over an iterable of defense-system names."""
    return {n: classify_defense_system(n) for n in names}


def category_counts_per_species(binary_df: pd.DataFrame, defense_cols: List[str]) -> pd.DataFrame:
    """Return one column per category containing the count of systems from
    that category present in each species. Useful for stacked-bar / ratio
    figures where individual systems are too granular.
    """
    mapping = classify_all(defense_cols)
    by_cat = {}
    for cat in set(mapping.values()):
        members = [c for c in defense_cols if mapping[c] == cat]
        if not members:
            continue
        by_cat[cat] = binary_df[members].sum(axis=1)
    result = pd.DataFrame(by_cat, index=binary_df.index)
    return result


def category_prevalence_summary(binary_df: pd.DataFrame, defense_cols: List[str],
                                plasmid_col: str = "has_plasmid_binary") -> pd.DataFrame:
    """Per-category summary: prevalence among plasmid+, plasmid-, and total.
    Used by several descriptive figures and the summary report.
    """
    counts = category_counts_per_species(binary_df, defense_cols)
    pos = binary_df[plasmid_col] == 1
    rows = []
    for cat in counts.columns:
        rows.append({
            "category": cat,
            "n_systems": sum(1 for c in defense_cols if classify_defense_system(c) == cat),
            "mean_count_plasmid_pos": float(counts.loc[pos, cat].mean()),
            "mean_count_plasmid_neg": float(counts.loc[~pos, cat].mean()),
            "mean_count_overall": float(counts[cat].mean()),
        })
    return pd.DataFrame(rows).sort_values("mean_count_overall", ascending=False)
