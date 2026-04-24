"""Data loading, plasmid-metadata stratification, genome covariates, and
species-level aggregation.

Strain-level binary defense tables are collapsed to species. Two species-level
tables are produced:

    - ``binary_df``  : max-across-strains (1 if any strain carries the system)
    - ``prevalence_df`` : proportion of strains carrying the system

plus ``n_strains`` (sequencing-depth weight), ``log_n_strains`` (the
log-transformed sampling-depth covariate that every downstream phylogenetic
model uses to partial out saturation of the max()-aggregated binary feature),
GTDB taxonomy, genome-scale covariates (mean genome size, GC, CDS count), and
a set of stratified plasmid outcomes derived from the plasmid metadata table:

    - has_plasmid_binary        : 1 if any plasmid in the species (legacy)
    - n_plasmids                : total plasmid count per species
    - plasmid_mean_size_log     : log-mean plasmid size (continuous)
    - any_plasmid_<class>       : 1 if the species has any plasmid of class X
    - frac_plasmid_<class>      : fraction of the species's plasmids in class X
    - n_plasmid_<class>         : raw count of plasmids of class X

Classes cover predicted mobility (conjugative / mobilizable / non-mobilizable),
size bins (small / medium / large), and the top-N replicon types by species
prevalence.

Plasmid carriage is propagated species-level in the upstream data pipeline. We
verify this invariant on the strain-level boolean and abort on violation. For
the *stratified* plasmid features we use the plasmid metadata table directly,
which already carries species-level assignments per plasmid row.
"""

import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config


METADATA_COLUMNS = (
    "genome", "has_plasmid", "source",
    "gtdb_domain", "gtdb_phylum", "gtdb_class",
    "gtdb_order", "gtdb_family", "gtdb_genus", "gtdb_species",
)

# Canonical mobility classes (mob_suite emits labels in these three buckets).
MOBILITY_CLASSES = ("conjugative", "mobilizable", "non-mobilizable")

# Size class names parallel the plasmid_size_bins_bp config entry.
SIZE_CLASSES = ("small", "medium", "large")


# ======================================================================
# Value normalisation
# ======================================================================

_MISSING_TOKENS = {"", "-", "na", "nan", "none", "unknown", "null"}


def _is_missing(val) -> bool:
    if val is None:
        return True
    try:
        if pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(val, str) and val.strip().lower() in _MISSING_TOKENS:
        return True
    return False


def _split_multi(val, sep_pattern: str = r"[;,]") -> List[str]:
    """Split a mob_suite-style multi-value cell into distinct labels.

    Returns an empty list for missing/blank/'-'. Whitespace trimmed; case-
    preserved (replicon codes are case-sensitive: IncF vs incF is real).
    """
    if _is_missing(val):
        return []
    parts = re.split(sep_pattern, str(val))
    return [p.strip() for p in parts if p.strip() and p.strip() not in _MISSING_TOKENS]


def _slugify(label: str) -> str:
    """Turn an arbitrary label into a safe column-name suffix."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(label)).strip("_")
    return s or "unknown"


# ======================================================================
# Strain-level input
# ======================================================================


def load_and_preprocess_data(config: Config, logger: logging.Logger,
                             input_file: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Load strain-level defense/plasmid table and return (df, defense_cols)."""
    path = input_file or config.input_file
    logger.info(f"Loading data from {path}")

    df = pd.read_csv(path, sep="\t", low_memory=False)
    logger.info(f"Loaded {len(df):,} strain genomes")

    defense_cols = [c for c in df.columns if c not in METADATA_COLUMNS]

    # Binarise counts. DefenseFinder emits non-negative integers; treat any
    # positive value as "present".
    for c in defense_cols:
        df[c] = (df[c] > 0).astype(int)

    df["has_plasmid_binary"] = (df["has_plasmid"].astype(str).str.lower() == "yes").astype(int)

    logger.info(f"Defense systems identified: {len(defense_cols)}")
    logger.info(
        f"Plasmid carriers (strain-level): {df['has_plasmid_binary'].sum():,} "
        f"({100 * df['has_plasmid_binary'].mean():.1f}%)"
    )
    return df, defense_cols


# ======================================================================
# Genome covariates
# ======================================================================


def load_genome_covariates(config: Config, strain_df: pd.DataFrame,
                           logger: logging.Logger) -> pd.DataFrame:
    """Load per-strain genome covariates and return the strain frame with the
    columns merged in. Missing rows are logged but not dropped (downstream
    aggregation handles NaN by taking the mean over present strains).
    """
    if not config.use_genome_covariates:
        return strain_df

    path = config.genome_covariates_file
    logger.info(f"Loading genome covariates from {path}")
    cov = pd.read_csv(path, sep="\t", low_memory=False)

    key = config.genome_covariates_key
    wanted = list(config.genome_covariate_columns)
    missing_cols = [c for c in [key] + wanted if c not in cov.columns]
    if missing_cols:
        raise ValueError(
            f"Genome covariates table is missing expected columns: {missing_cols}. "
            f"Found: {list(cov.columns)[:20]}"
        )

    cov = cov[[key] + wanted].drop_duplicates(subset=[key])
    for c in wanted:
        cov[c] = pd.to_numeric(cov[c], errors="coerce")

    before = len(strain_df)
    merged = strain_df.merge(cov, how="left", left_on="genome", right_on=key)
    n_missing = merged[wanted[0]].isna().sum()
    logger.info(
        f"Genome covariates merged: {before - n_missing:,}/{before:,} strains "
        f"have covariates ({100 * (before - n_missing) / before:.1f}%)"
    )
    if key != "genome" and key in merged.columns:
        merged = merged.drop(columns=[key])
    return merged


# ======================================================================
# Plasmid metadata + stratification
# ======================================================================


def load_plasmid_metadata(config: Config, logger: logging.Logger) -> pd.DataFrame:
    """Load the per-plasmid metadata table used for outcome stratification.

    Treats "-" and blanks as NaN. Returns a DataFrame with at minimum the
    gtdb_species join key and the mobility / replicon / size columns specified
    in config. Extra columns are retained for downstream use.
    """
    path = config.plasmid_metadata_file
    logger.info(f"Loading plasmid metadata from {path}")
    pm = pd.read_csv(path, sep="\t", low_memory=False, dtype=str)

    required = ["gtdb_species", config.plasmid_mobility_column,
                config.plasmid_reptype_column, config.plasmid_size_column]
    missing = [c for c in required if c not in pm.columns]
    if missing:
        raise ValueError(
            f"Plasmid metadata missing columns: {missing}. "
            f"Found (first 30): {list(pm.columns)[:30]}"
        )

    # Replace missing tokens with NaN for the columns we'll use.
    for c in [config.plasmid_mobility_column, config.plasmid_reptype_column]:
        pm[c] = pm[c].apply(lambda v: np.nan if _is_missing(v) else str(v).strip())
    pm[config.plasmid_size_column] = pd.to_numeric(
        pm[config.plasmid_size_column], errors="coerce")

    # gtdb_species missing rows are unusable (can't join to host species)
    n_before = len(pm)
    pm = pm[pm["gtdb_species"].apply(lambda v: not _is_missing(v))].copy()
    pm["gtdb_species"] = pm["gtdb_species"].str.strip()
    logger.info(
        f"Plasmid rows with usable gtdb_species: {len(pm):,}/{n_before:,}"
    )

    return pm


def _bin_size(size: Optional[float], bins_bp: Tuple[int, int]) -> Optional[str]:
    if size is None or not np.isfinite(size):
        return None
    if size < bins_bp[0]:
        return "small"
    if size < bins_bp[1]:
        return "medium"
    return "large"


def _canonical_mobility(val: Optional[str]) -> Optional[str]:
    if val is None or _is_missing(val):
        return None
    v = str(val).strip().lower().replace("_", "-").replace(" ", "-")
    if v in ("conjugative", "conjugable"):
        return "conjugative"
    if v in ("mobilizable", "mobilisable"):
        return "mobilizable"
    if v in ("non-mobilizable", "non-mobilisable", "nonmobilizable",
             "nonmobilisable", "non-conjugative", "non-conjugable"):
        return "non-mobilizable"
    # Anything else — keep the slug in case mob_suite adds new labels
    return _slugify(v).lower()


def build_species_plasmid_features(
    plasmid_md: pd.DataFrame,
    species_list: List[str],
    config: Config,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Compute per-species plasmid-stratification features.

    Returns ``(features_df, outcome_spec)`` where:
        - ``features_df`` is indexed by gtdb_species with columns:
            * ``n_plasmids``
            * ``plasmid_mean_size_log`` (natural log of mean bp; NaN if none)
            * For each mobility / size / top-rep-type class X:
                - ``n_plasmid_{X}``
                - ``frac_plasmid_{X}``  (NaN if n_plasmids == 0)
                - ``any_plasmid_{X}``    (1 if count > 0, else 0)
        - ``outcome_spec`` maps stratum name (e.g. "conjugative",
          "size_small", "reptype_IncF") to the triple
          ``[n_col, frac_col, any_col]`` so downstream code can iterate.

    The fraction outcome is the primary; the any-of-class binary is a
    backward-compatible secondary. Fractions are numerically 0 when a species
    has plasmids but none in class X; fractions are NaN when n_plasmids == 0,
    which removes the species from fraction-outcome models.
    """
    mobility_col = config.plasmid_mobility_column
    reptype_col = config.plasmid_reptype_column
    size_col = config.plasmid_size_column

    species_set = set(species_list)
    pm = plasmid_md[plasmid_md["gtdb_species"].isin(species_set)].copy()
    logger.info(
        f"Plasmid rows matched to analysis species: {len(pm):,} "
        f"(dropped {len(plasmid_md) - len(pm):,} rows with unmatched species)"
    )

    pm["_mobility"] = pm[mobility_col].map(_canonical_mobility)
    pm["_size_bin"] = pm[size_col].map(
        lambda s: _bin_size(s, config.plasmid_size_bins_bp))
    pm["_size_log"] = np.log(pm[size_col].where(pm[size_col] > 0))

    # -----------------------------------------------------------------
    # Which rep_type categories pass the prevalence gate?
    # We count each distinct label per plasmid row (split on ; or ,).
    # -----------------------------------------------------------------
    rep_per_row = pm[reptype_col].map(_split_multi)
    # Species -> set of rep types present across its plasmids (for prevalence count)
    sp_to_reptypes: Dict[str, set] = {}
    for sp, labels in zip(pm["gtdb_species"], rep_per_row):
        if not labels:
            continue
        sp_to_reptypes.setdefault(sp, set()).update(labels)
    rep_species_counts = Counter()
    for sp, labels in sp_to_reptypes.items():
        for lab in labels:
            rep_species_counts[lab] += 1
    eligible_reps = [lab for lab, n in rep_species_counts.items()
                     if n >= config.min_rep_type_species]
    eligible_reps = sorted(eligible_reps,
                           key=lambda lab: rep_species_counts[lab],
                           reverse=True)[:config.top_n_rep_types]
    logger.info(
        f"Replicon categories passing gate (>= {config.min_rep_type_species} "
        f"species): {eligible_reps}"
    )

    # -----------------------------------------------------------------
    # Build per-species rows
    # -----------------------------------------------------------------
    records: List[dict] = []
    for sp, grp in pm.groupby("gtdb_species"):
        n_pl = len(grp)
        rec = {
            "gtdb_species": sp,
            "n_plasmids": n_pl,
            "plasmid_mean_size_log": float(np.nanmean(grp["_size_log"]))
                if n_pl > 0 else np.nan,
        }
        # Mobility classes
        mob_counts = grp["_mobility"].value_counts(dropna=True).to_dict()
        for cls in MOBILITY_CLASSES:
            n = int(mob_counts.get(cls, 0))
            rec[f"n_plasmid_{cls}"] = n
            rec[f"frac_plasmid_{cls}"] = n / n_pl if n_pl else np.nan
            rec[f"any_plasmid_{cls}"] = int(n > 0)
        # Size bins
        size_counts = grp["_size_bin"].value_counts(dropna=True).to_dict()
        for cls in SIZE_CLASSES:
            n = int(size_counts.get(cls, 0))
            rec[f"n_plasmid_size_{cls}"] = n
            rec[f"frac_plasmid_size_{cls}"] = n / n_pl if n_pl else np.nan
            rec[f"any_plasmid_size_{cls}"] = int(n > 0)
        # Replicon categories (multi-label per plasmid)
        # Explode rep types for this species: list of (plasmid_row, label)
        reps_this_sp = grp[reptype_col].map(_split_multi)
        for cls in eligible_reps:
            slug = _slugify(cls)
            # Count plasmids in this species that carry this rep type at all
            n = int(sum(cls in labs for labs in reps_this_sp))
            rec[f"n_plasmid_reptype_{slug}"] = n
            rec[f"frac_plasmid_reptype_{slug}"] = n / n_pl if n_pl else np.nan
            rec[f"any_plasmid_reptype_{slug}"] = int(n > 0)
        records.append(rec)

    features_df = pd.DataFrame(records)

    # Build outcome_spec: stratum_name -> (n, frac, any) column triple
    outcome_spec: Dict[str, List[str]] = {}
    for cls in MOBILITY_CLASSES:
        outcome_spec[cls] = [f"n_plasmid_{cls}",
                             f"frac_plasmid_{cls}",
                             f"any_plasmid_{cls}"]
    for cls in SIZE_CLASSES:
        outcome_spec[f"size_{cls}"] = [f"n_plasmid_size_{cls}",
                                       f"frac_plasmid_size_{cls}",
                                       f"any_plasmid_size_{cls}"]
    for cls in eligible_reps:
        slug = _slugify(cls)
        outcome_spec[f"reptype_{slug}"] = [f"n_plasmid_reptype_{slug}",
                                           f"frac_plasmid_reptype_{slug}",
                                           f"any_plasmid_reptype_{slug}"]
    return features_df, outcome_spec


# ======================================================================
# Species-level aggregation
# ======================================================================


def aggregate_to_species_level(df: pd.DataFrame, defense_cols: List[str],
                               logger: logging.Logger,
                               config: Optional[Config] = None,
                               plasmid_md: Optional[pd.DataFrame] = None
                               ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """Collapse strain-level data to species level.

    Returns ``(prevalence_df, binary_df, outcome_spec)`` both indexed by
    ``gtdb_species`` and sharing columns: defense features,
    ``has_plasmid_binary``, ``n_strains``, GTDB taxonomy, genome covariates
    (if present), and the stratified plasmid outcomes (if plasmid_md provided).

    ``outcome_spec`` is a dict mapping stratum name to the triple of column
    names (n, frac, any) for downstream iteration. Always includes the key
    ``"any_plasmid"`` for the primary binary outcome that existed before.

    Raises ValueError if any species contains strains with conflicting
    ``has_plasmid`` annotations.
    """
    logger.info("Aggregating strains -> species")

    groups = df.groupby("gtdb_species")

    prevalence_data = groups[defense_cols].mean()
    binary_data = groups[defense_cols].max()

    plasmid_nunique = groups["has_plasmid_binary"].nunique()
    inconsistent = plasmid_nunique[plasmid_nunique > 1]
    if len(inconsistent) > 0:
        examples = inconsistent.index.tolist()[:10]
        raise ValueError(
            f"Input violates species-level plasmid invariant: {len(inconsistent)} "
            f"species have disagreeing has_plasmid labels across strains "
            f"(first 10: {examples})."
        )
    plasmid = groups["has_plasmid_binary"].first().rename("has_plasmid_binary")

    n_strains = groups.size().rename("n_strains")
    # log_n_strains is used by every phyloglm / PGLMM / PGLS fit as a
    # covariate, so the species-level sampling-depth saturation of the
    # max()-aggregated binary features is partialled out explicitly rather
    # than left as a latent confounder. log1p so a species with a single
    # strain (n_strains=1) maps to log1p(1) = log(2) rather than 0.
    log_n_strains = np.log1p(n_strains.astype(float)).rename("log_n_strains")
    taxonomy = groups[["gtdb_domain", "gtdb_phylum", "gtdb_class",
                       "gtdb_order", "gtdb_family", "gtdb_genus"]].first()

    parts = [prevalence_data, plasmid, n_strains, log_n_strains, taxonomy]
    bparts = [binary_data, plasmid, n_strains, log_n_strains, taxonomy]

    # Genome covariates — mean across strains within a species
    cov_cols: List[str] = []
    if config is not None and config.use_genome_covariates:
        cov_cols = [c for c in config.genome_covariate_columns if c in df.columns]
        if cov_cols:
            covariates = groups[cov_cols].mean()
            parts.append(covariates)
            bparts.append(covariates)

    prevalence_df = pd.concat(parts, axis=1).reset_index()
    binary_df = pd.concat(bparts, axis=1).reset_index()

    # Default outcome spec (binary any-plasmid outcome as the legacy fallback)
    outcome_spec: Dict[str, List[str]] = {
        "any_plasmid": ["n_plasmids", None, "has_plasmid_binary"],
    }

    # Stratified plasmid outcomes
    if plasmid_md is not None and config is not None:
        feats, strat_spec = build_species_plasmid_features(
            plasmid_md, prevalence_df["gtdb_species"].tolist(), config, logger)
        prevalence_df = prevalence_df.merge(feats, on="gtdb_species", how="left")
        binary_df = binary_df.merge(feats, on="gtdb_species", how="left")

        # Species not appearing in plasmid metadata: zero counts / NaN fractions.
        # We leave n_plasmids NaN for them to mark "no plasmid data" rather than
        # asserting zero — but log the discrepancy against has_plasmid_binary.
        n_missing_pm = int(prevalence_df["n_plasmids"].isna().sum())
        n_has_pl = int(prevalence_df["has_plasmid_binary"].sum())
        logger.info(
            f"Species with plasmid metadata: {len(prevalence_df) - n_missing_pm:,}; "
            f"species labelled has_plasmid=yes but missing from plasmid table: "
            f"{int(((prevalence_df['has_plasmid_binary'] == 1) & prevalence_df['n_plasmids'].isna()).sum()):,}"
        )
        # For species with no plasmid metadata but has_plasmid=0, set n=0 so
        # fraction outcomes treat them as structural zeros rather than missing.
        mask_legit_zero = (prevalence_df["has_plasmid_binary"] == 0) & \
                          prevalence_df["n_plasmids"].isna()
        for dfref in (prevalence_df, binary_df):
            dfref.loc[mask_legit_zero, "n_plasmids"] = 0
            for name, (nc, fc, ac) in strat_spec.items():
                if nc in dfref.columns:
                    dfref.loc[mask_legit_zero, nc] = 0
                if ac in dfref.columns:
                    dfref.loc[mask_legit_zero, ac] = 0
                # fraction stays NaN when n=0 (no denominator)

        outcome_spec.update(strat_spec)

    logger.info(f"Aggregated to {len(prevalence_df):,} species")
    logger.info(
        f"Plasmid-carrying species: {int(prevalence_df['has_plasmid_binary'].sum()):,} "
        f"({100 * prevalence_df['has_plasmid_binary'].mean():.1f}%)"
    )
    if cov_cols:
        n_with_all = int(prevalence_df[cov_cols].notna().all(axis=1).sum())
        logger.info(
            f"Species with complete genome covariates: {n_with_all:,}/"
            f"{len(prevalence_df):,}"
        )

    return prevalence_df, binary_df, outcome_spec


def add_defense_burden(prevalence_df: pd.DataFrame, binary_df: pd.DataFrame,
                       defense_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add ``defense_burden_count`` (sum of binary presences) and
    ``defense_burden_prevalence`` (sum of prevalences) to each species table.

    The count version is the one used for phylogenetically-corrected burden
    tests; the prevalence version is retained as a diagnostic.
    """
    prevalence_df = prevalence_df.copy()
    binary_df = binary_df.copy()

    prevalence_df["defense_burden_prevalence"] = prevalence_df[defense_cols].sum(axis=1)
    binary_df["defense_burden_count"] = binary_df[defense_cols].sum(axis=1)

    # Mirror the count into prevalence_df as well so downstream modules that
    # only receive one of the two tables always have access.
    prevalence_df["defense_burden_count"] = binary_df["defense_burden_count"].values

    return prevalence_df, binary_df
