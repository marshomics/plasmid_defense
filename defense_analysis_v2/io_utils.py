"""Data loading, plasmid-metadata stratification, genome covariates, and
species-level aggregation.

Strain-level binary defense tables are collapsed to species. Two species-level
tables are produced:

    - ``binary_df``  : max-across-strains (1 if any strain carries the system)
    - ``prevalence_df`` : proportion of strains carrying the system

plus ``n_strains`` (sequencing-depth weight), ``log_n_strains``, a
natural-cubic-spline basis on ``log_n_strains`` (``depth_ns1..k``) which is
what every downstream phylogenetic model actually uses to partial out
sampling-depth saturation, GTDB taxonomy, genome-scale covariates (mean genome
size, GC, CDS count), and a set of stratified plasmid outcomes derived from the
plasmid metadata table:

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

Why a spline and not ``log_n_strains``
--------------------------------------
Both the species-level plasmid label and the species-level defense call are
"at least one positive among n strains", i.e. ``1 - (1-p)^n``. On the logit
scale that is not linear in ``log n``, and the curvature depends on p, so a
rare defense system and common plasmid carriage sit on differently-shaped
curves. A single linear term therefore leaves substantial residual confounding
-- 28% false positives when defense is common and plasmid carriage rare, 40%
when sampling depth is clade-structured (which is what GTDB looks like). The
natural-spline basis spans the curve instead of assuming it away.
``log_n_strains`` is still emitted for diagnostics and stratification.
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
# Natural cubic spline basis (Harrell restricted cubic spline)
# ======================================================================


def restricted_cubic_spline_basis(x: np.ndarray, df: int
                                  ) -> Tuple[np.ndarray, List[float]]:
    """Return ``(basis, knots)`` for a restricted cubic spline on ``x``.

    ``df`` basis columns are produced from ``df + 1`` knots placed at evenly
    spaced quantiles of the observed distribution. Column 0 is the linear term,
    so ``df = 1`` degenerates to a plain linear fit and reproduces the legacy
    single-covariate behaviour.

    Restricted (natural) cubic splines are linear beyond the boundary knots,
    which is what we want here: the tail of the depth distribution is thin, and
    an unrestricted cubic would extrapolate wildly across the handful of
    species with thousands of genomes.

    Implementation follows Harrell, *Regression Modeling Strategies*, §2.4.4.
    NaNs propagate to NaN rows rather than raising, so downstream fits drop
    those species via the usual finite-row filter.
    """
    x = np.asarray(x, dtype=float)
    df = max(1, int(df))

    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.full((x.size, 1), np.nan), []

    n_knots = df + 1
    # Degenerate cases: not enough distinct values to place knots. Fall back to
    # the linear term rather than emitting collinear columns that would make
    # the R-side design matrix singular.
    n_unique = np.unique(finite).size
    if df == 1 or n_knots < 3 or n_unique < n_knots + 2:
        return x.reshape(-1, 1), []

    # Knot placement for a HEAVILY TIED variable.
    #
    # n_strains is an integer and most species have very few: on the real data
    # the 5/27.5/50/72.5/95% quantiles of log1p(n_strains) collapse to just
    # three distinct values, so a df=5 request silently produced a 2-column
    # basis. Quantile knots are the right default -- they put resolution where
    # the data is -- but when ties collapse them, top up from the distinct
    # observed values so the requested resolution is actually delivered across
    # the range that matters (the heavy tail, where the depth confound lives).
    qs = np.linspace(0.05, 0.95, n_knots)
    knots = list(np.unique(np.quantile(finite, qs)))
    if len(knots) < n_knots:
        uniq = np.unique(finite)
        if uniq.size >= n_knots:
            # Evenly spaced over the distinct values, which for a right-skewed
            # count reaches into the tail instead of piling up at the mode.
            idx = np.linspace(0, uniq.size - 1, n_knots).round().astype(int)
            topped = np.unique(np.concatenate([knots, uniq[idx]]))
            if topped.size > len(knots):
                knots = list(topped[:n_knots] if topped.size > n_knots
                             else topped)
    if len(knots) < 3:
        return x.reshape(-1, 1), []

    t = np.asarray(knots, dtype=float)
    k = t.size
    tk, tkm1, t1 = t[-1], t[-2], t[0]
    denom = (tk - t1) ** 2
    spread = tk - tkm1
    if denom <= 0 or spread <= 0:
        return x.reshape(-1, 1), []

    def cube_plus(v: np.ndarray) -> np.ndarray:
        return np.where(v > 0, v ** 3, 0.0)

    cols = [x]
    for j in range(k - 2):
        tj = t[j]
        term = (cube_plus(x - tj)
                - cube_plus(x - tkm1) * (tk - tj) / spread
                + cube_plus(x - tk) * (tkm1 - tj) / spread) / denom
        cols.append(term)

    basis = np.column_stack(cols)
    basis[~np.isfinite(x), :] = np.nan
    return basis, [float(v) for v in t]


def add_spline_basis(df: pd.DataFrame, source_col: str, prefix: str,
                     spline_df: int, logger: logging.Logger) -> pd.DataFrame:
    """Attach ``prefix1..prefixN`` spline-basis columns derived from
    ``source_col``. Returns the frame with the columns added in place.

    Existing basis columns with the same prefix are replaced, so the function
    is idempotent across the sensitivity reruns that rebuild the frame.
    """
    stale = [c for c in df.columns
             if c.startswith(prefix) and c[len(prefix):].isdigit()]
    if stale:
        df = df.drop(columns=stale)

    basis, knots = restricted_cubic_spline_basis(df[source_col].values,
                                                 spline_df)
    names = [f"{prefix}{i + 1}" for i in range(basis.shape[1])]
    for i, name in enumerate(names):
        df[name] = basis[:, i]

    if knots:
        logger.info(
            f"Spline basis '{prefix}' on {source_col}: {len(names)} columns, "
            f"knots at {[round(k, 3) for k in knots]}"
        )
    else:
        logger.info(
            f"Spline basis '{prefix}' on {source_col}: degenerate "
            f"(too few distinct values); using a single linear term"
        )
    return df


def add_depth_basis(df: pd.DataFrame, config: Config,
                    logger: logging.Logger) -> pd.DataFrame:
    """Attach the sampling-depth spline basis used by every phylogenetic fit.

    Must be called after ``log_n_strains`` exists and after any row filtering
    that changes the depth distribution -- the knots are quantiles of the rows
    actually being fit, so a depth-filtered rerun needs its own basis rather
    than the one computed on the full table.
    """
    if not config.use_n_strains_covariate:
        return df
    return add_spline_basis(df, "log_n_strains", config.depth_spline_prefix,
                            config.depth_spline_df, logger)


def add_plasmid_count_basis(df: pd.DataFrame, config: Config,
                            logger: logging.Logger) -> pd.DataFrame:
    """Attach the log(n_plasmids) spline basis for binary stratified
    outcomes. Requires ``log_n_plasmids`` to exist."""
    if not config.use_plasmid_count_covariate_on_binary:
        return df
    if "log_n_plasmids" not in df.columns:
        return df
    return add_spline_basis(df, "log_n_plasmids",
                            config.plasmid_count_spline_prefix,
                            config.plasmid_count_spline_df, logger)


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
    # log_n_strains seeds the spline basis that every phyloglm / PGLMM / PGLS
    # fit uses as a covariate, so the sampling-depth saturation shared by the
    # max()-aggregated defense call and the species-propagated plasmid label
    # is partialled out rather than left as a latent common cause. log1p so a
    # species with a single strain maps to log(2) rather than 0.
    #
    # The basis itself is attached at the end of this function, after the
    # frames are assembled -- see add_depth_basis.
    log_n_strains = np.log1p(n_strains.astype(float)).rename("log_n_strains")
    taxonomy = groups[["gtdb_domain", "gtdb_phylum", "gtdb_class",
                       "gtdb_order", "gtdb_family", "gtdb_genus"]].first()

    parts = [prevalence_data, plasmid, n_strains, log_n_strains, taxonomy]
    bparts = [binary_data, plasmid, n_strains, log_n_strains, taxonomy]

    # Genome covariates — mean across strains within a species.
    #
    # Heavy-tailed covariates (genome size, CDS count) are log-transformed
    # HERE, on the Python side. config.py previously claimed this happened "at
    # the R layer"; it did not -- the R scripts only centre and scale, so
    # heavy-tailed covariates entered the models untransformed. Transform
    # before averaging is wrong (Jensen), so we average then transform, which
    # matches the "mean genome size of the species" quantity we want.
    cov_cols: List[str] = []
    if config is not None and config.use_genome_covariates:
        cov_cols = [c for c in config.genome_covariate_columns if c in df.columns]
        if cov_cols:
            covariates = groups[cov_cols].mean()
            to_log = [c for c in cov_cols
                      if c in getattr(config, "log_transform_covariates", ())]
            for c in to_log:
                vals = covariates[c].astype(float)
                if (vals.dropna() <= 0).any():
                    logger.warning(
                        f"Covariate '{c}' has non-positive values; using "
                        f"log1p instead of log for the transform."
                    )
                    covariates[c] = np.log1p(vals.clip(lower=0))
                else:
                    covariates[c] = np.log(vals)
            if to_log:
                logger.info(f"Log-transformed genome covariates: {to_log}")
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

    # Sampling-depth spline basis. Attached last so the knots are quantiles of
    # the final species set. Every phylogenetic fit consumes these columns.
    if config is not None:
        prevalence_df = add_depth_basis(prevalence_df, config, logger)
        binary_df = add_depth_basis(binary_df, config, logger)

        # log(n_plasmids) + its spline basis, for binary stratified outcomes.
        for dfref in (prevalence_df, binary_df):
            if "n_plasmids" in dfref.columns:
                dfref["log_n_plasmids"] = np.log1p(
                    pd.to_numeric(dfref["n_plasmids"], errors="coerce"))
        prevalence_df = add_plasmid_count_basis(prevalence_df, config, logger)
        binary_df = add_plasmid_count_basis(binary_df, config, logger)

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
