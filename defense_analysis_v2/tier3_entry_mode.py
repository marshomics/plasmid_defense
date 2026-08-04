"""A4 — pre-registered entry-mode (single-stranded DNA) prediction.

The mechanism
-------------
Conjugative plasmids enter a recipient cell as SINGLE-STRANDED DNA. Type II
restriction endonucleases -- and restriction-like systems generally -- cleave
DOUBLE-stranded DNA. Non-conjugative plasmids arriving by transformation enter
as dsDNA and are therefore exposed to restriction immediately, while the
transient ssDNA intermediate of conjugative transfer buys time for the
plasmid's own methylase or anti-restriction proteins to act. This is one of the
documented reasons conjugation resists restriction.

The prediction, fixed before looking
------------------------------------
    dsDNA-restricting systems (RM, Type IV restriction, BREX, DISARM, Wadjet,
    Dnd) exclude NON-CONJUGATIVE plasmids more strongly than conjugative ones.

    Abortive-infection and nucleotide-signalling systems (CBASS, Thoeris,
    Pycsar, retrons, Abi, Lamassu, viperin, RADAR) sense infection rather than
    cleave incoming DNA, so entry mode should NOT modulate their effect.

Systems whose mechanism does not license a directional prediction either way
are reported but excluded from the confirmatory contrast. The partition lives
in ``config.entry_mode_predicted_categories`` /
``entry_mode_not_predicted_categories`` and must be edited BEFORE any
entry-mode result is inspected -- the entire inferential value of this analysis
comes from having fixed it in advance. Reported as a single confirmatory test
with one degree of freedom, not as 435 exploratory ones.

Why the design is depth-robust
------------------------------
The primary model is a WITHIN-SPECIES composition contrast: of the plasmids a
species carries, what fraction is non-conjugative? Both the numerator and the
denominator come from the same species, so every species-level property --
genome size, GC, clade, and critically sequencing depth -- is differenced out
by construction. Deeper sequencing gives a *better-estimated* composition, not
a systematically different one. The depth covariates are retained anyway.

This makes A4 an orthogonal line of evidence to the phyloglm sweep, with
entirely different failure modes.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import Config
from .r_bridge import call_r_script, write_shared_frame
from .stats_utils import apply_fdr
from .taxonomy import classify_defense_system


# ======================================================================
# Loading the entry-mode table
# ======================================================================

_YES = {"yes", "y", "true", "t", "1", "conjugative"}
_NO = {"no", "n", "false", "f", "0", "non-conjugative", "nonconjugative"}


def _parse_conjugative(val) -> Optional[int]:
    """Map the yes/no conjugative flag to 1/0. Anything unrecognised is None
    and the plasmid is dropped rather than silently coerced -- a mis-parsed
    entry-mode label would invert the very contrast being tested."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip().lower()
    if s in _YES:
        return 1
    if s in _NO:
        return 0
    return None


def load_entry_mode_table(config: Config, logger: logging.Logger,
                          plasmid_md: Optional[pd.DataFrame] = None
                          ) -> pd.DataFrame:
    """Load the conjugative/non-conjugative annotation and attach a species.

    Returns a frame with ``plasmid_id``, ``gtdb_species`` and ``conjugative``
    (1/0). Species is taken from the entry-mode table when present, otherwise
    recovered by joining to the main plasmid metadata on plasmid id.
    """
    path = config.entry_mode_metadata_file
    logger.info(f"Loading entry-mode (conjugative) annotation from {path}")
    em = pd.read_csv(path, sep="\t", low_memory=False, dtype=str)

    id_col = config.entry_mode_plasmid_id_column
    conj_col = config.entry_mode_conjugative_column
    missing = [c for c in (id_col, conj_col) if c not in em.columns]
    if missing:
        raise ValueError(
            f"Entry-mode table is missing required columns {missing}. "
            f"Found (first 30): {list(em.columns)[:30]}")

    em = em[[c for c in em.columns
             if c in (id_col, conj_col, config.entry_mode_species_column)]].copy()
    em = em.rename(columns={id_col: "plasmid_id", conj_col: "_conj_raw"})
    em["plasmid_id"] = em["plasmid_id"].astype(str).str.strip()

    n_before = len(em)
    em["conjugative"] = em["_conj_raw"].map(_parse_conjugative)
    unparsed = em["conjugative"].isna().sum()
    if unparsed:
        bad = (em.loc[em["conjugative"].isna(), "_conj_raw"]
               .value_counts().head(5).to_dict())
        logger.warning(
            f"Entry-mode: {unparsed:,}/{n_before:,} plasmids have an "
            f"unrecognised '{conj_col}' value and are dropped. "
            f"Most common unparsed values: {bad}")
    em = em[em["conjugative"].notna()].copy()
    em["conjugative"] = em["conjugative"].astype(int)
    em = em.drop(columns=["_conj_raw"])
    em = em.drop_duplicates(subset=["plasmid_id"])

    # ---- attach species ----
    sp_col = config.entry_mode_species_column
    if sp_col in em.columns and em[sp_col].notna().any():
        em = em.rename(columns={sp_col: "gtdb_species"})
        logger.info("Entry-mode: species taken directly from the entry-mode table")
    else:
        if plasmid_md is None:
            raise ValueError(
                f"Entry-mode table has no '{sp_col}' column and no main plasmid "
                f"metadata was supplied to join against. Provide one or the other.")
        pid = config.plasmid_id_column
        if pid not in plasmid_md.columns:
            raise ValueError(
                f"Cannot recover species for the entry-mode table: the main "
                f"plasmid metadata has no '{pid}' column to join on. "
                f"Found (first 30): {list(plasmid_md.columns)[:30]}")
        key = plasmid_md[[pid, "gtdb_species"]].copy()
        key[pid] = key[pid].astype(str).str.strip()
        key = key.drop_duplicates(subset=[pid])
        before = len(em)
        em = em.merge(key, left_on="plasmid_id", right_on=pid, how="inner")
        if pid != "plasmid_id" and pid in em.columns:
            em = em.drop(columns=[pid])
        logger.info(
            f"Entry-mode: species recovered by joining to the main plasmid "
            f"metadata on '{pid}' — {len(em):,}/{before:,} plasmids matched")

    em["gtdb_species"] = em["gtdb_species"].astype(str).str.strip()
    em = em[em["gtdb_species"].str.len() > 0]
    logger.info(
        f"Entry-mode: {len(em):,} plasmids usable; "
        f"{int(em['conjugative'].sum()):,} conjugative "
        f"({100 * em['conjugative'].mean():.1f}%)")
    return em


def build_entry_mode_features(entry_mode: pd.DataFrame,
                              species_list: List[str],
                              config: Config,
                              logger: logging.Logger) -> pd.DataFrame:
    """Per-species conjugative / non-conjugative plasmid counts.

    Columns: ``n_plasmids_entrymode``, ``n_plasmid_conjugative``,
    ``n_plasmid_nonconjugative``, ``frac_plasmid_nonconjugative``,
    ``any_plasmid_conjugative``, ``any_plasmid_nonconjugative``.
    """
    em = entry_mode[entry_mode["gtdb_species"].isin(set(species_list))]
    logger.info(
        f"Entry-mode: {len(em):,} plasmids matched to analysis species "
        f"({len(entry_mode) - len(em):,} dropped as unmatched)")
    if em.empty:
        return pd.DataFrame(columns=["gtdb_species"])

    g = em.groupby("gtdb_species")["conjugative"]
    feats = pd.DataFrame({
        "n_plasmids_entrymode": g.size(),
        "n_plasmid_conjugative": g.sum(),
    })
    feats["n_plasmid_nonconjugative"] = (feats["n_plasmids_entrymode"]
                                         - feats["n_plasmid_conjugative"])
    feats["frac_plasmid_nonconjugative"] = (
        feats["n_plasmid_nonconjugative"] / feats["n_plasmids_entrymode"])
    feats["any_plasmid_conjugative"] = (feats["n_plasmid_conjugative"] > 0).astype(int)
    feats["any_plasmid_nonconjugative"] = (
        feats["n_plasmid_nonconjugative"] > 0).astype(int)
    feats = feats.reset_index()

    logger.info(
        f"Entry-mode features: {len(feats):,} species; median plasmids/species "
        f"{feats['n_plasmids_entrymode'].median():.0f}; "
        f"{int((feats['n_plasmids_entrymode'] >= config.entry_mode_min_plasmids_per_species).sum()):,} "
        f"species have >= {config.entry_mode_min_plasmids_per_species} plasmids "
        f"and can contribute to the composition model")
    return feats


# ======================================================================
# Mechanism partition (pre-registered)
# ======================================================================

def assign_mechanism_groups(defense_cols: List[str],
                            config: Config) -> pd.DataFrame:
    """Label every defense system predicted / not_predicted / unclassified."""
    pred = set(config.entry_mode_predicted_categories)
    notp = set(config.entry_mode_not_predicted_categories)
    rows = []
    for c in defense_cols:
        cat = classify_defense_system(c)
        if cat in pred:
            grp = "predicted_dsDNA_restricting"
        elif cat in notp:
            grp = "not_predicted"
        else:
            grp = "unclassified"
        rows.append({"defense_system": c, "defense_category": cat,
                     "mechanism_group": grp})
    return pd.DataFrame(rows)


# ======================================================================
# Per-system composition model
# ======================================================================

def _fit_one_system_pglmm(phylo_data: pd.DataFrame, system: str,
                          covariates: List[str], tree_path: str,
                          config: Config, logger: logging.Logger,
                          workdir: Path) -> dict:
    """Univariate binomial PGLMM: cbind(n_nonconj, n_conj) ~ system + covs."""
    r = call_r_script(
        "pglmm_mv.R",
        tree_path=tree_path,
        data=phylo_data,
        args={
            "predictors": [system],
            "covariates": covariates,
            "tip_column": "tip",
            "outcome_mode": "binomial",
            "response_k_column": "n_plasmid_nonconjugative",
            "response_n_column": "n_plasmids_entrymode",
            "interaction_pairs": [],
            "bayes": False,
            "reml": True,
        },
        logger=logger,
        r_executable=config.r_executable,
        workdir=workdir / f"entry_mode_{_safe(system)}",
        timeout=max(60, int(config.pglmm_timeout_hours) * 3600),
    )
    if not r.ok or r.dataframe is None or r.dataframe.empty:
        return {"defense_system": system, "skip_reason": "pglmm_call_failed"}
    df = r.dataframe.rename(columns={"term": "defense_system"})
    row = df[df["defense_system"].astype(str).str.strip("`") == system]
    if row.empty:
        return {"defense_system": system, "skip_reason": "coefficient_row_absent"}
    row = row.iloc[0]
    degenerate = bool(row.get("pglmm_fit_degenerate", False))
    return {
        "defense_system": system,
        "entry_mode_coefficient": np.nan if degenerate else row.get("pglmm_coefficient"),
        "entry_mode_std_err": np.nan if degenerate else row.get("pglmm_std_err"),
        "entry_mode_p_value": np.nan if degenerate else row.get("pglmm_p_value"),
        "n_species_fit": row.get("n_species_fit"),
        "engine": "pglmm_binomial",
        "skip_reason": "degenerate_fit" if degenerate else np.nan,
    }


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(s))[:80]


def _fit_all_systems_batched(phylo_data: pd.DataFrame, systems: List[str],
                             covariates: List[str], tree_path: str,
                             config: Config, logger: logging.Logger,
                             workdir: Path) -> pd.DataFrame:
    """Fit every system, CHUNKED across parallel R processes.

    Two costs pull in opposite directions. One R invocation per system pays
    interpreter start-up, package loading and tree parsing 435 times. A single
    invocation for all 435 amortises that away but is one process, so it
    cannot use more than one core and the stage becomes a ~124 h serial block.

    Chunking gets both: ``n_jobs`` processes, each amortising start-up across
    its own slice of the systems. The frame is written once and shared, so the
    extra processes cost nothing in I/O.
    """
    n_jobs = config.n_jobs if config.n_jobs > 0 else mp.cpu_count()
    if config.max_concurrent_r_calls > 0:
        n_jobs = min(n_jobs, config.max_concurrent_r_calls)
    n_chunks = max(1, min(n_jobs, len(systems)))
    chunks = [list(c) for c in np.array_split(np.array(systems, dtype=object),
                                              n_chunks) if len(c)]
    shared = write_shared_frame(phylo_data, workdir, "entry_mode", logger)
    logger.info(
        f"Entry-mode: {len(systems)} systems across {len(chunks)} R processes "
        f"({len(chunks[0])} per process), one shared data frame")

    frames = Parallel(n_jobs=len(chunks), backend="threading", verbose=0)(
        delayed(_fit_chunk)(shared, chunk, covariates, tree_path, config,
                            logger, workdir, i)
        for i, chunk in enumerate(chunks))
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fit_chunk(shared, systems: List[str], covariates: List[str],
               tree_path: str, config: Config, logger: logging.Logger,
               workdir: Path, chunk_id: int) -> pd.DataFrame:
    r = call_r_script(
        "pglmm_uni_binomial.R",
        tree_path=tree_path,
        shared=shared,
        args={
            "predictors": list(systems),
            "covariates": list(covariates),
            "tip_column": "tip",
            "response_k_column": "n_plasmid_nonconjugative",
            "response_n_column": "n_plasmids_entrymode",
            "min_count": config.min_count_per_category,
            "reml": True,
            "per_fit_seconds": 900,
        },
        logger=logger,
        r_executable=config.r_executable,
        workdir=workdir / f"entry_mode_chunk{chunk_id:02d}",
        timeout=max(3600, int(config.pglmm_timeout_hours) * 3600),
        max_retries=int(config.r_max_retries),
    )
    if not r.ok or r.dataframe is None or r.dataframe.empty:
        logger.warning(f"entry-mode chunk {chunk_id} failed: {r.error}")
        return pd.DataFrame()

    df = r.dataframe.rename(columns={
        "pglmm_coefficient": "entry_mode_coefficient",
        "pglmm_std_err": "entry_mode_std_err",
        "pglmm_p_value": "entry_mode_p_value",
    })
    df["engine"] = "pglmm_binomial"
    keep = ["defense_system", "entry_mode_coefficient", "entry_mode_std_err",
            "entry_mode_p_value", "n_species_fit", "engine", "skip_reason"]
    return df[[c for c in keep if c in df.columns]]


def _fit_one_system_pgls(phylo_data: pd.DataFrame, system: str,
                         covariates: List[str], tree_path: str,
                         config: Config, logger: logging.Logger,
                         workdir: Path) -> dict:
    """Empirical-logit PGLS fallback.

    y = log((k + 0.5) / (n - k + 0.5)) is the standard empirical-logit
    transform; the 0.5 continuity correction keeps species with all-or-nothing
    composition finite. Much cheaper than a binomial GLMM and adequate when the
    per-species plasmid counts are not tiny, but it is an approximation to the
    binomial likelihood, which is why ``entry_mode_engine`` defaults to pglmm.
    """
    d = phylo_data.copy()
    k = d["n_plasmid_nonconjugative"].astype(float)
    n = d["n_plasmids_entrymode"].astype(float)
    d["_emp_logit"] = np.log((k + 0.5) / (n - k + 0.5))
    pass_cols = ["tip", "_emp_logit", system] + [c for c in covariates
                                                 if c in d.columns]
    r = call_r_script(
        "pgls_burden.R",
        tree_path=tree_path,
        data=d[pass_cols],
        args={"response": "_emp_logit", "predictor": system,
              "covariates": [c for c in covariates if c in d.columns],
              "tip_column": "tip", "transform": "none"},
        logger=logger,
        r_executable=config.r_executable,
        workdir=workdir / f"entry_mode_pgls_{_safe(system)}",
    )
    if not r.ok or r.dataframe is None or r.dataframe.empty:
        return {"defense_system": system, "skip_reason": "pgls_call_failed"}
    row = r.dataframe.iloc[0]
    if pd.notna(row.get("error")):
        return {"defense_system": system,
                "skip_reason": f"pgls_error:{row.get('error')}"}
    return {
        "defense_system": system,
        "entry_mode_coefficient": row.get("pgls_coefficient"),
        "entry_mode_std_err": row.get("pgls_std_err"),
        "entry_mode_p_value": row.get("pgls_p_value"),
        "n_species_fit": row.get("n_species"),
        "engine": "pgls_empirical_logit",
        "skip_reason": np.nan,
    }


def run_entry_mode_composition(phylo_data: pd.DataFrame,
                               defense_cols: List[str],
                               tree_path: str,
                               config: Config,
                               logger: logging.Logger,
                               workdir: Path) -> pd.DataFrame:
    """Per-system within-species composition model.

    Outcome: fraction of a species' plasmids that are NON-conjugative.
    A NEGATIVE coefficient means the defense system is associated with a
    plasmid pool depleted of non-conjugative (dsDNA-entering) plasmids, which
    is the direction the ssDNA-evasion mechanism predicts for dsDNA-restricting
    systems.
    """
    need = ["n_plasmid_nonconjugative", "n_plasmids_entrymode"]
    missing = [c for c in need if c not in phylo_data.columns]
    if missing:
        logger.warning(f"Entry-mode composition skipped — missing {missing}")
        return pd.DataFrame()

    d = phylo_data[
        phylo_data["n_plasmids_entrymode"].fillna(0)
        >= config.entry_mode_min_plasmids_per_species].copy()
    if len(d) < 50:
        logger.warning(
            f"Entry-mode composition skipped — only {len(d)} species with "
            f">= {config.entry_mode_min_plasmids_per_species} plasmids")
        return pd.DataFrame()

    # Rebuild the depth basis on this subset: the knots are quantiles of the
    # rows actually being fit.
    from .io_utils import add_depth_basis
    d = add_depth_basis(d, config, logger)

    covariates = list(config.resolve_covariates(
        config.covariate_columns_for_mode(config.primary_covariate_mode,
                                          include_plasmid_count=False), d))

    frac = float((d["n_plasmid_nonconjugative"] / d["n_plasmids_entrymode"]).mean())
    logger.info(
        f"Entry-mode composition: {len(d):,} species, mean non-conjugative "
        f"fraction {frac:.1%}, engine={config.entry_mode_engine}")

    testable = [c for c in defense_cols
                if c in d.columns
                and d[c].nunique(dropna=True) > 1
                and min(int((d[c] == 1).sum()), int((d[c] == 0).sum()))
                >= config.min_count_per_category]
    logger.info(f"Entry-mode: {len(testable)}/{len(defense_cols)} systems "
                f"have enough variance among composition-eligible species")
    if not testable:
        return pd.DataFrame()

    out = pd.DataFrame()
    if config.entry_mode_engine == "pglmm" and config.entry_mode_batch_in_r:
        # ONE R process iterating every system internally, instead of one R
        # invocation per system. Interpreter start-up, package loading, tree
        # parsing and data parsing are fixed costs that dominate here, because
        # the model itself runs only on species with enough plasmids to have a
        # composition -- a few thousand tips, not 40,000. Amortising them turns
        # 435 invocations into 1.
        out = _fit_all_systems_batched(d, testable, covariates, tree_path,
                                       config, logger, workdir)

    if out.empty and config.entry_mode_engine == "pglmm" \
            and config.entry_mode_auto_fallback_to_pgls:
        # Automatic fallback rather than losing the stage. Recorded in the
        # `engine` column so the substitution is never invisible.
        logger.warning(
            "Entry-mode: binomial PGLMM produced nothing; falling back to the "
            "empirical-logit PGLS. Results are tagged engine='pgls_empirical_logit'.")
        n_jobs = config.n_jobs if config.n_jobs > 0 else mp.cpu_count()
        n_jobs = max(1, min(n_jobs, len(testable)))
        results = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
            delayed(_fit_one_system_pgls)(d, s, covariates, tree_path, config,
                                          logger, workdir)
            for s in testable)
        out = pd.DataFrame([r for r in results if r])
    elif out.empty:
        n_jobs = config.n_jobs if config.n_jobs > 0 else mp.cpu_count()
        n_jobs = max(1, min(n_jobs, len(testable)))
        results = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
            delayed(_fit_one_system_pgls)(d, s, covariates, tree_path, config,
                                          logger, workdir)
            for s in testable)
        out = pd.DataFrame([r for r in results if r])

    if out.empty:
        return out

    out["entry_mode_fdr_qvalue"] = apply_fdr(
        out["entry_mode_p_value"], method=config.fdr_method).values
    out = out.merge(assign_mechanism_groups(list(out["defense_system"]), config),
                    on="defense_system", how="left")
    out["outcome"] = "frac_nonconjugative"
    out["covariate_mode"] = config.primary_covariate_mode

    n_fit = int(out["entry_mode_p_value"].notna().sum())
    logger.info(f"Entry-mode composition: {n_fit}/{len(out)} systems fit; "
                f"{int((out['entry_mode_fdr_qvalue'] < config.alpha).sum())} "
                f"at FDR q < {config.alpha}")
    return out


# ======================================================================
# The confirmatory contrast
# ======================================================================

def run_entry_mode_confirmatory(composition: pd.DataFrame,
                                config: Config,
                                logger: logging.Logger) -> pd.DataFrame:
    """The single pre-registered test.

    H0: the entry-mode composition effect does not differ between
        dsDNA-restricting systems and abortive/signalling systems.
    H1 (directional, pre-registered): dsDNA-restricting systems have a MORE
        NEGATIVE effect on the non-conjugative fraction, i.e. they deplete the
        plasmid pool of dsDNA-entering plasmids relative to systems that
        cannot discriminate by entry mode.

    Statistic: inverse-variance-weighted mean difference in coefficients
    between the two pre-declared groups.

    Null: permute the GROUP LABELS across systems. This is the right null
    because the per-system estimates are dependent -- through shared phylogeny
    and through co-occurrence in defense islands -- and permuting labels leaves
    that dependence structure untouched while destroying only the association
    between mechanism class and effect. A two-sample t-test would assume
    independence the estimates do not have.

    One-sided, because the prediction has a direction. The two-sided p-value is
    reported alongside for readers who want it.
    """
    if composition is None or composition.empty:
        return pd.DataFrame()

    d = composition.dropna(subset=["entry_mode_coefficient",
                                   "entry_mode_std_err"]).copy()
    d = d[d["entry_mode_std_err"] > 0]
    d = d[d["mechanism_group"].isin(["predicted_dsDNA_restricting",
                                     "not_predicted"])]
    n_pred = int((d["mechanism_group"] == "predicted_dsDNA_restricting").sum())
    n_not = int((d["mechanism_group"] == "not_predicted").sum())
    if n_pred < 3 or n_not < 3:
        logger.warning(
            f"Entry-mode confirmatory test skipped — need >= 3 systems per "
            f"group, have predicted={n_pred}, not_predicted={n_not}")
        return pd.DataFrame()

    beta = d["entry_mode_coefficient"].values.astype(float)
    w = 1.0 / (d["entry_mode_std_err"].values.astype(float) ** 2)
    is_pred = (d["mechanism_group"] == "predicted_dsDNA_restricting").values

    def _stat(mask) -> float:
        a = np.sum(w[mask] * beta[mask]) / np.sum(w[mask])
        b = np.sum(w[~mask] * beta[~mask]) / np.sum(w[~mask])
        return float(a - b)

    observed = _stat(is_pred)

    rng = np.random.default_rng(config.random_seed)
    n_perm = int(config.entry_mode_n_permutations)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = _stat(rng.permutation(is_pred))

    # One-sided in the pre-registered direction (predicted group MORE negative).
    p_one = float(((null <= observed).sum() + 1) / (n_perm + 1))
    p_two = float(((np.abs(null) >= abs(observed)).sum() + 1) / (n_perm + 1))

    mean_pred = float(np.sum(w[is_pred] * beta[is_pred]) / np.sum(w[is_pred]))
    mean_not = float(np.sum(w[~is_pred] * beta[~is_pred]) / np.sum(w[~is_pred]))

    supported = bool(p_one < config.alpha and observed < 0)
    out = pd.DataFrame([{
        "test": "entry_mode_prediction",
        "hypothesis": ("dsDNA-restricting systems deplete non-conjugative "
                       "(dsDNA-entering) plasmids more than abortive/"
                       "signalling systems"),
        "n_predicted_systems": n_pred,
        "n_not_predicted_systems": n_not,
        "weighted_mean_coef_predicted": mean_pred,
        "weighted_mean_coef_not_predicted": mean_not,
        "observed_difference": observed,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "p_one_sided_preregistered": p_one,
        "p_two_sided": p_two,
        "n_permutations": n_perm,
        "prediction_supported": supported,
    }])

    logger.info(
        f"ENTRY-MODE CONFIRMATORY TEST: weighted mean coefficient "
        f"predicted={mean_pred:+.3f} (n={n_pred}) vs "
        f"not-predicted={mean_not:+.3f} (n={n_not}); "
        f"difference={observed:+.3f}, one-sided p={p_one:.4g}")
    if supported:
        logger.info(
            "  PREDICTION SUPPORTED — the entry-mode contrast behaves as the "
            "ssDNA-evasion mechanism predicts.")
    else:
        logger.info(
            "  Prediction not supported at alpha. This is an informative "
            "negative: it argues against entry mode being the dominant axis "
            "on which these systems discriminate.")
    return out


def run_entry_mode_binary_secondary(phylo_data: pd.DataFrame,
                                    defense_cols: List[str],
                                    tree_path: str,
                                    config: Config,
                                    logger: logging.Logger,
                                    workdir: Path) -> pd.DataFrame:
    """Secondary: two separate binary phyloglm fits, one per entry mode.

    Reported for interpretability -- the composition model gives a contrast but
    not the two constituent effects. NOT used for the confirmatory test: the
    two fits share species, so their difference has an unknown covariance and a
    naive contrast would be miscalibrated. The composition model handles that
    by construction.
    """
    pieces = []
    for label, col in (("conjugative", "any_plasmid_conjugative"),
                       ("nonconjugative", "any_plasmid_nonconjugative")):
        if col not in phylo_data.columns:
            continue
        covariates = list(config.resolve_covariates(
            config.covariate_columns_for_mode(
                config.primary_covariate_mode, include_plasmid_count=True),
            phylo_data))
        r = call_r_script(
            "phyloglm_uni.R",
            tree_path=tree_path,
            data=phylo_data,
            args={"response": col, "predictors": defense_cols,
                  "mode": "predictor", "defense_side": "predictor",
                  "covariates": covariates, "tip_column": "tip",
                  "evolutionary_model": config.phyloglm_estimator,
                  "btol": 20, "boot": 0,
                  "min_count": config.min_count_per_category,
                  "min_count_response": config.min_count_per_category},
            logger=logger, r_executable=config.r_executable,
            workdir=workdir / f"entry_mode_binary_{label}",
        )
        if not r.ok:
            logger.warning(f"entry-mode binary [{label}] failed: {r.error}")
            continue
        df = r.dataframe.rename(columns={"test_label": "defense_system"})
        df["entry_mode"] = label
        df["phyloglm_fdr_qvalue"] = apply_fdr(
            df["phyloglm_p_value"], method=config.fdr_method).values
        pieces.append(df)
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True)
    return out.merge(assign_mechanism_groups(defense_cols, config),
                     on="defense_system", how="left")
