"""Publication-ready plotting for the pipeline outputs.

All figures are saved as PNG (300 dpi) and SVG (Illustrator-compatible) pairs
under ``<output_dir>/figures/``. Colour palette is Okabe & Ito (colourblind-
friendly). Fonts default to Arial at 9pt; override via matplotlib rcParams.

Every function takes the relevant results DataFrame(s) plus a ``figure_dir``
path. No function mutates its inputs. Failures in a single figure are logged
but don't cascade.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .taxonomy import classify_defense_system


# Okabe & Ito palette
PALETTE = {
    "blue":    "#0072B2",
    "orange":  "#E69F00",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "skyblue": "#56B4E9",
    "vermil":  "#D55E00",
    "pink":    "#CC79A7",
    "grey":    "#999999",
    "black":   "#000000",
}


def set_publication_rcparams():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,   # TrueType so fonts are editable in Illustrator
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _save(fig: plt.Figure, figure_dir: Path, name: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def _top_n(df: pd.DataFrame, by: str, n: int = 30, ascending: bool = True) -> pd.DataFrame:
    return df.sort_values(by, ascending=ascending).head(n)


# ======================================================================
# Tier 1 / Tier 2 plots
# ======================================================================

def plot_phyloglm_forest(phyloglm: pd.DataFrame, figure_dir: Path,
                         n: int = 30, alpha: float = 0.05) -> None:
    """Forest plot of phyloglm odds ratios for the top-n systems by q-value."""
    if phyloglm.empty:
        return
    set_publication_rcparams()
    d = _top_n(phyloglm, "phyloglm_fdr_qvalue", n=n).dropna(subset=["phyloglm_odds_ratio"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, max(3.0, 0.22 * len(d))))
    y = np.arange(len(d))
    colors = [PALETTE["vermil"] if q < alpha else PALETTE["grey"]
              for q in d["phyloglm_fdr_qvalue"]]
    ax.errorbar(np.log2(d["phyloglm_odds_ratio"]), y,
                xerr=[np.log2(d["phyloglm_odds_ratio"]) - np.log2(d["phyloglm_ci_low"]),
                      np.log2(d["phyloglm_ci_high"]) - np.log2(d["phyloglm_odds_ratio"])],
                fmt="o", ecolor=PALETTE["grey"], elinewidth=0.7,
                mfc="none", mec="none")
    ax.scatter(np.log2(d["phyloglm_odds_ratio"]), y, c=colors, s=18, zorder=3)
    ax.axvline(0, color=PALETTE["black"], lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(d["defense_system"], fontsize=7)
    ax.set_xlabel("log$_2$(odds ratio) — phyloglm")
    ax.set_title(f"Top {len(d)} defense systems by phyloglm FDR q-value")
    ax.invert_yaxis()
    _save(fig, figure_dir, "phyloglm_forest")


def plot_consensus_heatmap(consensus: pd.DataFrame, figure_dir: Path,
                           n: int = 30) -> None:
    """Rank heatmap across phyloglm, PGLMM, Pagel's for top-n systems."""
    if consensus is None or consensus.empty:
        return
    set_publication_rcparams()
    d = _top_n(consensus, "rank_product", n=n)
    rank_cols = ["phyloglm_p_value_rank", "pglmm_p_value_rank", "pagel_p_value_rank"]
    available = [c for c in rank_cols if c in d.columns]
    if not available:
        return
    mat = d[available].values
    fig, ax = plt.subplots(figsize=(3.5, max(3.0, 0.22 * len(d))))
    im = ax.imshow(mat, cmap="viridis_r", aspect="auto")
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d["defense_system"], fontsize=7)
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([c.replace("_p_value_rank", "") for c in available], rotation=45, ha="right")
    cb = plt.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("rank (lower = stronger)", fontsize=7)
    ax.set_title(f"Consensus ranks, top {len(d)}")
    _save(fig, figure_dir, "consensus_rank_heatmap")


def plot_misclassification_trajectories(mc_summary: pd.DataFrame, figure_dir: Path,
                                        system_highlights: Optional[list] = None) -> None:
    """One line per defense system: fraction of MC replicates significant as
    a function of assumed FNR. Highlighted systems drawn in colour; others grey.
    """
    if mc_summary is None or mc_summary.empty:
        return
    set_publication_rcparams()
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    highlight = set(system_highlights or [])
    for system, g in mc_summary.groupby("defense_system"):
        g = g.sort_values("fnr")
        c = PALETTE["vermil"] if system in highlight else PALETTE["grey"]
        lw = 1.4 if system in highlight else 0.4
        alpha = 1.0 if system in highlight else 0.35
        ax.plot(g["fnr"], g["frac_fdr_sig"], color=c, lw=lw, alpha=alpha)
        if system in highlight:
            ax.text(g["fnr"].iloc[-1] + 0.005, g["frac_fdr_sig"].iloc[-1], system,
                    fontsize=7, color=c, va="center")
    ax.axhline(0.5, color=PALETTE["black"], lw=0.4, ls=":")
    ax.set_xlabel("Assumed plasmid-detection FNR")
    ax.set_ylabel("Fraction of MC replicates still significant (FDR)")
    ax.set_title("Misclassification sensitivity (Monte Carlo)")
    ax.set_ylim(-0.02, 1.02)
    _save(fig, figure_dir, "misclass_mc_trajectories")


def plot_analytical_bias_correction(mc_analytical: pd.DataFrame, figure_dir: Path,
                                    n: int = 20) -> None:
    """One line per system: adjusted OR vs FNR (analytical correction)."""
    if mc_analytical is None or mc_analytical.empty:
        return
    set_publication_rcparams()
    systems = mc_analytical.groupby("defense_system")["obs_OR"] \
        .first().sort_values(ascending=False).head(n).index.tolist()
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for i, system in enumerate(systems):
        g = mc_analytical[mc_analytical["defense_system"] == system].sort_values("fnr")
        ax.plot(g["fnr"], g["adj_OR"], color=plt.cm.viridis(i / max(1, len(systems) - 1)),
                lw=1.2, label=system)
    ax.axhline(1.0, color=PALETTE["black"], lw=0.6, ls="--")
    ax.set_xlabel("Assumed FNR")
    ax.set_ylabel("Adjusted OR (analytical, Bross 1954)")
    ax.set_yscale("log")
    ax.set_title(f"Analytical bias correction, top {len(systems)} systems")
    ax.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    _save(fig, figure_dir, "misclass_analytical_curves")


def plot_loco_heterogeneity(loco_summary: pd.DataFrame, figure_dir: Path,
                            rank: str = "gtdb_class", n: int = 30) -> None:
    """Cochran's Q I^2 bar chart for top-n systems by Q p-value."""
    if loco_summary is None or loco_summary.empty:
        return
    q_col = f"{rank}_Q_p_bonferroni"
    i2_col = f"{rank}_I2"
    if q_col not in loco_summary.columns:
        return
    set_publication_rcparams()
    d = loco_summary.sort_values(q_col, na_position="last").head(n)
    fig, ax = plt.subplots(figsize=(5.5, max(3.0, 0.22 * len(d))))
    colors = [PALETTE["vermil"] if hq < 0.05 else PALETTE["skyblue"]
              for hq in d[q_col].fillna(1.0)]
    ax.barh(d["defense_system"], d[i2_col].fillna(0), color=colors)
    ax.axvline(0.5, color=PALETTE["black"], lw=0.4, ls=":")
    ax.set_xlabel(f"I² across {rank} leave-outs")
    ax.invert_yaxis()
    ax.set_title(f"LOCO heterogeneity ({rank}); red = Q p-Bonf < 0.05")
    _save(fig, figure_dir, f"loco_heterogeneity_{rank}")


# ======================================================================
# Burden (phylogenetically-corrected)
# ======================================================================

def plot_burden_summary(pgls_burden: pd.DataFrame, burden_phyloglm: pd.DataFrame,
                        figure_dir: Path) -> None:
    """Two-panel: (a) PGLS coefficient w/ 95% CI; (b) phyloglm OR for burden."""
    if (pgls_burden is None or pgls_burden.empty) and \
       (burden_phyloglm is None or burden_phyloglm.empty):
        return
    set_publication_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    if pgls_burden is not None and not pgls_burden.empty:
        r = pgls_burden.iloc[0]
        b = r.get("pgls_coefficient", np.nan)
        se = r.get("pgls_std_err", np.nan)
        lam = r.get("pagel_lambda", np.nan)
        axes[0].errorbar([0], [b], yerr=[1.96 * se] if np.isfinite(se) else [0],
                         fmt="o", color=PALETTE["blue"])
        axes[0].axhline(0, color=PALETTE["black"], lw=0.5)
        axes[0].set_xticks([0]); axes[0].set_xticklabels(["plasmid+ vs plasmid-"])
        axes[0].set_ylabel("PGLS coefficient (burden)")
        axes[0].set_title(f"λ̂ = {lam:.2f} (Pagel ML)" if np.isfinite(lam) else "PGLS")

    if burden_phyloglm is not None and not burden_phyloglm.empty:
        r = burden_phyloglm.iloc[0]
        b = r.get("phyloglm_coefficient", np.nan)
        se = r.get("phyloglm_std_err", np.nan)
        axes[1].errorbar([0], [np.exp(b)],
                         yerr=[[np.exp(b) - np.exp(b - 1.96 * se)],
                               [np.exp(b + 1.96 * se) - np.exp(b)]]
                         if np.isfinite(se) else [[0], [0]],
                         fmt="o", color=PALETTE["orange"])
        axes[1].axhline(1.0, color=PALETTE["black"], lw=0.5)
        axes[1].set_xticks([0]); axes[1].set_xticklabels(["burden_count"])
        axes[1].set_ylabel("phyloglm odds ratio")
        axes[1].set_title("plasmid ~ burden")
    _save(fig, figure_dir, "burden_phylo_summary")


# ======================================================================
# Multivariate + stability
# ======================================================================

def plot_multivariate_stability(stability: pd.DataFrame, figure_dir: Path) -> None:
    """Two-panel: PGLMM coefficient + LASSO subsample-selection frequency."""
    if stability is None or stability.empty:
        return
    set_publication_rcparams()
    d = stability.sort_values("pglmm_p_value" if "pglmm_p_value" in stability.columns
                              else "lasso_stab_freq", na_position="last")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, max(3.0, 0.22 * len(d))),
                             sharey=True)
    y = np.arange(len(d))
    if "pglmm_coefficient" in d.columns:
        colors = [PALETTE["vermil"] if (np.isfinite(q) and q < 0.05) else PALETTE["skyblue"]
                  for q in d.get("pglmm_fdr_qvalue", pd.Series([np.nan] * len(d)))]
        axes[0].scatter(d["pglmm_coefficient"], y, c=colors, s=22)
        axes[0].axvline(0, color=PALETTE["black"], lw=0.5)
        axes[0].set_yticks(y); axes[0].set_yticklabels(d["defense_system"], fontsize=7)
        axes[0].set_xlabel("PGLMM coefficient")
        axes[0].set_title("Multivariate PGLMM (Brownian random effect)")
        axes[0].invert_yaxis()
    if "lasso_stab_freq" in d.columns:
        axes[1].barh(y, d["lasso_stab_freq"].fillna(0), color=PALETTE["orange"])
        axes[1].axvline(0.60, color=PALETTE["black"], lw=0.4, ls=":")
        axes[1].set_xlabel("LASSO subsample selection frequency")
        axes[1].set_title("Stability (one-SE rule)")
    _save(fig, figure_dir, "multivariate_stability")


# ======================================================================
# Burden descriptive figures
# ======================================================================

def plot_burden_violin(binary_df: pd.DataFrame, pgls_burden_df: Optional[pd.DataFrame],
                       figure_dir: Path) -> None:
    if "defense_burden_count" not in binary_df.columns:
        return
    set_publication_rcparams()
    pos = binary_df.loc[binary_df["has_plasmid_binary"] == 1, "defense_burden_count"].values
    neg = binary_df.loc[binary_df["has_plasmid_binary"] == 0, "defense_burden_count"].values
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    parts = ax.violinplot([neg, pos], showmeans=False, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(PALETTE["grey"] if i == 0 else PALETTE["orange"])
        pc.set_alpha(0.6)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["no plasmid", "plasmid"])
    ax.set_ylabel("Total defense systems per species")
    # Annotate with PGLS p-value (phylogenetically corrected), not Mann-Whitney
    if pgls_burden_df is not None and not pgls_burden_df.empty:
        pval = pgls_burden_df.iloc[0].get("pgls_p_value", np.nan)
        lam = pgls_burden_df.iloc[0].get("pagel_lambda", np.nan)
        if np.isfinite(pval):
            ax.set_title(f"Defense burden by plasmid status "
                         f"(PGLS p={pval:.2g}, λ̂={lam:.2f})")
        else:
            ax.set_title("Defense burden by plasmid status")
    else:
        ax.set_title("Defense burden by plasmid status")
    _save(fig, figure_dir, "burden_violin")


def plot_burden_histogram(binary_df: pd.DataFrame, figure_dir: Path) -> None:
    if "defense_burden_count" not in binary_df.columns:
        return
    set_publication_rcparams()
    pos = binary_df.loc[binary_df["has_plasmid_binary"] == 1, "defense_burden_count"].values
    neg = binary_df.loc[binary_df["has_plasmid_binary"] == 0, "defense_burden_count"].values
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    bins = np.arange(0, max(binary_df["defense_burden_count"].max() + 2, 2))
    ax.hist(neg, bins=bins, alpha=0.55, color=PALETTE["grey"], label="no plasmid",
            density=True)
    ax.hist(pos, bins=bins, alpha=0.55, color=PALETTE["orange"], label="plasmid",
            density=True)
    ax.set_xlabel("Defense systems per species")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    _save(fig, figure_dir, "burden_histogram")


def plot_plasmid_probability_curve(binary_df: pd.DataFrame, figure_dir: Path) -> None:
    if "defense_burden_count" not in binary_df.columns:
        return
    set_publication_rcparams()
    # Observed P(plasmid | burden = k)
    df = binary_df[["defense_burden_count", "has_plasmid_binary"]].copy()
    agg = df.groupby("defense_burden_count")["has_plasmid_binary"].agg(["mean", "count"]).reset_index()
    agg = agg[agg["count"] >= 5]      # suppress tails with too few species
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(agg["defense_burden_count"], agg["mean"], "o-",
            color=PALETTE["blue"], ms=4)
    # Bar of species counts on right axis for transparency
    ax2 = ax.twinx()
    ax2.bar(agg["defense_burden_count"], agg["count"], alpha=0.15,
            color=PALETTE["grey"], width=0.8)
    ax.set_xlabel("Defense systems per species")
    ax.set_ylabel("Observed P(plasmid)")
    ax2.set_ylabel("n species (shaded bars)", color=PALETTE["grey"])
    ax.set_ylim(0, 1.02)
    _save(fig, figure_dir, "burden_plasmid_probability")


def plot_prevalence_scatter(tier1: pd.DataFrame, figure_dir: Path) -> None:
    """Per-system: prevalence among plasmid+ vs plasmid-. Diagonal reference."""
    if tier1 is None or tier1.empty:
        return
    needed = {"plasmid_rate_with_defense", "plasmid_rate_without_defense"}
    # We actually want prevalence OF the system IN plasmid+ vs IN plasmid-.
    # Those can be reconstructed from fisher contingency counts:
    #   prev_in_plasmid_pos = n_present_with_plasmid / (n_present_with_plasmid + n_absent_with_plasmid)
    # All four counts are in tier1 columns (diag_fisher_n_present_with_plasmid etc.)
    if not all(c in tier1.columns for c in
               ["diag_fisher_n_present_with_plasmid",
                "diag_fisher_n_absent_with_plasmid",
                "diag_fisher_n_present_no_plasmid",
                "diag_fisher_n_absent_no_plasmid"]):
        return
    d = tier1.copy()
    denom_pos = d["diag_fisher_n_present_with_plasmid"] + d["diag_fisher_n_absent_with_plasmid"]
    denom_neg = d["diag_fisher_n_present_no_plasmid"] + d["diag_fisher_n_absent_no_plasmid"]
    d["prev_in_plasmid_pos"] = d["diag_fisher_n_present_with_plasmid"] / denom_pos
    d["prev_in_plasmid_neg"] = d["diag_fisher_n_present_no_plasmid"] / denom_neg

    set_publication_rcparams()
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    colors = [PALETTE["vermil"]
              if (np.isfinite(q) and q < 0.05) else PALETTE["grey"]
              for q in d.get("firth_weighted_fdr_qvalue", pd.Series([np.nan] * len(d)))]
    ax.scatter(d["prev_in_plasmid_neg"], d["prev_in_plasmid_pos"],
               c=colors, s=18, alpha=0.75, edgecolor="none")
    lim = max(d[["prev_in_plasmid_pos", "prev_in_plasmid_neg"]].max().max(), 0.01)
    ax.plot([0, lim], [0, lim], color=PALETTE["black"], lw=0.5, ls="--")
    ax.set_xlabel("Prevalence in plasmid-negative species")
    ax.set_ylabel("Prevalence in plasmid-positive species")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(f"{len(d)} defense systems; red = Tier 1 FDR < 0.05")
    _save(fig, figure_dir, "burden_prevalence_scatter")


def plot_volcano(tier2_phyloglm: pd.DataFrame, figure_dir: Path, alpha: float = 0.05) -> None:
    """phyloglm coefficient vs -log10(q): the standard volcano, but based on
    the phylogenetically corrected test (not Fisher, as the old script did).
    """
    if tier2_phyloglm is None or tier2_phyloglm.empty:
        return
    set_publication_rcparams()
    d = tier2_phyloglm.copy()
    d = d[np.isfinite(d["phyloglm_coefficient"]) & np.isfinite(d["phyloglm_fdr_qvalue"])]
    if d.empty:
        return
    y = -np.log10(np.clip(d["phyloglm_fdr_qvalue"], 1e-300, 1.0))
    sig = d["phyloglm_fdr_qvalue"] < alpha
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.scatter(d.loc[~sig, "phyloglm_coefficient"], y[~sig],
               c=PALETTE["grey"], s=12, alpha=0.6, edgecolor="none")
    ax.scatter(d.loc[sig, "phyloglm_coefficient"], y[sig],
               c=PALETTE["vermil"], s=18, alpha=0.9, edgecolor="none")
    ax.axhline(-np.log10(alpha), color=PALETTE["black"], lw=0.4, ls=":")
    ax.axvline(0, color=PALETTE["black"], lw=0.4, ls=":")
    ax.set_xlabel("phyloglm coefficient (logit)")
    ax.set_ylabel(r"$-\log_{10}$(phyloglm FDR q)")
    # Label the top 8 by |coefficient| among significant hits
    top = d.loc[sig].reindex(d.loc[sig, "phyloglm_coefficient"].abs()
                              .sort_values(ascending=False).index).head(8)
    for _, r in top.iterrows():
        ax.annotate(r["defense_system"],
                    (r["phyloglm_coefficient"],
                     -np.log10(max(r["phyloglm_fdr_qvalue"], 1e-300))),
                    fontsize=6, alpha=0.9,
                    xytext=(3, 3), textcoords="offset points")
    _save(fig, figure_dir, "volcano_phyloglm")


def plot_composition_by_burden(binary_df: pd.DataFrame, defense_cols: List[str],
                               figure_dir: Path) -> None:
    if "defense_burden_count" not in binary_df.columns:
        return
    set_publication_rcparams()
    bins = pd.cut(binary_df["defense_burden_count"],
                  bins=[-0.1, 0, 2, 5, 10, 20, binary_df["defense_burden_count"].max() + 1],
                  labels=["0", "1-2", "3-5", "6-10", "11-20", "21+"])
    cats = sorted({classify_defense_system(c) for c in defense_cols})
    # Fraction of total category count among species in each bin
    rows = []
    for b in bins.cat.categories:
        mask = bins == b
        if mask.sum() == 0:
            continue
        sub = binary_df.loc[mask, defense_cols]
        cat_counts = {cat: 0 for cat in cats}
        for c in defense_cols:
            cat_counts[classify_defense_system(c)] += sub[c].sum()
        total = sum(cat_counts.values())
        rows.append({"bin": b,
                     **{cat: cat_counts[cat] / total if total > 0 else 0 for cat in cats},
                     "n_species": int(mask.sum())})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    bottom = np.zeros(len(df))
    palette_cycle = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"],
                     PALETTE["skyblue"], PALETTE["vermil"], PALETTE["pink"],
                     PALETTE["yellow"], PALETTE["grey"]]
    for i, cat in enumerate(cats):
        vals = df[cat].values
        ax.bar(df["bin"].astype(str), vals, bottom=bottom,
               color=palette_cycle[i % len(palette_cycle)], label=cat,
               edgecolor="white", lw=0.4)
        bottom += vals
    ax.set_ylabel("Fraction of defense-system hits")
    ax.set_xlabel("Defense burden bin")
    ax.legend(fontsize=6, ncol=2, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    _save(fig, figure_dir, "burden_composition_by_category")


def plot_enrichment_bars(tier2_phyloglm: pd.DataFrame, figure_dir: Path,
                         n: int = 25) -> None:
    """Top-n defense systems by phyloglm |coefficient|, as paired bars
    (plasmid+ prevalence and plasmid- prevalence reconstructed from tier1
    counts if available). Falls back to phyloglm coefficient if counts absent.
    """
    if tier2_phyloglm is None or tier2_phyloglm.empty:
        return
    set_publication_rcparams()
    d = tier2_phyloglm.reindex(tier2_phyloglm["phyloglm_coefficient"].abs()
                                .sort_values(ascending=False).index).head(n)
    fig, ax = plt.subplots(figsize=(5.0, max(3.0, 0.2 * len(d))))
    y = np.arange(len(d))
    ax.barh(y, d["phyloglm_coefficient"],
            color=[PALETTE["orange"] if c > 0 else PALETTE["blue"]
                   for c in d["phyloglm_coefficient"]])
    ax.axvline(0, color=PALETTE["black"], lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels(d["defense_system"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("phyloglm coefficient")
    ax.set_title(f"Top {len(d)} systems by |phyloglm effect|")
    _save(fig, figure_dir, "enrichment_top_systems")


def plot_burden_by_phylum(binary_df: pd.DataFrame, figure_dir: Path,
                          n: int = 10) -> None:
    if "defense_burden_count" not in binary_df.columns or \
       "gtdb_phylum" not in binary_df.columns:
        return
    set_publication_rcparams()
    top_phyla = binary_df["gtdb_phylum"].value_counts().head(n).index
    rows = []
    for ph in top_phyla:
        sub = binary_df[binary_df["gtdb_phylum"] == ph]
        pos = sub.loc[sub["has_plasmid_binary"] == 1, "defense_burden_count"].mean()
        neg = sub.loc[sub["has_plasmid_binary"] == 0, "defense_burden_count"].mean()
        rows.append({"phylum": ph,
                     "mean_pos": pos, "mean_neg": neg,
                     "gap": (pos or 0) - (neg or 0),
                     "n": len(sub)})
    df = pd.DataFrame(rows).sort_values("gap", ascending=True)
    fig, ax = plt.subplots(figsize=(5.5, max(3.0, 0.3 * len(df))))
    y = np.arange(len(df))
    ax.barh(y, df["gap"], color=[PALETTE["orange"] if g > 0 else PALETTE["blue"]
                                 for g in df["gap"]])
    ax.axvline(0, color=PALETTE["black"], lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels(df["phylum"], fontsize=7)
    for i, r in enumerate(df.itertuples()):
        ax.text(r.gap, i, f" n={r.n}", va="center", fontsize=6, color=PALETTE["grey"])
    ax.set_xlabel("Mean burden (plasmid+) − mean burden (plasmid−)")
    ax.set_title(f"Burden gap across top {len(df)} phyla")
    _save(fig, figure_dir, "burden_by_phylum")


def plot_conditional_plasmid_rate(binary_df: pd.DataFrame, tier1: Optional[pd.DataFrame],
                                  figure_dir: Path, min_prev: float = 0.10,
                                  n: int = 20) -> None:
    """P(plasmid | system present) for systems whose prevalence >= min_prev.
    Baseline plasmid rate shown as dashed line.
    """
    if tier1 is None or tier1.empty:
        return
    set_publication_rcparams()
    d = tier1[tier1["defense_prevalence"] >= min_prev].copy()
    if d.empty:
        return
    d = d.sort_values("plasmid_rate_with_defense", ascending=False).head(n)
    baseline = float(binary_df["has_plasmid_binary"].mean())
    fig, ax = plt.subplots(figsize=(5.5, max(3.0, 0.22 * len(d))))
    y = np.arange(len(d))
    ax.barh(y, d["plasmid_rate_with_defense"],
            color=PALETTE["orange"], label="P(plasmid | system present)")
    ax.axvline(baseline, color=PALETTE["black"], lw=0.8, ls="--",
               label=f"overall P(plasmid) = {baseline:.2f}")
    ax.set_yticks(y); ax.set_yticklabels(d["defense_system"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("P(plasmid | system present)")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    _save(fig, figure_dir, "conditional_plasmid_rate")


def plot_cooccurrence_heatmap(binary_df: pd.DataFrame, defense_cols: List[str],
                              figure_dir: Path, top_n: int = 20) -> None:
    """Jaccard similarity among the top-n most prevalent defense systems."""
    prev = binary_df[defense_cols].mean().sort_values(ascending=False).head(top_n)
    cols = prev.index.tolist()
    if len(cols) < 2:
        return
    set_publication_rcparams()
    m = binary_df[cols].values.astype(int)
    jacc = np.zeros((len(cols), len(cols)))
    for i in range(len(cols)):
        for j in range(len(cols)):
            inter = int(((m[:, i] == 1) & (m[:, j] == 1)).sum())
            union = int(((m[:, i] == 1) | (m[:, j] == 1)).sum())
            jacc[i, j] = inter / union if union > 0 else 0
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(jacc, cmap="viridis", vmin=0, vmax=min(1.0, jacc.max()))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=6)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=6)
    cb = plt.colorbar(im, ax=ax, shrink=0.7); cb.set_label("Jaccard similarity", fontsize=7)
    _save(fig, figure_dir, "cooccurrence_jaccard")


# ======================================================================
# Consensus / whole-pipeline figures
# ======================================================================

def plot_manhattan(tier2_phyloglm: pd.DataFrame, figure_dir: Path,
                   alpha: float = 0.05) -> None:
    """Manhattan-style plot: -log10 q across systems, ordered by absolute
    coefficient, coloured by category."""
    if tier2_phyloglm is None or tier2_phyloglm.empty:
        return
    set_publication_rcparams()
    d = tier2_phyloglm.dropna(subset=["phyloglm_fdr_qvalue"]).copy()
    if d.empty:
        return
    d["category"] = d["defense_system"].map(classify_defense_system)
    d = d.sort_values("category")
    cats = d["category"].unique()
    cat_color = {c: list(PALETTE.values())[i % len(PALETTE)] for i, c in enumerate(cats)}
    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    x = np.arange(len(d))
    y = -np.log10(np.clip(d["phyloglm_fdr_qvalue"], 1e-300, 1.0))
    colors = [cat_color[c] for c in d["category"]]
    ax.scatter(x, y, c=colors, s=12, alpha=0.8, edgecolor="none")
    ax.axhline(-np.log10(alpha), color=PALETTE["black"], lw=0.5, ls=":")
    # Category labels on x
    ticks, labels = [], []
    for cat in cats:
        idx = np.where(d["category"].values == cat)[0]
        ticks.append(idx.mean()); labels.append(cat)
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel(r"$-\log_{10}$(phyloglm FDR q)")
    ax.set_title(f"{len(d)} systems — phylogenetic Manhattan, red line = q={alpha}")
    _save(fig, figure_dir, "manhattan_phyloglm")


def plot_top_systems_summary_panel(consensus: pd.DataFrame,
                                   tier2_phyloglm: pd.DataFrame,
                                   pglmm: Optional[pd.DataFrame],
                                   figure_dir: Path, n: int = 15) -> None:
    """Four-panel summary: for top-n systems by rank product,
    show (A) phyloglm OR w/ CI, (B) PGLMM coefficient, (C) Pagel p-value,
    (D) consensus rank product.
    """
    if consensus is None or consensus.empty:
        return
    set_publication_rcparams()
    d = consensus.sort_values("rank_product").head(n)
    systems = d["defense_system"].tolist()
    phy = tier2_phyloglm.set_index("defense_system").reindex(systems) \
        if tier2_phyloglm is not None and not tier2_phyloglm.empty else None
    mv = pglmm[pglmm["defense_system"] != "(Intercept)"].set_index("defense_system").reindex(systems) \
        if pglmm is not None and not pglmm.empty else None

    fig, axes = plt.subplots(1, 4, figsize=(11, max(3.0, 0.25 * n)), sharey=True)
    y = np.arange(n)

    # (A) phyloglm OR
    if phy is not None:
        or_ = phy["phyloglm_odds_ratio"].values
        lo = phy["phyloglm_ci_low"].values
        hi = phy["phyloglm_ci_high"].values
        axes[0].errorbar(np.log2(or_), y,
                         xerr=[np.log2(or_) - np.log2(lo), np.log2(hi) - np.log2(or_)],
                         fmt="o", color=PALETTE["blue"], mfc=PALETTE["blue"], ms=4)
        axes[0].axvline(0, color=PALETTE["black"], lw=0.5)
        axes[0].set_xlabel("log₂(OR) — phyloglm")

    # (B) PGLMM
    if mv is not None:
        axes[1].scatter(mv["pglmm_coefficient"], y, color=PALETTE["orange"], s=24)
        axes[1].axvline(0, color=PALETTE["black"], lw=0.5)
        axes[1].set_xlabel("PGLMM β")

    # (C) Pagel p
    if "pagel_p_value" in d.columns:
        axes[2].barh(y, -np.log10(d["pagel_p_value"].fillna(1.0)), color=PALETTE["green"])
        axes[2].axvline(-np.log10(0.05), color=PALETTE["black"], lw=0.5, ls=":")
        axes[2].set_xlabel(r"$-\log_{10}$(Pagel p)")

    # (D) rank product
    axes[3].barh(y, d["rank_product"], color=PALETTE["pink"])
    axes[3].set_xlabel("rank product (↓ = stronger)")
    axes[3].set_yticks(y); axes[3].set_yticklabels(d["defense_system"], fontsize=7)
    axes[3].invert_yaxis()

    for ax in axes[:3]:
        ax.set_yticks(y); ax.set_yticklabels([])
        ax.invert_yaxis()

    _save(fig, figure_dir, "top_systems_summary_panel")


def plot_consensus_dot_bubble(consensus: pd.DataFrame, figure_dir: Path,
                              n: int = 25) -> None:
    if consensus is None or consensus.empty:
        return
    set_publication_rcparams()
    d = consensus.sort_values("rank_product").head(n).copy()
    fig, ax = plt.subplots(figsize=(6.0, max(3.5, 0.28 * n)))
    y = np.arange(len(d))
    sizes = 20 + 40 * (1 - (d["cauchy_combined_p"].fillna(1.0).rank(pct=True) - 0))
    colors = [PALETTE["orange"] if (np.isfinite(c) and c > 0) else PALETTE["blue"]
              for c in d.get("phyloglm_coefficient", pd.Series([np.nan] * len(d)))]
    ax.scatter(d["rank_product"], y, s=sizes, c=colors, alpha=0.8, edgecolor="none")
    ax.set_yticks(y); ax.set_yticklabels(d["defense_system"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Rank product")
    ax.set_title("Consensus bubble chart; size ∝ Cauchy combined p strength")
    _save(fig, figure_dir, "consensus_dot_bubble")


def plot_consensus_vs_phylo_signal(consensus: pd.DataFrame,
                                   phylo_signal: Optional[pd.DataFrame],
                                   figure_dir: Path) -> None:
    if consensus is None or consensus.empty or phylo_signal is None or phylo_signal.empty:
        return
    set_publication_rcparams()
    d = consensus.merge(phylo_signal.rename(columns={"column": "defense_system"}),
                        on="defense_system", how="left")
    d = d.dropna(subset=["rank_product", "D"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.scatter(d["D"], d["rank_product"], s=10, c=PALETTE["grey"], alpha=0.5, edgecolor="none")
    top = d.sort_values("rank_product").head(10)
    ax.scatter(top["D"], top["rank_product"], s=30, c=PALETTE["vermil"], zorder=3)
    for _, r in top.iterrows():
        ax.annotate(r["defense_system"], (r["D"], r["rank_product"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axvline(0, color=PALETTE["black"], lw=0.4, ls=":")
    ax.axvline(1, color=PALETTE["black"], lw=0.4, ls=":")
    ax.set_xlabel("Fritz-Purvis D (0=Brownian, 1=random)")
    ax.set_ylabel("Rank product (↓ = stronger evidence)")
    ax.set_title("Consensus vs. phylogenetic signal")
    _save(fig, figure_dir, "consensus_vs_phylo_signal")


def plot_consensus_with_robustness(consensus: pd.DataFrame,
                                   loco_summary: Optional[pd.DataFrame],
                                   figure_dir: Path, n: int = 25) -> None:
    if consensus is None or consensus.empty or loco_summary is None or loco_summary.empty:
        return
    set_publication_rcparams()
    d = consensus.sort_values("rank_product").head(n)
    d = d.merge(loco_summary, on="defense_system", how="left")
    rank_col = "gtdb_class_I2" if "gtdb_class_I2" in d.columns else "gtdb_phylum_I2"
    het_col = "gtdb_class_is_heterogeneous" \
        if "gtdb_class_is_heterogeneous" in d.columns else "gtdb_phylum_is_heterogeneous"
    fig, ax = plt.subplots(figsize=(6.5, max(3.5, 0.28 * n)))
    y = np.arange(len(d))
    ax.barh(y, d["rank_product"], color=PALETTE["skyblue"])
    for i, (_, r) in enumerate(d.iterrows()):
        tag = ""
        if het_col in d.columns and bool(r.get(het_col)):
            tag = "⚠ heterogeneous"
        if rank_col in d.columns and np.isfinite(r.get(rank_col, np.nan)):
            tag += f"  I²={r[rank_col]:.2f}"
        ax.text(r["rank_product"], i, " " + tag, va="center", fontsize=6,
                color=PALETTE["vermil"] if "⚠" in tag else PALETTE["grey"])
    ax.set_yticks(y); ax.set_yticklabels(d["defense_system"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Rank product")
    _save(fig, figure_dir, "consensus_with_robustness")


def plot_rank_product_bar(consensus: pd.DataFrame, figure_dir: Path, n: int = 30) -> None:
    if consensus is None or consensus.empty:
        return
    set_publication_rcparams()
    d = consensus.sort_values("rank_product").head(n)
    fig, ax = plt.subplots(figsize=(5.0, max(3.0, 0.22 * len(d))))
    y = np.arange(len(d))
    ax.barh(y, 1 / d["rank_product"], color=PALETTE["blue"])
    ax.set_yticks(y); ax.set_yticklabels(d["defense_system"], fontsize=7)
    ax.set_xlabel("1 / rank product")
    ax.invert_yaxis()
    ax.set_title(f"Top {len(d)} by rank product across phyloglm + PGLMM + Pagel's")
    _save(fig, figure_dir, "rank_product_bar")


def plot_rank_product_heatmap(consensus: pd.DataFrame, figure_dir: Path, n: int = 30) -> None:
    if consensus is None or consensus.empty:
        return
    set_publication_rcparams()
    d = consensus.sort_values("rank_product").head(n)
    rank_cols = [c for c in ["phyloglm_p_value_rank",
                             "pglmm_p_value_rank",
                             "pagel_p_value_rank"] if c in d.columns]
    if not rank_cols:
        return
    mat = d[rank_cols].values
    fig, ax = plt.subplots(figsize=(3.5, max(3.0, 0.22 * len(d))))
    im = ax.imshow(mat, cmap="viridis_r", aspect="auto")
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d["defense_system"], fontsize=7)
    ax.set_xticks(range(len(rank_cols)))
    ax.set_xticklabels([c.replace("_p_value_rank", "") for c in rank_cols],
                       rotation=45, ha="right")
    plt.colorbar(im, ax=ax, shrink=0.7, label="rank (lower = stronger)")
    _save(fig, figure_dir, "rank_product_heatmap")


# ======================================================================
# Key-findings figures
# ======================================================================

def plot_gabija_multi_method(combined: pd.DataFrame, figure_dir: Path) -> None:
    """Pull Gabija-flagged rows and show phyloglm, PGLMM, Pagel, LASSO, Fisher
    side-by-side."""
    if combined is None or combined.empty:
        return
    set_publication_rcparams()
    mask = combined["defense_system"].str.lower().str.contains("gabija")
    d = combined[mask]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, max(2.5, 0.25 * len(d) * 3)))
    methods = [
        ("phyloglm_coefficient", "phyloglm β"),
        ("pglmm_coefficient", "PGLMM β"),
        ("lasso_coef", "LASSO β (phylo-resid)"),
        ("firth_weighted_coefficient", "Firth-weighted β (Tier 1)"),
    ]
    y = np.arange(len(methods))
    for i, (_, row) in enumerate(d.iterrows()):
        vals = [row.get(k, np.nan) for k, _ in methods]
        ax.plot(vals, y - 0.15 + 0.3 * i / max(1, len(d) - 1), "o-",
                label=row["defense_system"], alpha=0.9)
    ax.axvline(0, color=PALETTE["black"], lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels([m[1] for m in methods])
    ax.invert_yaxis()
    ax.set_title(f"Gabija across {len(d)} methods")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    _save(fig, figure_dir, "gabija_multi_method")


def plot_rm_prevalence_confounding(combined: pd.DataFrame, figure_dir: Path) -> None:
    """Compare |Fisher log-OR| vs |phyloglm β| for Restriction-Modification
    systems. RM systems should show strong attenuation (Fisher large,
    phyloglm small) — the canonical prevalence-confounding story.
    """
    if combined is None or combined.empty:
        return
    set_publication_rcparams()
    cat_col = "category" if "category" in combined.columns else None
    if cat_col is None:
        combined = combined.copy()
        combined["category"] = combined["defense_system"].map(classify_defense_system)
    rm = combined[combined["category"] == "Restriction-Modification"].dropna(
        subset=["diag_fisher_odds_ratio", "phyloglm_coefficient"])
    if rm.empty:
        return
    fisher_logor = np.log2(rm["diag_fisher_odds_ratio"].replace(0, np.nan))
    phyloglm_logor = rm["phyloglm_coefficient"] / np.log(2)   # convert logit to log2 OR
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(fisher_logor, phyloglm_logor, s=24, color=PALETTE["vermil"], alpha=0.8)
    lim = max(abs(fisher_logor).max(), abs(phyloglm_logor).max(), 1.0)
    ax.plot([-lim, lim], [-lim, lim], color=PALETTE["black"], lw=0.5, ls="--")
    ax.axhline(0, color=PALETTE["black"], lw=0.4)
    ax.axvline(0, color=PALETTE["black"], lw=0.4)
    ax.set_xlabel("log₂ Fisher OR (Tier 1)")
    ax.set_ylabel("log₂ phyloglm OR (Tier 2)")
    ax.set_title(f"RM systems: phylogenetic attenuation (n={len(rm)})")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    _save(fig, figure_dir, "rm_prevalence_confounding")


def plot_loco_reversal_detail(loco_detail: pd.DataFrame, full_phyloglm: pd.DataFrame,
                              figure_dir: Path, top_n: int = 15) -> None:
    """For each system where at least one clade-drop reversed the direction
    of the phyloglm coefficient, show the full-dataset effect plus each
    per-clade effect."""
    if loco_detail is None or loco_detail.empty or \
       full_phyloglm is None or full_phyloglm.empty:
        return
    set_publication_rcparams()
    full = full_phyloglm.set_index("defense_system")["phyloglm_coefficient"].to_dict()

    def _has_reversal(g):
        full_sign = np.sign(full.get(g.name, 0))
        if full_sign == 0:
            return False
        drop_signs = np.sign(g["phyloglm_coefficient"].dropna())
        return (drop_signs != full_sign).any()

    rev = loco_detail.groupby("defense_system").filter(_has_reversal)
    if rev.empty:
        return
    # Pick top_n by magnitude of full-dataset coefficient
    systems = (pd.Series({k: abs(v) for k, v in full.items() if k in rev["defense_system"].unique()})
               .sort_values(ascending=False).head(top_n).index.tolist())
    d = rev[rev["defense_system"].isin(systems)]
    fig, ax = plt.subplots(figsize=(6.5, max(3.0, 0.3 * len(systems))))
    y_map = {s: i for i, s in enumerate(systems)}
    # Per-clade dots
    for _, row in d.iterrows():
        ax.scatter(row["phyloglm_coefficient"], y_map[row["defense_system"]],
                   color=PALETTE["grey"], s=10, alpha=0.6, edgecolor="none")
    # Full-dataset coefficient as larger marker
    for s in systems:
        ax.scatter(full[s], y_map[s], color=PALETTE["vermil"], s=50,
                   zorder=3, label="full dataset" if s == systems[0] else "")
    ax.axvline(0, color=PALETTE["black"], lw=0.5)
    ax.set_yticks(list(y_map.values())); ax.set_yticklabels(list(y_map.keys()), fontsize=7)
    ax.set_xlabel("phyloglm coefficient")
    ax.invert_yaxis()
    ax.set_title(f"{len(systems)} systems whose direction reverses when one clade is dropped")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    _save(fig, figure_dir, "loco_reversal_detail")


def _top_takehome(df: pd.DataFrame, coef_col: str, p_col: str, q_col: str,
                  title: str, figure_dir: Path, name: str, n: int = 15,
                  alpha: float = 0.05) -> None:
    if df is None or df.empty or coef_col not in df.columns:
        return
    set_publication_rcparams()
    d = df.dropna(subset=[coef_col]).reindex(
        df[coef_col].abs().sort_values(ascending=False).index).head(n)
    fig, ax = plt.subplots(figsize=(5.5, max(3.0, 0.22 * len(d))))
    y = np.arange(len(d))
    sig_col = q_col if q_col in d.columns else p_col
    colors = [PALETTE["vermil"] if (np.isfinite(x) and x < alpha) else PALETTE["grey"]
              for x in d.get(sig_col, pd.Series([np.nan] * len(d)))]
    ax.barh(y, d[coef_col], color=colors)
    ax.axvline(0, color=PALETTE["black"], lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels(d["defense_system"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(coef_col)
    ax.set_title(title)
    _save(fig, figure_dir, name)


def plot_takehome_fisher(tier1: pd.DataFrame, figure_dir: Path) -> None:
    if tier1 is None or tier1.empty:
        return
    d = tier1.copy()
    d["log2_or"] = np.log2(d["diag_fisher_odds_ratio"].replace(0, np.nan))
    _top_takehome(d, "log2_or", "diag_fisher_p_value", "diag_fisher_fdr_qvalue",
                  "Top Fisher log₂ OR (Tier 1, diagnostic)", figure_dir,
                  "takehome_fisher")


def plot_takehome_phyloglm(tier2_phyloglm: pd.DataFrame, figure_dir: Path) -> None:
    _top_takehome(tier2_phyloglm, "phyloglm_coefficient",
                  "phyloglm_p_value", "phyloglm_fdr_qvalue",
                  "Top phyloglm β (Tier 2, primary)",
                  figure_dir, "takehome_phyloglm")


def plot_takehome_multivariate(pglmm: Optional[pd.DataFrame], figure_dir: Path) -> None:
    if pglmm is None or pglmm.empty:
        return
    mv = pglmm[pglmm["defense_system"] != "(Intercept)"]
    _top_takehome(mv, "pglmm_coefficient", "pglmm_p_value", "pglmm_fdr_qvalue",
                  "Top PGLMM β (multivariate + phylo random effect)",
                  figure_dir, "takehome_pglmm")


def plot_takehome_lasso(lasso: Optional[pd.DataFrame], figure_dir: Path) -> None:
    if lasso is None or lasso.empty or "coefficient" not in lasso.columns:
        return
    set_publication_rcparams()
    d = lasso.reindex(lasso["coefficient"].abs().sort_values(ascending=False).index).head(15)
    fig, ax = plt.subplots(figsize=(5.5, max(3.0, 0.22 * len(d))))
    y = np.arange(len(d))
    colors = [PALETTE["vermil"] if sel else PALETTE["grey"]
              for sel in d.get("stable_selection", pd.Series([False] * len(d)))]
    ax.barh(y, d["coefficient"], color=colors)
    ax.axvline(0, color=PALETTE["black"], lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels(d["defense_system"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("LASSO coefficient (one-SE rule, phylo-residualised)")
    ax.set_title("LASSO takehome (red = stable selection)")
    _save(fig, figure_dir, "takehome_lasso")


def plot_takehome_prevalence_matching(prev_match: Optional[pd.DataFrame],
                                      figure_dir: Path) -> None:
    if prev_match is None or prev_match.empty:
        return
    _top_takehome(prev_match, "matched_effect",
                  "matched_p_value", "matched_fdr_qvalue",
                  "Top prevalence-matched effect",
                  figure_dir, "takehome_prevalence_matched")


def plot_cross_method_consistency(tier2_phyloglm: Optional[pd.DataFrame],
                                  pglmm: Optional[pd.DataFrame],
                                  figure_dir: Path) -> None:
    if tier2_phyloglm is None or tier2_phyloglm.empty \
            or pglmm is None or pglmm.empty:
        return
    set_publication_rcparams()
    mv = pglmm[pglmm["defense_system"] != "(Intercept)"]
    d = tier2_phyloglm.merge(mv, on="defense_system", how="inner",
                             suffixes=("_uni", "_mv"))
    d = d.dropna(subset=["phyloglm_coefficient", "pglmm_coefficient"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(d["phyloglm_coefficient"], d["pglmm_coefficient"],
               s=18, c=PALETTE["skyblue"], alpha=0.8, edgecolor="none")
    # Flag disagreements
    disagree = np.sign(d["phyloglm_coefficient"]) != np.sign(d["pglmm_coefficient"])
    ax.scatter(d.loc[disagree, "phyloglm_coefficient"],
               d.loc[disagree, "pglmm_coefficient"],
               s=28, c=PALETTE["vermil"], alpha=0.9, label="sign disagreement")
    lim = max(np.nanmax(np.abs(d["phyloglm_coefficient"])),
              np.nanmax(np.abs(d["pglmm_coefficient"])), 1.0)
    ax.plot([-lim, lim], [-lim, lim], color=PALETTE["black"], lw=0.5, ls="--")
    ax.axhline(0, color=PALETTE["black"], lw=0.3)
    ax.axvline(0, color=PALETTE["black"], lw=0.3)
    ax.set_xlabel("phyloglm β (univariate)")
    ax.set_ylabel("PGLMM β (multivariate)")
    try:
        rho, pv = sp_stats.spearmanr(d["phyloglm_coefficient"], d["pglmm_coefficient"])
        ax.set_title(f"Cross-method consistency (Spearman ρ={rho:.2f}, p={pv:.2g})")
    except Exception:
        ax.set_title("Cross-method consistency")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, figure_dir, "cross_method_consistency")


def plot_effect_attenuation(tier1: Optional[pd.DataFrame],
                            tier2_phyloglm: Optional[pd.DataFrame],
                            figure_dir: Path) -> None:
    if tier1 is None or tier1.empty or tier2_phyloglm is None or tier2_phyloglm.empty:
        return
    set_publication_rcparams()
    d = tier1.merge(tier2_phyloglm, on="defense_system", how="inner")
    d = d.dropna(subset=["diag_fisher_odds_ratio", "phyloglm_coefficient"])
    d["log2_fisher_or"] = np.log2(d["diag_fisher_odds_ratio"].replace(0, np.nan))
    d["log2_phyloglm_or"] = d["phyloglm_coefficient"] / np.log(2)
    d = d.dropna(subset=["log2_fisher_or", "log2_phyloglm_or"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.scatter(d["log2_fisher_or"], d["log2_phyloglm_or"],
               s=10, color=PALETTE["grey"], alpha=0.6, edgecolor="none")
    # Highlight top-10 Fisher effects with labels
    top = d.reindex(d["log2_fisher_or"].abs().sort_values(ascending=False).index).head(10)
    ax.scatter(top["log2_fisher_or"], top["log2_phyloglm_or"],
               s=28, color=PALETTE["vermil"], zorder=3)
    for _, r in top.iterrows():
        ax.annotate(r["defense_system"],
                    (r["log2_fisher_or"], r["log2_phyloglm_or"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    lim = max(np.abs(d["log2_fisher_or"]).max(),
              np.abs(d["log2_phyloglm_or"]).max(), 1.0)
    ax.plot([-lim, lim], [-lim, lim], color=PALETTE["black"], lw=0.5, ls="--")
    ax.axhline(0, color=PALETTE["black"], lw=0.3)
    ax.axvline(0, color=PALETTE["black"], lw=0.3)
    ax.set_xlabel("log₂ OR (Fisher, Tier 1)")
    ax.set_ylabel("log₂ OR (phyloglm, Tier 2)")
    ax.set_title("Effect attenuation under phylogenetic correction")
    _save(fig, figure_dir, "effect_attenuation")


# ======================================================================
# Random-forest figures
# ======================================================================

def plot_rf_importance(rf_binary: Optional[pd.DataFrame],
                       rf_prevalence: Optional[pd.DataFrame],
                       figure_dir: Path, n: int = 25) -> None:
    if (rf_binary is None or rf_binary.empty) and \
       (rf_prevalence is None or rf_prevalence.empty):
        return
    set_publication_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, max(3.0, 0.22 * n)), sharey=False)
    for ax, df, title in zip(axes, [rf_binary, rf_prevalence],
                             ["Binary features", "Prevalence features"]):
        if df is None or df.empty or "rf_perm_importance_mean" not in df.columns:
            ax.axis("off"); continue
        d = df.sort_values("rf_perm_importance_mean", ascending=False).head(n)
        y = np.arange(len(d))
        ax.barh(y, d["rf_perm_importance_mean"],
                xerr=d["rf_perm_importance_std"].fillna(0),
                color=PALETTE["skyblue"])
        ax.set_yticks(y); ax.set_yticklabels(d["defense_system"], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Clade-blocked permutation importance")
        ax.set_title(title)
    _save(fig, figure_dir, "rf_clade_blocked_importance")


def plot_rf_fold_auc(fold_aucs: Optional[pd.DataFrame], figure_dir: Path) -> None:
    """Per-fold AUC boxplot to show variance across clades."""
    if fold_aucs is None or fold_aucs.empty:
        return
    set_publication_rcparams()
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    data = [fold_aucs.loc[fold_aucs["feature_set"] == fs, "cv_auc"].values
            for fs in ["binary", "prevalence"]]
    ax.boxplot(data, tick_labels=["binary", "prevalence"])
    ax.axhline(0.5, color=PALETTE["black"], lw=0.5, ls=":")
    ax.set_ylabel("Clade-blocked CV ROC-AUC")
    ax.set_title("RF per-fold AUC (each point = one held-out clade)")
    _save(fig, figure_dir, "rf_fold_auc")


# ======================================================================
# Entry point
# ======================================================================

def make_all_figures(outputs: dict, figure_dir: Path,
                     logger: logging.Logger,
                     highlight: Optional[list] = None,
                     binary_df: Optional[pd.DataFrame] = None,
                     prevalence_df: Optional[pd.DataFrame] = None,
                     defense_cols: Optional[list] = None) -> None:
    """Generate every figure we have data for, logging any that fail.

    ``binary_df``, ``prevalence_df``, and ``defense_cols`` unlock the
    descriptive burden/composition figures. Passing None just skips those.
    """
    figure_dir = Path(figure_dir)

    jobs = [
        # --- Tier 1/2 primary ---
        ("phyloglm forest", plot_phyloglm_forest,
         dict(phyloglm=outputs.get("tier2_phyloglm"), figure_dir=figure_dir)),
        ("manhattan phyloglm", plot_manhattan,
         dict(tier2_phyloglm=outputs.get("tier2_phyloglm"), figure_dir=figure_dir)),
        ("volcano phyloglm", plot_volcano,
         dict(tier2_phyloglm=outputs.get("tier2_phyloglm"), figure_dir=figure_dir)),
        ("enrichment top systems", plot_enrichment_bars,
         dict(tier2_phyloglm=outputs.get("tier2_phyloglm"), figure_dir=figure_dir)),

        # --- Consensus ---
        ("consensus heatmap", plot_consensus_heatmap,
         dict(consensus=outputs.get("consensus"), figure_dir=figure_dir)),
        ("rank product bar", plot_rank_product_bar,
         dict(consensus=outputs.get("consensus"), figure_dir=figure_dir)),
        ("rank product heatmap", plot_rank_product_heatmap,
         dict(consensus=outputs.get("consensus"), figure_dir=figure_dir)),
        ("consensus dot-bubble", plot_consensus_dot_bubble,
         dict(consensus=outputs.get("consensus"), figure_dir=figure_dir)),
        ("consensus vs phylo-signal", plot_consensus_vs_phylo_signal,
         dict(consensus=outputs.get("consensus"),
              phylo_signal=outputs.get("tier3_phylo_signal"),
              figure_dir=figure_dir)),
        ("consensus with robustness", plot_consensus_with_robustness,
         dict(consensus=outputs.get("consensus"),
              loco_summary=outputs.get("tier3_loco_summary"),
              figure_dir=figure_dir)),
        ("top systems summary panel", plot_top_systems_summary_panel,
         dict(consensus=outputs.get("consensus"),
              tier2_phyloglm=outputs.get("tier2_phyloglm"),
              pglmm=outputs.get("tier2_pglmm_mv"),
              figure_dir=figure_dir)),

        # --- Sensitivity / robustness ---
        ("misclass MC trajectories", plot_misclassification_trajectories,
         dict(mc_summary=outputs.get("misclass_summary"),
              figure_dir=figure_dir, system_highlights=highlight)),
        ("misclass analytical", plot_analytical_bias_correction,
         dict(mc_analytical=outputs.get("misclass_analytical"),
              figure_dir=figure_dir)),
        ("LOCO class heterogeneity", plot_loco_heterogeneity,
         dict(loco_summary=outputs.get("tier3_loco_summary"),
              figure_dir=figure_dir, rank="gtdb_class")),
        ("LOCO phylum heterogeneity", plot_loco_heterogeneity,
         dict(loco_summary=outputs.get("tier3_loco_summary"),
              figure_dir=figure_dir, rank="gtdb_phylum")),
        ("LOCO reversal detail", plot_loco_reversal_detail,
         dict(loco_detail=outputs.get("tier3_loco_detail"),
              full_phyloglm=outputs.get("tier2_phyloglm"),
              figure_dir=figure_dir)),

        # --- Burden ---
        ("burden summary", plot_burden_summary,
         dict(pgls_burden=outputs.get("burden_pgls"),
              burden_phyloglm=outputs.get("burden_phyloglm"),
              figure_dir=figure_dir)),
        ("burden violin", plot_burden_violin,
         dict(binary_df=binary_df, pgls_burden_df=outputs.get("burden_pgls"),
              figure_dir=figure_dir)),
        ("burden histogram", plot_burden_histogram,
         dict(binary_df=binary_df, figure_dir=figure_dir)),
        ("burden probability curve", plot_plasmid_probability_curve,
         dict(binary_df=binary_df, figure_dir=figure_dir)),
        ("prevalence scatter", plot_prevalence_scatter,
         dict(tier1=outputs.get("tier1"), figure_dir=figure_dir)),
        ("composition by burden", plot_composition_by_burden,
         dict(binary_df=binary_df, defense_cols=defense_cols, figure_dir=figure_dir)),
        ("burden by phylum", plot_burden_by_phylum,
         dict(binary_df=binary_df, figure_dir=figure_dir)),
        ("conditional plasmid rate", plot_conditional_plasmid_rate,
         dict(binary_df=binary_df, tier1=outputs.get("tier1"), figure_dir=figure_dir)),
        ("co-occurrence heatmap", plot_cooccurrence_heatmap,
         dict(binary_df=binary_df, defense_cols=defense_cols, figure_dir=figure_dir)),

        # --- Multivariate / stability ---
        ("multivariate stability", plot_multivariate_stability,
         dict(stability=outputs.get("mv_stability"), figure_dir=figure_dir)),

        # --- Takehome narrative ---
        ("gabija multi-method", plot_gabija_multi_method,
         dict(combined=outputs.get("combined"), figure_dir=figure_dir)),
        ("RM prevalence confounding", plot_rm_prevalence_confounding,
         dict(combined=outputs.get("combined"), figure_dir=figure_dir)),
        ("takehome fisher", plot_takehome_fisher,
         dict(tier1=outputs.get("tier1"), figure_dir=figure_dir)),
        ("takehome phyloglm", plot_takehome_phyloglm,
         dict(tier2_phyloglm=outputs.get("tier2_phyloglm"), figure_dir=figure_dir)),
        ("takehome multivariate", plot_takehome_multivariate,
         dict(pglmm=outputs.get("tier2_pglmm_mv"), figure_dir=figure_dir)),
        ("takehome lasso", plot_takehome_lasso,
         dict(lasso=outputs.get("lasso"), figure_dir=figure_dir)),
        ("takehome prevalence matching", plot_takehome_prevalence_matching,
         dict(prev_match=outputs.get("tier3_prevalence_matched"), figure_dir=figure_dir)),
        ("cross-method consistency", plot_cross_method_consistency,
         dict(tier2_phyloglm=outputs.get("tier2_phyloglm"),
              pglmm=outputs.get("tier2_pglmm_mv"), figure_dir=figure_dir)),
        ("effect attenuation", plot_effect_attenuation,
         dict(tier1=outputs.get("tier1"),
              tier2_phyloglm=outputs.get("tier2_phyloglm"), figure_dir=figure_dir)),

        # --- RF ---
        ("RF clade-blocked importance", plot_rf_importance,
         dict(rf_binary=outputs.get("rf_binary"),
              rf_prevalence=outputs.get("rf_prevalence"),
              figure_dir=figure_dir)),
        ("RF per-fold AUC", plot_rf_fold_auc,
         dict(fold_aucs=outputs.get("rf_fold_aucs"), figure_dir=figure_dir)),
    ]
    for name, fn, kw in jobs:
        try:
            fn(**kw)
            logger.info(f"  figure saved: {name}")
        except Exception as e:
            logger.warning(f"  figure '{name}' skipped: {e}")
