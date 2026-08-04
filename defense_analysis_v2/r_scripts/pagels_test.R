#!/usr/bin/env Rscript
# Pagel's test for correlated binary-trait evolution on a phylogeny.
#
# For each predictor, fits independent-evolution and correlated-evolution
# continuous-time Markov models and returns the likelihood-ratio p-value.
# No selection-inference filter on Tier 1 results (that would bias the null).
#
# Usage: Rscript pagels_test.R <tree.nwk> <data.tsv> <args.json> <out.tsv>

suppressPackageStartupMessages({
  required <- c("ape", "phytools", "jsonlite")
  missing <- setdiff(required, rownames(installed.packages()))
  if (length(missing) > 0)
    stop("Missing R packages: ", paste(missing, collapse = ", "))
  invisible(lapply(required, library, character.only = TRUE))
})

args <- commandArgs(trailingOnly = TRUE)
tree_path <- args[1]; data_path <- args[2]; args_path <- args[3]; out_path <- args[4]

params <- jsonlite::fromJSON(args_path, simplifyVector = TRUE)
response   <- params$response
predictors <- params$predictors
tip_column <- if (!is.null(params$tip_column)) params$tip_column else "tip"
max_species <- if (!is.null(params$max_species)) params$max_species else 1500
min_count   <- if (!is.null(params$min_count)) params$min_count else 5
seed        <- if (!is.null(params$seed)) params$seed else 42
# B2: also fit the two restricted dependent models so evolutionary ORDERING
# can be read off by AIC. Triples the per-system cost, so it is switchable.
fit_directional <- if (!is.null(params$fit_directional)) isTRUE(params$fit_directional) else FALSE
# Only fit the restricted dependent models where the standard test rejects
# independence. Direction is only meaningful given dependence, and this gate
# typically removes 80-90% of the systems -- turning a 3x cost increase into
# roughly 1.2x. The screen uses a LENIENT alpha so borderline-dependent systems
# are still characterised.
dir_screen_alpha <- if (!is.null(params$directional_screen_alpha)) as.numeric(params$directional_screen_alpha) else 0.10
dir_only_if_dep  <- if (!is.null(params$directional_only_if_dependent)) isTRUE(params$directional_only_if_dependent) else TRUE

# Sanitise strings destined for single-cell TSV output. R error messages
# routinely contain tabs and newlines, which would corrupt the frame the
# Python bridge parses with a fixed column count.
tsv_safe <- function(s) {
  if (is.null(s) || length(s) == 0 || all(is.na(s))) return(NA_character_)
  s <- gsub("[\t\r\n]+", " ", paste(as.character(s), collapse = "; "))
  substr(gsub('"', "'", s, fixed = TRUE), 1, 300)
}

# Every row must carry the same columns or do.call(rbind, ...) fails on a
# ragged list, so the directional columns are always present and simply NA
# when the directional fits were not requested or did not converge.
make_row <- function(p, pval, dlogl, ll_indep, ll_dep, reason) {
  data.frame(
    defense_system = p,
    pagel_p_value = pval,
    pagel_delta_logL = dlogl,
    pagel_logL_indep = ll_indep,
    pagel_logL_dep = ll_dep,
    pagel_logL_plasmid_drives_defense = NA_real_,
    pagel_logL_defense_drives_plasmid = NA_real_,
    pagel_aic_independent = NA_real_,
    pagel_aic_plasmid_drives_defense = NA_real_,
    pagel_aic_defense_drives_plasmid = NA_real_,
    pagel_aic_mutual = NA_real_,
    pagel_directional_error = NA_character_,
    skip_reason = tsv_safe(reason),
    stringsAsFactors = FALSE
  )
}

tree <- ape::read.tree(tree_path)
data <- read.delim(data_path, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)

# Shared-frame support: apply the optional row filter / column overrides that
# let one serialised frame serve many R calls. See _shared_data.R.
.script_dir <- tryCatch({
  ca <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", ca[grep("^--file=", ca)])
  if (length(f)) dirname(normalizePath(f[1])) else "."
}, error = function(e) ".")
if (file.exists(file.path(.script_dir, "_shared_data.R"))) {
  source(file.path(.script_dir, "_shared_data.R"))
  data <- apply_shared_filters(data, params)
}
# Normalise tip labels — collapse spaces to underscores. Brackets in
# labels are meaningful species identifiers here; do NOT strip them.
normalise_tips <- function(s) {
  s <- trimws(s)
  # Strip literal outer single quotes — see phyloglm_uni.R comment.
  s <- gsub("^'+|'+$", "", s)
  s <- trimws(s)
  gsub(" ", "_", s, fixed = TRUE)
}
tree$tip.label <- normalise_tips(tree$tip.label)
data[[tip_column]] <- normalise_tips(data[[tip_column]])
rownames(data) <- data[[tip_column]]
kept <- intersect(tree$tip.label, data[[tip_column]])
tree <- ape::drop.tip(tree, setdiff(tree$tip.label, kept))
data <- data[tree$tip.label, , drop = FALSE]

# Uniform subsample for computational feasibility. Uniform (not
# outcome-stratified) preserves the null distribution of Pagel's test.
if (length(tree$tip.label) > max_species) {
  set.seed(seed)
  kept_tips <- sample(tree$tip.label, max_species)
  tree <- ape::drop.tip(tree, setdiff(tree$tip.label, kept_tips))
  data <- data[tree$tip.label, , drop = FALSE]
}

rows <- list()
for (p in predictors) {
  x <- data[[p]]
  y <- data[[response]]
  n_xp <- sum(x == 1); n_xn <- sum(x == 0)
  n_yp <- sum(y == 1); n_yn <- sum(y == 0)
  if (n_xp < min_count || n_xn < min_count || n_yp < min_count || n_yn < min_count) {
    rows[[p]] <- make_row(p, NA_real_, NA_real_, NA_real_, NA_real_,
                          "low_count")
    next
  }

  x_named <- setNames(as.character(x), rownames(data))
  y_named <- setNames(as.character(y), rownames(data))

  # Per-system wall-clock safety net. fitPagel can occasionally thrash on
  # a hard-to-converge system (rare-feature * rare-outcome combination
  # producing a near-singular rate matrix); without this, one bad system
  # could consume the entire subsample budget while 434 others wait. 600
  # seconds is generous for typical 500-tip subsamples — bump it up if
  # you bump the subsample size.
  per_system_seconds <- 600
  setTimeLimit(elapsed = per_system_seconds, transient = TRUE)
  fit <- tryCatch(
    phytools::fitPagel(tree, x = x_named, y = y_named, method = "fitMk"),
    error = function(e) e
  )
  setTimeLimit(elapsed = Inf, transient = TRUE)

  if (inherits(fit, "error")) {
    rows[[p]] <- make_row(p, NA_real_, NA_real_, NA_real_, NA_real_,
                          paste("fitPagel_error:", conditionMessage(fit)))
    next
  }

  row <- make_row(p, fit$P,
                  fit$dependent.logL - fit$independent.logL,
                  fit$independent.logL, fit$dependent.logL,
                  NA_character_)

  # ------------------------------------------------------------------
  # B2 — DIRECTIONALITY.
  #
  # fitPagel's `dep.var` selects which character's transition rates are
  # allowed to depend on the other's state:
  #
  #   dep.var = "x"   rates of change in X depend on the state of Y
  #   dep.var = "y"   rates of change in Y depend on the state of X
  #   dep.var = "xy"  both (the default, fitted above)
  #
  # In THIS script x = the DEFENSE SYSTEM and y = the PLASMID outcome
  # (assigned at the top of the loop: x <- data[[p]], y <- data[[response]]).
  # Therefore:
  #
  #   dep.var = "x"  ->  defense transitions depend on plasmid state
  #                  ->  "PLASMID CARRIAGE DRIVES DEFENSE gain/loss"
  #   dep.var = "y"  ->  plasmid transitions depend on defense state
  #                  ->  "DEFENSE STATE DRIVES PLASMID gain/loss"
  #
  # Getting this mapping backwards would invert the paper's conclusion, so
  # the output column names below state the biology, not the letter.
  #
  # Parameter counts for AIC (Mk rate matrices over the 4 joint states):
  #   independent      4   (2 rates per character, no dependence)
  #   dep.var = x/y    6   (dependent character gets 2 rates per state of
  #                         the other; independent character keeps 2)
  #   dep.var = xy     8   (both characters get 2 rates per partner state)
  run_dir <- isTRUE(fit_directional) &&
             (!dir_only_if_dep ||
              (is.finite(fit$P) && fit$P < dir_screen_alpha))
  if (run_dir) {
    k_indep <- 4; k_one <- 6; k_both <- 8
    aic <- function(ll, k) if (is.finite(ll)) (-2 * ll + 2 * k) else NA_real_

    setTimeLimit(elapsed = per_system_seconds, transient = TRUE)
    fit_x <- tryCatch(
      phytools::fitPagel(tree, x = x_named, y = y_named,
                         dep.var = "x", method = "fitMk"),
      error = function(e) e)
    setTimeLimit(elapsed = Inf, transient = TRUE)

    setTimeLimit(elapsed = per_system_seconds, transient = TRUE)
    fit_y <- tryCatch(
      phytools::fitPagel(tree, x = x_named, y = y_named,
                         dep.var = "y", method = "fitMk"),
      error = function(e) e)
    setTimeLimit(elapsed = Inf, transient = TRUE)

    ll_plasmid_drives_defense <- if (inherits(fit_x, "error")) NA_real_
                                 else fit_x$dependent.logL
    ll_defense_drives_plasmid <- if (inherits(fit_y, "error")) NA_real_
                                 else fit_y$dependent.logL

    row$pagel_logL_plasmid_drives_defense <- ll_plasmid_drives_defense
    row$pagel_logL_defense_drives_plasmid <- ll_defense_drives_plasmid
    row$pagel_aic_independent  <- aic(fit$independent.logL, k_indep)
    row$pagel_aic_plasmid_drives_defense <- aic(ll_plasmid_drives_defense, k_one)
    row$pagel_aic_defense_drives_plasmid <- aic(ll_defense_drives_plasmid, k_one)
    row$pagel_aic_mutual <- aic(fit$dependent.logL, k_both)
    row$pagel_directional_error <- NA_character_
    if (inherits(fit_x, "error") || inherits(fit_y, "error")) {
      msgs <- c(if (inherits(fit_x, "error")) paste("dep_x:", conditionMessage(fit_x)),
                if (inherits(fit_y, "error")) paste("dep_y:", conditionMessage(fit_y)))
      row$pagel_directional_error <- tsv_safe(paste(msgs, collapse = "; "))
    }
  }

  rows[[p]] <- row
}

out <- do.call(rbind, rows)
write.table(out, out_path, sep = "\t", quote = FALSE, row.names = FALSE,
            na = "NA")
