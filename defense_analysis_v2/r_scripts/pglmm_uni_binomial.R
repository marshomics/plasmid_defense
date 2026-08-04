#!/usr/bin/env Rscript
# Univariate binomial PGLMM, iterating every predictor INSIDE one R process.
#
# Why this script exists
# ----------------------
# The entry-mode composition model needs a separate univariate fit per defense
# system. Calling `pglmm_mv.R` once per system meant ~435 R invocations, each
# paying interpreter start-up, package loading, tree parsing and data parsing
# before fitting anything. Those fixed costs dominate: the model itself runs on
# a few thousand tips (only species with enough plasmids to have a composition),
# so the overhead was several times the useful work.
#
# Iterating internally amortises all of it across the whole sweep -- the same
# structure `phyloglm_uni.R` already uses -- turning 435 invocations into 1.
#
# Usage: Rscript pglmm_uni_binomial.R <tree.nwk> <data.tsv> <args.json> <out.tsv>
#
# args.json keys:
#   predictors           : character vector, one univariate fit each
#   covariates           : character vector, in every fit
#   response_k_column    : successes  (e.g. n_plasmid_nonconjugative)
#   response_n_column    : trials     (e.g. n_plasmids_entrymode)
#   tip_column           : default "tip"
#   min_count            : minimum presence AND absence count per predictor
#   per_fit_seconds      : per-system wall-clock safety net (default 900)
#   reml                 : default TRUE

suppressPackageStartupMessages({
  required <- c("ape", "phyr", "jsonlite")
  missing <- setdiff(required, rownames(installed.packages()))
  if (length(missing) > 0)
    stop("Missing R packages: ", paste(missing, collapse = ", "))
  invisible(lapply(required, library, character.only = TRUE))
})

args <- commandArgs(trailingOnly = TRUE)
tree_path <- args[1]; data_path <- args[2]; args_path <- args[3]; out_path <- args[4]

params <- jsonlite::fromJSON(args_path, simplifyVector = TRUE)
predictors <- params$predictors
covariates <- if (!is.null(params$covariates)) params$covariates else character(0)
k_col      <- params$response_k_column
n_col      <- params$response_n_column
tip_column <- if (!is.null(params$tip_column)) params$tip_column else "tip"
min_count  <- if (!is.null(params$min_count)) params$min_count else 10
per_fit_seconds <- if (!is.null(params$per_fit_seconds)) params$per_fit_seconds else 900
reml_flag  <- if (!is.null(params$reml)) isTRUE(params$reml) else TRUE

tsv_safe <- function(s) {
  if (is.null(s) || length(s) == 0 || all(is.na(s))) return(NA_character_)
  s <- gsub("[\t\r\n]+", " ", paste(as.character(s), collapse = "; "))
  substr(gsub('"', "'", s, fixed = TRUE), 1, 300)
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

normalise_tips <- function(s) {
  s <- trimws(s); s <- gsub("^'+|'+$", "", s); s <- trimws(s)
  gsub(" ", "_", s, fixed = TRUE)
}
tree$tip.label <- normalise_tips(tree$tip.label)
data[[tip_column]] <- normalise_tips(data[[tip_column]])
data <- data[!duplicated(data[[tip_column]]), , drop = FALSE]
rownames(data) <- data[[tip_column]]

# Drop species with no trials before pruning: a species with zero plasmids has
# no composition and contributes nothing but a zero row.
data$.k <- suppressWarnings(as.numeric(data[[k_col]]))
data$.n <- suppressWarnings(as.numeric(data[[n_col]]))
data <- data[is.finite(data$.n) & data$.n > 0 & is.finite(data$.k), , drop = FALSE]
data$.failures <- data$.n - data$.k

kept <- intersect(tree$tip.label, rownames(data))
if (length(kept) < 20) stop("Too few matched tips (", length(kept), ")")
tree <- ape::drop.tip(tree, setdiff(tree$tip.label, kept))
data <- data[tree$tip.label, , drop = FALSE]
data$species <- rownames(data)

cat(file = stderr(), sprintf(
  "[pglmm_uni_binomial.R] %d tips with >0 trials; %d predictors; %d covariates\n",
  nrow(data), length(predictors), length(covariates)))

# Centre and scale covariates once for the whole sweep.
if (length(covariates) > 0) {
  covariates <- intersect(covariates, colnames(data))
  for (cc in covariates) {
    v <- suppressWarnings(as.numeric(data[[cc]]))
    vf <- v[is.finite(v)]
    if (length(vf) > 1 && sd(vf) > 0) v <- (v - mean(vf)) / sd(vf)
    data[[cc]] <- v
  }
}

make_row <- function(nm, reason) {
  data.frame(
    defense_system = nm, n_species_fit = nrow(data),
    pglmm_coefficient = NA_real_, pglmm_std_err = NA_real_,
    pglmm_z_value = NA_real_, pglmm_p_value = NA_real_,
    pglmm_phylo_variance = NA_real_, pglmm_iid_variance = NA_real_,
    pglmm_converged = NA, pglmm_fit_degenerate = NA,
    skip_reason = tsv_safe(reason), stringsAsFactors = FALSE
  )
}

cov_rhs <- if (length(covariates) > 0)
  paste0(" + ", paste(sprintf("`%s`", covariates), collapse = " + ")) else ""

rows <- list()
for (nm in predictors) {
  if (!(nm %in% colnames(data))) {
    rows[[nm]] <- make_row(nm, "column_missing"); next
  }
  x <- suppressWarnings(as.numeric(data[[nm]]))
  if (!all(is.finite(x))) { rows[[nm]] <- make_row(nm, "nonfinite_predictor"); next }
  is_bin <- all(x %in% c(0, 1))
  if (is_bin && (sum(x == 1) < min_count || sum(x == 0) < min_count)) {
    rows[[nm]] <- make_row(nm, "low_count_defense"); next
  }
  if (!is_bin && sd(x) <= 0) { rows[[nm]] <- make_row(nm, "predictor_no_variance"); next }

  fml <- as.formula(sprintf("cbind(.k, .failures) ~ `%s`%s + (1 | species__) + (1 | species)",
                            nm, cov_rhs))
  warns <- character(0)
  setTimeLimit(elapsed = per_fit_seconds, transient = TRUE)
  fit <- withCallingHandlers(
    tryCatch(
      phyr::pglmm(fml, data = data, family = "binomial",
                  cov_ranef = list(species = tree),
                  REML = reml_flag, bayes = FALSE,
                  add.obs.re = FALSE, verbose = FALSE),
      error = function(e) e),
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") })
  setTimeLimit(elapsed = Inf, transient = TRUE)

  if (inherits(fit, "error")) {
    rows[[nm]] <- make_row(nm, paste("pglmm_error:", conditionMessage(fit))); next
  }

  coefs <- tryCatch(summary(fit)$coefficients, error = function(e) NULL)
  if (is.null(coefs)) { rows[[nm]] <- make_row(nm, "summary_unavailable"); next }
  ridx <- grep(nm, gsub("`", "", rownames(coefs)), fixed = TRUE)
  if (length(ridx) == 0) { rows[[nm]] <- make_row(nm, "coefficient_row_absent"); next }
  ridx <- ridx[1]

  # Variance components selected BY NAME: the `__` suffix marks the
  # phylogenetic term. Taking [1] positionally returns the i.i.d. component
  # whenever species__ expands into two terms.
  re_var <- tryCatch(fit$s2r, error = function(e) numeric(0))
  if (is.null(re_var)) re_var <- numeric(0)
  rn <- names(re_var); if (is.null(rn)) rn <- rep(NA_character_, length(re_var))
  pi <- which(grepl("__", rn, fixed = TRUE))
  ii <- which(!grepl("__", rn, fixed = TRUE))
  v_phylo <- if (length(pi)) as.numeric(re_var[pi[1]]) else
             if (length(re_var) == 1) as.numeric(re_var[1]) else NA_real_
  v_iid <- if (length(ii)) as.numeric(re_var[ii[1]]) else NA_real_

  se <- coefs[ridx, "Std.Error"]
  degenerate <- !is.finite(v_phylo) || (is.finite(v_phylo) && v_phylo <= 0) ||
                !is.finite(se) || se <= 0 || se > 1e4 ||
                any(grepl("converg|boundary|singular|failed|NaN", warns,
                          ignore.case = TRUE))

  rows[[nm]] <- data.frame(
    defense_system = nm, n_species_fit = nrow(data),
    pglmm_coefficient = coefs[ridx, "Value"],
    pglmm_std_err = se,
    pglmm_z_value = coefs[ridx, "Zscore"],
    # Degenerate fits forfeit the p-value so they cannot enter FDR downstream.
    pglmm_p_value = if (degenerate) NA_real_ else coefs[ridx, "Pvalue"],
    pglmm_phylo_variance = v_phylo,
    pglmm_iid_variance = v_iid,
    pglmm_converged = !degenerate,
    pglmm_fit_degenerate = degenerate,
    skip_reason = if (degenerate) "degenerate_fit" else NA_character_,
    stringsAsFactors = FALSE
  )
}

out <- do.call(rbind, rows)
n_ok <- sum(is.na(out$skip_reason))
cat(file = stderr(), sprintf(
  "[pglmm_uni_binomial.R] %d/%d systems fit cleanly\n", n_ok, nrow(out)))
write.table(out, out_path, sep = "\t", quote = FALSE, row.names = FALSE,
            na = "NA")
