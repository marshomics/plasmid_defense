#!/usr/bin/env Rscript
# Multivariate phylogenetic generalised linear mixed model (PGLMM).
#
# This is the primary test for which defense systems predict a plasmid outcome
# *independently* of the others, while controlling for shared ancestry and
# (optionally) genome-scale covariates. Plain multivariate logistic regression
# ignores phylogeny and produces biased coefficients when the tree explains
# both outcome and predictor covariation.
#
# The fit uses phyr::pglmm with a binomial family and a phylogenetic random
# effect on the intercept via (1 | species__), where species__ is phyr's
# convention for a tip-level random effect with the supplied tree covariance.
#
# Two outcome modes:
#   * "binary" (default):  response is a 0/1 column.
#   * "binomial":          response is provided as two columns
#                          (response_k = successes, response_n = trials),
#                          passed via `response_k_column` and
#                          `response_n_column`. phyr uses cbind(k, n - k).
#
# Usage: Rscript pglmm_mv.R <tree.nwk> <data.tsv> <args.json> <out.tsv>
#
# args.json keys:
#   response            : name of binary 0/1 outcome column (binary mode)
#   response_k_column   : successes column (binomial mode)
#   response_n_column   : trials column (binomial mode)
#   outcome_mode        : "binary" | "binomial"
#   predictors          : character vector of predictor columns
#   covariates          : character vector of extra numeric covariates
#   interaction_pairs   : list of 2-element vectors, each a (A, B) pair to add as A:B
#   tip_column          : default "tip"
#   bayes               : if TRUE, use INLA; else REML/PQL (default FALSE)
#   center_covariates   : centre+scale covariates (default TRUE)
#
# Writes one row per fixed effect (intercept, each predictor, each covariate,
# each interaction).

suppressPackageStartupMessages({
  required <- c("ape", "phyr", "jsonlite")
  missing <- setdiff(required, rownames(installed.packages()))
  if (length(missing) > 0)
    stop("Missing R packages: ", paste(missing, collapse = ", "))
  invisible(lapply(required, library, character.only = TRUE))
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 4)
tree_path <- args[1]; data_path <- args[2]; args_path <- args[3]; out_path <- args[4]

params <- jsonlite::fromJSON(args_path, simplifyVector = TRUE)
predictors       <- params$predictors
covariates       <- if (!is.null(params$covariates)) params$covariates else character(0)
interaction_pairs <- if (!is.null(params$interaction_pairs)) params$interaction_pairs else list()
tip_column        <- if (!is.null(params$tip_column)) params$tip_column else "tip"
use_bayes         <- if (!is.null(params$bayes)) params$bayes else FALSE
outcome_mode      <- if (!is.null(params$outcome_mode)) params$outcome_mode else "binary"
center_covariates <- if (!is.null(params$center_covariates)) params$center_covariates else TRUE

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
# Normalise tip labels — collapse spaces to underscores so ape's
# unquoted-underscore-to-space conversion on read doesn't break the
# intersect. Bracket '[...]' annotations are left alone because in this
# dataset they are meaningful species identifiers (see phyloglm_uni.R).
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
if (length(kept) < 20) stop("Too few matched tips for PGLMM (", length(kept), ")")
tree <- ape::drop.tip(tree, setdiff(tree$tip.label, kept))
data <- data[tree$tip.label, , drop = FALSE]

# Numeric coercion + centering for covariates
if (length(covariates) > 0) {
  missing_cov <- setdiff(covariates, colnames(data))
  if (length(missing_cov) > 0)
    stop("Requested covariates missing from data: ", paste(missing_cov, collapse = ", "))
  for (c in covariates) {
    data[[c]] <- suppressWarnings(as.numeric(data[[c]]))
    if (isTRUE(center_covariates)) {
      v <- data[[c]]
      vf <- v[is.finite(v)]
      if (length(vf) > 1 && sd(vf) > 0) {
        data[[c]] <- (v - mean(vf, na.rm = TRUE)) / sd(vf, na.rm = TRUE)
      }
    }
  }
}

# Keep only complete cases across response, predictors, covariates
needed_cols <- c(predictors, covariates)
if (outcome_mode == "binary") {
  needed_cols <- c(needed_cols, params$response)
} else {
  needed_cols <- c(needed_cols, params$response_k_column, params$response_n_column)
}
complete_mask <- complete.cases(data[, needed_cols, drop = FALSE])
for (c in needed_cols) {
  v <- data[[c]]
  if (is.numeric(v)) complete_mask <- complete_mask & is.finite(v)
}
data <- data[complete_mask, , drop = FALSE]
tree <- ape::drop.tip(tree, setdiff(tree$tip.label, rownames(data)))
data <- data[tree$tip.label, , drop = FALSE]

# Build RHS
rhs_terms <- c(sprintf("`%s`", predictors))
if (length(covariates) > 0) {
  rhs_terms <- c(rhs_terms, sprintf("`%s`", covariates))
}
if (length(interaction_pairs) > 0) {
  for (pair in interaction_pairs) {
    if (length(pair) == 2) {
      rhs_terms <- c(rhs_terms,
                     sprintf("`%s`:`%s`", pair[[1]], pair[[2]]))
    }
  }
}
rhs <- paste(rhs_terms, collapse = " + ")

data$species <- rownames(data)

# Random-effect structure.
#
# `(1 | species__)` expands in phyr to TWO terms: a non-phylogenetic i.i.d.
# species effect plus the phylogenetic one. In BINARY mode the data are one
# row per tip, so the i.i.d. term has the identity design matrix -- the same
# as the observation-level random effect phyr adds for binomial families --
# on a Bernoulli likelihood that supplies no residual variance to separate
# them. The two are not identified, which is a plausible mechanism for the
# `Estimation of B failed` PQL boundary aborts recorded in
# docs/pglmm_step_recommendations.md.
#
# In binary mode we therefore request the phylogenetic term ONLY, and disable
# the observation-level random effect. In binomial mode each row has n > 1
# trials, so the i.i.d. term is identified and is kept as an overdispersion
# term.
ranef_binary  <- "(1 | species__)"
ranef_binomial <- "(1 | species__) + (1 | species)"

# `REML` and `bayes` are orthogonal: REML-vs-ML is a variance-estimation
# choice, Bayes-vs-frequentist an inference paradigm. They were previously
# tied as `REML = !use_bayes`, so every PQL binomial fit silently got
# REML = TRUE with no deliberation. Now explicit.
reml_flag <- if (!is.null(params$reml)) isTRUE(params$reml) else TRUE

fit_pglmm <- function(fml, dat, phy, add_obs_re) {
  # Wrap the fit. Previously bare: any phyr failure aborted the entire script,
  # so the outcome's row was simply absent from the output with no record that
  # a fit had even been attempted.
  warns <- character(0)
  res <- withCallingHandlers(
    tryCatch(
      phyr::pglmm(fml, data = dat, family = "binomial",
                  cov_ranef = list(species = phy),
                  REML = reml_flag, bayes = use_bayes,
                  add.obs.re = add_obs_re,
                  verbose = FALSE),
      error = function(e) e
    ),
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(fit = res, warnings = warns)
}

if (outcome_mode == "binary") {
  fml_text <- sprintf("`%s` ~ %s + %s", params$response, rhs, ranef_binary)
  fml <- as.formula(fml_text)
  fitres <- fit_pglmm(fml, data, tree, add_obs_re = FALSE)
} else {
  # Binomial mode with successes / trials, via cbind(successes, failures).
  data$.k <- data[[params$response_k_column]]
  data$.n <- data[[params$response_n_column]]
  # Drop species with zero trials (fraction undefined)
  data <- data[data$.n > 0, , drop = FALSE]
  tree <- ape::drop.tip(tree, setdiff(tree$tip.label, rownames(data)))
  data <- data[tree$tip.label, , drop = FALSE]
  data$species <- rownames(data)

  data$.failures <- data$.n - data$.k
  fml_text <- sprintf("cbind(.k, .failures) ~ %s + %s", rhs, ranef_binomial)
  fml <- as.formula(fml_text)
  fitres <- fit_pglmm(fml, data, tree, add_obs_re = FALSE)
}

fit <- fitres$fit
fit_warnings <- fitres$warnings

tsv_safe <- function(s) {
  if (is.null(s) || length(s) == 0 || all(is.na(s))) return(NA_character_)
  s <- gsub("[\t\r\n]+", " ", paste(as.character(s), collapse = "; "))
  substr(gsub('"', "'", s, fixed = TRUE), 1, 300)
}

# A failed fit writes a single explicit failure row and exits 0, so the caller
# records that the outcome was attempted and failed rather than seeing an
# empty frame indistinguishable from "stage not run".
if (inherits(fit, "error")) {
  out <- data.frame(
    term = NA_character_,
    pglmm_coefficient = NA_real_, pglmm_std_err = NA_real_,
    pglmm_z_value = NA_real_, pglmm_p_value = NA_real_,
    pglmm_phylo_variance = NA_real_,
    pglmm_iid_variance = NA_real_,
    n_species_fit = nrow(data),
    outcome_mode = outcome_mode,
    pglmm_convcode = NA_integer_,
    pglmm_converged = FALSE,
    pglmm_fit_degenerate = TRUE,
    pglmm_n_fixed_effects = NA_integer_,
    pglmm_formula = tsv_safe(fml_text),
    pglmm_warnings = tsv_safe(fit_warnings),
    pglmm_error = tsv_safe(conditionMessage(fit)),
    stringsAsFactors = FALSE
  )
  write.table(out, out_path, sep = "\t", quote = FALSE, row.names = FALSE,
              na = "NA")
  cat(file = stderr(), sprintf("[pglmm_mv.R] FIT FAILED (%s): %s\n",
                               outcome_mode, conditionMessage(fit)))
  quit(status = 0)
}

coefs <- summary(fit)$coefficients
out <- data.frame(
  term = rownames(coefs),
  pglmm_coefficient = coefs[, "Value"],
  pglmm_std_err     = coefs[, "Std.Error"],
  pglmm_z_value     = coefs[, "Zscore"],
  pglmm_p_value     = coefs[, "Pvalue"],
  stringsAsFactors = FALSE
)

# ---- Variance components, selected BY NAME ----
# `fit$s2r` is a NAMED vector with one entry per random-effect term. Taking
# `[1]` positionally returned the NON-phylogenetic i.i.d. component whenever
# `species__` expanded into two terms -- so `pglmm_phylo_variance` was
# reporting the wrong number, and that wrong number then drove the degeneracy
# flag below. Match on the `__` suffix that marks the phylogenetic term.
re_var <- tryCatch(fit$s2r, error = function(e) numeric(0))
if (is.null(re_var)) re_var <- numeric(0)
re_names <- names(re_var)
if (is.null(re_names)) re_names <- rep(NA_character_, length(re_var))

phylo_idx <- which(grepl("__", re_names, fixed = TRUE))
iid_idx   <- which(!grepl("__", re_names, fixed = TRUE))

attr_phylo <- if (length(phylo_idx) > 0) as.numeric(re_var[phylo_idx[1]]) else
              if (length(re_var) == 1) as.numeric(re_var[1]) else NA_real_
attr_iid   <- if (length(iid_idx) > 0) as.numeric(re_var[iid_idx[1]]) else NA_real_

if (length(re_var) > 0 && length(phylo_idx) == 0 && length(re_var) > 1) {
  cat(file = stderr(), paste0(
    "[pglmm_mv.R] WARNING: could not identify the phylogenetic variance ",
    "component by name (names: ", paste(re_names, collapse = ", "),
    "); pglmm_phylo_variance is NA rather than a guess.\n"))
}

out$pglmm_phylo_variance <- attr_phylo
out$pglmm_iid_variance   <- attr_iid
out$n_species_fit <- nrow(data)
out$outcome_mode <- outcome_mode

# Convergence. `fit$convcode` is not populated on every phyr path; note that
# `as.integer(NULL)` returns integer(0) rather than erroring, so the previous
# tryCatch never fired and conv_code silently became NA -- making
# `pglmm_converged` FALSE for every row regardless of the actual fit. Absent
# convcode is now treated as "unknown", not "failed", and the degeneracy
# verdict carries the decision.
conv_code <- tryCatch({
  cc <- fit$convcode
  if (is.null(cc) || length(cc) == 0) NA_integer_ else as.integer(cc)[1]
}, error = function(e) NA_integer_)

any_nonfinite_se <- any(!is.finite(coefs[, "Std.Error"]))
any_extreme_se   <- any(is.finite(coefs[, "Std.Error"]) &
                        coefs[, "Std.Error"] > 1e4)
warned_converge  <- any(grepl("converg|boundary|singular|failed|NaN",
                              fit_warnings, ignore.case = TRUE))
looks_degenerate <- !is.finite(attr_phylo) ||
                    (is.finite(attr_phylo) && attr_phylo <= 0) ||
                    any_nonfinite_se || any_extreme_se || warned_converge ||
                    (is.finite(conv_code) && conv_code != 0)

out$pglmm_convcode        <- conv_code
out$pglmm_converged       <- !looks_degenerate
out$pglmm_fit_degenerate  <- looks_degenerate
out$pglmm_n_fixed_effects <- nrow(coefs)
out$pglmm_formula         <- tsv_safe(fml_text)
out$pglmm_warnings        <- tsv_safe(fit_warnings)
out$pglmm_error           <- NA_character_

# Degenerate fits keep their coefficients for forensics but surrender their
# p-values, so they cannot enter FDR, consensus, or a figure.
if (looks_degenerate) {
  out$pglmm_p_value <- NA_real_
  cat(file = stderr(), sprintf(
    "[pglmm_mv.R] DEGENERATE fit (%s): phylo_var=%s nonfinite_se=%s warnings=%s\n",
    outcome_mode, format(attr_phylo), any_nonfinite_se,
    tsv_safe(fit_warnings)))
}

write.table(out, out_path, sep = "\t", quote = FALSE, row.names = FALSE,
            na = "NA")
