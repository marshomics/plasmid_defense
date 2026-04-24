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
# Normalise tip labels — collapse spaces to underscores so ape's
# unquoted-underscore-to-space conversion on read doesn't break the
# intersect. Bracket '[...]' annotations are left alone because in this
# dataset they are meaningful species identifiers (see phyloglm_uni.R).
normalise_tips <- function(s) {
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

if (outcome_mode == "binary") {
  fml_text <- sprintf("`%s` ~ %s + (1 | species__)", params$response, rhs)
  fml <- as.formula(fml_text)
  fit <- phyr::pglmm(fml, data = data, family = "binomial",
                     cov_ranef = list(species = tree),
                     REML = !use_bayes, bayes = use_bayes,
                     verbose = FALSE)
} else {
  # Binomial mode with successes / trials. phyr supports this via supplying
  # the trials column as a size argument.
  data$.k <- data[[params$response_k_column]]
  data$.n <- data[[params$response_n_column]]
  # Drop species with zero trials (fraction undefined)
  data <- data[data$.n > 0, , drop = FALSE]
  tree <- ape::drop.tip(tree, setdiff(tree$tip.label, rownames(data)))
  data <- data[tree$tip.label, , drop = FALSE]

  # phyr::pglmm takes a two-column response via cbind(successes, failures)
  data$.failures <- data$.n - data$.k
  fml_text <- sprintf("cbind(.k, .failures) ~ %s + (1 | species__)", rhs)
  fml <- as.formula(fml_text)
  fit <- phyr::pglmm(fml, data = data, family = "binomial",
                     cov_ranef = list(species = tree),
                     REML = !use_bayes, bayes = use_bayes,
                     verbose = FALSE)
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

re_var <- tryCatch(fit$s2r, error = function(e) NA_real_)
attr_phylo <- if (length(re_var) > 0) re_var[1] else NA_real_
out$pglmm_phylo_variance <- attr_phylo
out$n_species_fit <- nrow(data)
out$outcome_mode <- outcome_mode

# Record convergence information so downstream consumers can filter out fits
# that look successful at the summary level but didn't actually converge.
# phyr reports `convcode` (0 = converged for PQL-based fits) and we fall back
# to checking for any recorded warnings or non-finite standard errors.
conv_code <- tryCatch(as.integer(fit$convcode), error = function(e) NA_integer_)
if (length(conv_code) == 0 || !is.finite(conv_code)) conv_code <- NA_integer_
any_nonfinite_se <- any(!is.finite(coefs[, "Std.Error"]))
looks_degenerate <- !is.finite(attr_phylo) ||
                    (is.finite(attr_phylo) && attr_phylo <= 0) ||
                    any_nonfinite_se
out$pglmm_convcode       <- conv_code
out$pglmm_converged       <- is.finite(conv_code) && conv_code == 0 && !looks_degenerate
out$pglmm_fit_degenerate  <- looks_degenerate
out$pglmm_n_fixed_effects <- nrow(coefs)

write.table(out, out_path, sep = "\t", quote = FALSE, row.names = FALSE)
