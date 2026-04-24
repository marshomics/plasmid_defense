#!/usr/bin/env Rscript
# Phylogenetic Generalised Least Squares (PGLS) on defense-system count
# (burden) as a function of a plasmid-outcome predictor, with optional
# covariates.
#
# Tests whether plasmid-carrying species have a different total number of
# defense systems than non-carriers, after accounting for shared ancestry
# (Pagel's lambda estimated by ML) and genome-scale confounders.
#
# Usage: Rscript pgls_burden.R <tree.nwk> <data.tsv> <args.json> <out.tsv>
#
# args.json keys:
#   response          : name of the numeric burden column (e.g. "defense_burden_count")
#   predictor         : name of the (binary or numeric) plasmid predictor
#   covariates        : optional character vector of numeric covariates
#   tip_column        : default "tip"
#   center_covariates : centre + scale covariates (default TRUE)

suppressPackageStartupMessages({
  required <- c("ape", "nlme", "jsonlite")
  missing <- setdiff(required, rownames(installed.packages()))
  if (length(missing) > 0)
    stop("Missing R packages: ", paste(missing, collapse = ", "))
  invisible(lapply(required, library, character.only = TRUE))
})

args <- commandArgs(trailingOnly = TRUE)
tree_path <- args[1]; data_path <- args[2]; args_path <- args[3]; out_path <- args[4]

params <- jsonlite::fromJSON(args_path, simplifyVector = TRUE)
response          <- params$response
predictor         <- params$predictor
covariates        <- if (!is.null(params$covariates)) params$covariates else character(0)
tip_column        <- if (!is.null(params$tip_column)) params$tip_column else "tip"
center_covariates <- if (!is.null(params$center_covariates)) params$center_covariates else TRUE

tree <- ape::read.tree(tree_path)
data <- read.delim(data_path, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
# Normalise tip labels — collapse spaces to underscores. Brackets in
# labels are meaningful species identifiers here; do NOT strip them.
normalise_tips <- function(s) {
  s <- trimws(s)
  gsub(" ", "_", s, fixed = TRUE)
}
tree$tip.label <- normalise_tips(tree$tip.label)
data[[tip_column]] <- normalise_tips(data[[tip_column]])
rownames(data) <- data[[tip_column]]

kept <- intersect(tree$tip.label, data[[tip_column]])
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

# Complete cases across all columns used in the fit
needed <- c(response, predictor, covariates)
complete_mask <- complete.cases(data[, needed, drop = FALSE])
for (c in needed) {
  v <- data[[c]]
  if (is.numeric(v)) complete_mask <- complete_mask & is.finite(v)
}
data <- data[complete_mask, , drop = FALSE]
tree <- ape::drop.tip(tree, setdiff(tree$tip.label, rownames(data)))
data <- data[tree$tip.label, , drop = FALSE]

rhs <- c(predictor, covariates)
fml <- as.formula(paste0("`", response, "` ~ ",
                         paste(sprintf("`%s`", rhs), collapse = " + ")))

cor_struct <- ape::corPagel(0.5, phy = tree, form = ~1, fixed = FALSE)
fit <- tryCatch(
  nlme::gls(fml, data = data, correlation = cor_struct, method = "ML",
            control = nlme::glsControl(opt = "optim", msMaxIter = 200)),
  error = function(e) e
)

out <- data.frame(
  analysis = "pgls_burden",
  n_species = nrow(data),
  n_covariates_used = length(covariates),
  stringsAsFactors = FALSE
)

if (inherits(fit, "error")) {
  out$pgls_coefficient    <- NA_real_
  out$pgls_std_err        <- NA_real_
  out$pgls_t_value        <- NA_real_
  out$pgls_p_value        <- NA_real_
  out$pagel_lambda        <- NA_real_
  out$error               <- conditionMessage(fit)
  write.table(out, out_path, sep = "\t", quote = FALSE, row.names = FALSE)
  quit(status = 0)
}

s <- summary(fit)
tt <- s$tTable
# The predictor is always the first non-intercept row (index 2); use the name
# match so the row index is unambiguous under back-tick quoting.
predictor_row <- grep(predictor, rownames(tt), fixed = TRUE)
row_idx <- if (length(predictor_row) > 0) predictor_row[1] else 2
out$pgls_coefficient <- tt[row_idx, "Value"]
out$pgls_std_err     <- tt[row_idx, "Std.Error"]
out$pgls_t_value     <- tt[row_idx, "t-value"]
out$pgls_p_value     <- tt[row_idx, "p-value"]
out$pagel_lambda     <- coef(fit$modelStruct$corStruct, unconstrained = FALSE)[1]
out$residual_std_err <- s$sigma
out$logLik           <- as.numeric(logLik(fit))
out$error            <- NA_character_

write.table(out, out_path, sep = "\t", quote = FALSE, row.names = FALSE)
