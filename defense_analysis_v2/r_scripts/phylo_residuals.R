#!/usr/bin/env Rscript
# Generate phylogenetically-decorrelated features for downstream LASSO /
# Elastic Net regularisation.
#
# Approach (Butler & King 2004; Felsenstein 1985 independent contrasts, binary
# adaptation via phylolm residualisation):
#   - Fit a phylogenetic logistic regression of the response on an intercept
#     only (marginal mean accounting for tree structure), take the Pearson
#     residuals as a "phylogenetically-corrected" outcome.
#   - For each continuous/binary predictor, fit a phylogenetic linear model
#     (pgls via nlme::gls with corBrownian), take residuals as the
#     decorrelated predictor.
#   - Standardise residuals to unit variance before regularisation.
#
# This gives a standard Python-side LASSO / Elastic Net problem where the
# phylogeny has already been partialed out, so the regularisation path
# selects "extra" signal beyond tree structure.
#
# Usage: Rscript phylo_residuals.R <tree.nwk> <data.tsv> <args.json> <out.tsv>
#
# args.json keys:
#   response   : name of binary outcome column
#   predictors : character vector of predictor columns (numeric or binary)
#   tip_column : default "tip"

suppressPackageStartupMessages({
  required <- c("ape", "nlme", "phylolm", "jsonlite")
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

tree <- ape::read.tree(tree_path)
data <- read.delim(data_path, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
rownames(data) <- data[[tip_column]]
kept <- intersect(tree$tip.label, data[[tip_column]])
tree <- ape::drop.tip(tree, setdiff(tree$tip.label, kept))
data <- data[tree$tip.label, , drop = FALSE]

# Response residuals from a phylogenetic intercept-only logistic model
y <- data[[response]]
fit_y <- phylolm::phyloglm(y ~ 1, data = data.frame(y = y, row.names = rownames(data)),
                           phy = tree, method = "logistic_MPLE", btol = 20)
# phyloglm fitted probabilities, then Pearson residuals
p_hat <- fit_y$fitted.values
resid_y <- (y - p_hat) / sqrt(p_hat * (1 - p_hat))

out <- data.frame(tip = rownames(data), response_residual = resid_y)

# Predictor residuals via Brownian corStruct
cor_struct <- ape::corBrownian(phy = tree, form = ~1)
for (p in predictors) {
  x <- data[[p]]
  # Centre on mean to avoid convergence issues when x is near-constant
  df_p <- data.frame(x = x - mean(x, na.rm = TRUE), row.names = rownames(data))
  fit_p <- tryCatch(
    nlme::gls(x ~ 1, data = df_p, correlation = cor_struct, method = "ML"),
    error = function(e) e
  )
  if (inherits(fit_p, "error")) {
    out[[paste0("predictor_", p)]] <- scale(x)[, 1]   # fall back to raw standardised
  } else {
    r <- residuals(fit_p)
    r <- (r - mean(r)) / sd(r)
    out[[paste0("predictor_", p)]] <- r
  }
}

write.table(out, out_path, sep = "\t", quote = FALSE, row.names = FALSE)
