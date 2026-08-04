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

# ---- Response residuals from a phylogenetic intercept-only logistic model ----
# This fit was previously bare, so its failure took the whole script down and
# triggered the caller's "fall back to raw data" path -- meaning a LASSO
# advertised as running on phylogenetically-decorrelated features could in
# fact be running on raw presence/absence with nothing in the output to say so.
y <- data[[response]]
fit_y <- tryCatch(
  phylolm::phyloglm(y ~ 1,
                    data = data.frame(y = y, row.names = rownames(data)),
                    phy = tree, method = "logistic_MPLE", btol = 20),
  error = function(e) e
)
if (inherits(fit_y, "error")) {
  cat(file = stderr(), sprintf(
    "[phylo_residuals.R] response residualisation FAILED: %s\n",
    conditionMessage(fit_y)))
  resid_y <- as.numeric(scale(y)[, 1])
  response_decorrelated <- FALSE
} else {
  p_hat <- fit_y$fitted.values
  # Guard against fitted probabilities at 0/1, which make the Pearson
  # denominator zero and emit Inf residuals.
  p_hat <- pmin(pmax(p_hat, 1e-6), 1 - 1e-6)
  resid_y <- (y - p_hat) / sqrt(p_hat * (1 - p_hat))
  response_decorrelated <- TRUE
}

out <- data.frame(tip = rownames(data), response_residual = resid_y)
# Flag column so the caller can assert what it actually received instead of
# assuming. Read by tier2_multivariate._run_phylo_residuals.
out$response_decorrelated <- response_decorrelated

# ---- Predictor residuals via Brownian corStruct ----
# Per-predictor fallback is retained (one pathological column should not sink
# the stage) but is now RECORDED per predictor, in a companion status frame.
cor_struct <- ape::corBrownian(phy = tree, form = ~1)
status <- list()
for (p in predictors) {
  x <- data[[p]]
  # Centre on mean to avoid convergence issues when x is near-constant
  df_p <- data.frame(x = x - mean(x, na.rm = TRUE), row.names = rownames(data))
  fit_p <- tryCatch(
    nlme::gls(x ~ 1, data = df_p, correlation = cor_struct, method = "ML"),
    error = function(e) e
  )
  if (inherits(fit_p, "error")) {
    out[[paste0("predictor_", p)]] <- scale(x)[, 1]   # raw standardised
    status[[p]] <- data.frame(predictor = p, decorrelated = FALSE,
                              error = gsub("[\t\r\n]+", " ",
                                           conditionMessage(fit_p)),
                              stringsAsFactors = FALSE)
  } else {
    r <- residuals(fit_p)
    sdr <- sd(r)
    if (!is.finite(sdr) || sdr <= 0) {
      out[[paste0("predictor_", p)]] <- scale(x)[, 1]
      status[[p]] <- data.frame(predictor = p, decorrelated = FALSE,
                                error = "zero_residual_variance",
                                stringsAsFactors = FALSE)
    } else {
      out[[paste0("predictor_", p)]] <- (r - mean(r)) / sdr
      status[[p]] <- data.frame(predictor = p, decorrelated = TRUE,
                                error = NA_character_,
                                stringsAsFactors = FALSE)
    }
  }
}

write.table(out, out_path, sep = "\t", quote = FALSE, row.names = FALSE,
            na = "NA")

# Companion status table: <out_path>.status.tsv. The caller reads this and
# refuses to label the run "phylo-residualised" unless every predictor and the
# response actually were.
status_df <- do.call(rbind, status)
if (!is.null(status_df)) {
  status_df$response_decorrelated <- response_decorrelated
  write.table(status_df, paste0(out_path, ".status.tsv"), sep = "\t",
              quote = FALSE, row.names = FALSE, na = "NA")
  n_failed <- sum(!status_df$decorrelated)
  if (n_failed > 0) {
    cat(file = stderr(), sprintf(
      "[phylo_residuals.R] %d/%d predictors fell back to raw standardised values\n",
      n_failed, nrow(status_df)))
  }
}
