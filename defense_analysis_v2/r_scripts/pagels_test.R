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

tree <- ape::read.tree(tree_path)
data <- read.delim(data_path, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
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
    rows[[p]] <- data.frame(defense_system = p,
                            pagel_p_value = NA_real_,
                            pagel_delta_logL = NA_real_,
                            pagel_logL_indep = NA_real_,
                            pagel_logL_dep = NA_real_,
                            skip_reason = "low_count",
                            stringsAsFactors = FALSE)
    next
  }

  x_named <- setNames(as.character(x), rownames(data))
  y_named <- setNames(as.character(y), rownames(data))

  fit <- tryCatch(
    phytools::fitPagel(tree, x = x_named, y = y_named, method = "fitMk"),
    error = function(e) e
  )

  if (inherits(fit, "error")) {
    rows[[p]] <- data.frame(defense_system = p,
                            pagel_p_value = NA_real_,
                            pagel_delta_logL = NA_real_,
                            pagel_logL_indep = NA_real_,
                            pagel_logL_dep = NA_real_,
                            skip_reason = paste("fitPagel_error:", conditionMessage(fit)),
                            stringsAsFactors = FALSE)
    next
  }

  rows[[p]] <- data.frame(defense_system = p,
                          pagel_p_value = fit$P,
                          pagel_delta_logL = fit$dependent.logL - fit$independent.logL,
                          pagel_logL_indep = fit$independent.logL,
                          pagel_logL_dep = fit$dependent.logL,
                          skip_reason = NA_character_,
                          stringsAsFactors = FALSE)
}

out <- do.call(rbind, rows)
write.table(out, out_path, sep = "\t", quote = FALSE, row.names = FALSE)
