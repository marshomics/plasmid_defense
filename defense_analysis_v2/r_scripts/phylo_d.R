#!/usr/bin/env Rscript
# Fritz & Purvis D statistic of phylogenetic signal for each binary predictor
# and the response. D is the conventional signal metric for binary traits:
#
#   D  = 0   -> Brownian-motion expected clustering
#   D  = 1   -> random across tips
#   D  < 0   -> more clustered than Brownian
#   D  > 1   -> overdispersed (more scattered than random)
#
# P-values are reported against both null hypotheses (random, Brownian) from
# caper's 1000-iter permutation. These give methods-section-ready justification
# for using phylogenetic correction.
#
# Usage: Rscript phylo_d.R <tree.nwk> <data.tsv> <args.json> <out.tsv>

suppressPackageStartupMessages({
  required <- c("ape", "caper", "jsonlite")
  missing <- setdiff(required, rownames(installed.packages()))
  if (length(missing) > 0)
    stop("Missing R packages: ", paste(missing, collapse = ", "))
  invisible(lapply(required, library, character.only = TRUE))
})

args <- commandArgs(trailingOnly = TRUE)
tree_path <- args[1]; data_path <- args[2]; args_path <- args[3]; out_path <- args[4]

params <- jsonlite::fromJSON(args_path, simplifyVector = TRUE)
columns    <- params$columns        # character vector (response + predictors)
tip_column <- if (!is.null(params$tip_column)) params$tip_column else "tip"
n_perm     <- if (!is.null(params$n_perm)) params$n_perm else 1000
# `caper::phylo.d` is a permutation test; without a seed the reported
# p_random / p_brownian differ between runs of an otherwise identical
# pipeline. Seeded from the pipeline's global random_seed.
seed       <- if (!is.null(params$seed)) as.integer(params$seed) else 42L
set.seed(seed)

tree <- ape::read.tree(tree_path)
# caper::comparative.data refuses trees where any internal node label
# matches a tip label. GTDB trees often carry internal labels (named
# clades, bootstrap values) that happen to overlap tip labels in this
# dataset. The D-statistic doesn't use internal labels, so we drop them
# entirely. Other phylogenetic packages (phylolm, phytools) tolerate the
# overlap and so don't trigger this in other R scripts.
if (!is.null(tree$node.label)) {
  tree$node.label <- NULL
}
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
data <- data[data[[tip_column]] %in% tree$tip.label, , drop = FALSE]
tree <- ape::drop.tip(tree, setdiff(tree$tip.label, data[[tip_column]]))

# caper::comparative.data uses non-standard evaluation on `names.col` —
# passing `tip_column` directly would make caper look for a literal column
# named "tip_column". Use do.call with as.name() to construct the right
# symbol from the variable's value.
cdata <- do.call(caper::comparative.data,
                 list(phy = tree, data = data,
                      names.col = as.name(tip_column),
                      na.omit = FALSE, warn.dropped = FALSE))

rows <- list()
for (c in columns) {
  # Same NSE issue with caper::phylo.d's `binvar` arg. `!!` is rlang/
  # tidyverse syntax (not loaded here), not base R; use do.call + as.name.
  fit <- tryCatch(
    do.call(caper::phylo.d,
            list(data = cdata, binvar = as.name(c), permut = n_perm)),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    rows[[c]] <- data.frame(column = c, D = NA_real_,
                            p_random = NA_real_, p_brownian = NA_real_,
                            error = conditionMessage(fit),
                            stringsAsFactors = FALSE)
    next
  }
  rows[[c]] <- data.frame(
    column = c,
    D = fit$DEstimate,
    p_random = fit$Pval1,
    p_brownian = fit$Pval0,
    n_permutations = fit$Permutations,
    error = NA_character_,
    stringsAsFactors = FALSE
  )
}

out <- do.call(rbind, rows)
write.table(out, out_path, sep = "\t", quote = FALSE, row.names = FALSE)
