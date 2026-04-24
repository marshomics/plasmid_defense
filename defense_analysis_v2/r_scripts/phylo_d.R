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

tree <- ape::read.tree(tree_path)
data <- read.delim(data_path, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
# Normalise tip labels — see phyloglm_uni.R comment.
tree$tip.label <- gsub(" ", "_", tree$tip.label, fixed = TRUE)
data[[tip_column]] <- gsub(" ", "_", data[[tip_column]], fixed = TRUE)
data <- data[data[[tip_column]] %in% tree$tip.label, , drop = FALSE]
tree <- ape::drop.tip(tree, setdiff(tree$tip.label, data[[tip_column]]))

cdata <- caper::comparative.data(phy = tree, data = data, names.col = tip_column,
                                 na.omit = FALSE, warn.dropped = FALSE)

rows <- list()
for (c in columns) {
  fit <- tryCatch(
    caper::phylo.d(cdata, binvar = !!as.name(c), permut = n_perm),
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
