#!/usr/bin/env Rscript
# Univariate phylogenetic logistic regression, with optional covariates and
# support for two iteration modes (forward and reverse directional tests).
#
# Mode "predictor" (default):
#   response = single column name (e.g. "any_plasmid_conjugative")
#   predictors = list of columns (each tested in a separate fit as x)
#   for each p in predictors: fit response ~ p + covariates, report p's
#   coefficient.
#
# Mode "response":
#   response = list of column names (each tested in a separate fit as y)
#   predictors = single-element list (the fixed predictor, e.g. a plasmid
#   outcome)
#   for each r in responses: fit r ~ predictor + covariates, report predictor's
#   coefficient.
#
# Output column `test_label` holds whichever varied (response in reverse mode,
# predictor in forward mode). The caller uses this to index back to the
# defense system whose signal each row represents.
#
# Usage: Rscript phyloglm_uni.R <tree.nwk> <data.tsv> <args.json> <out.tsv>
#
# args.json keys:
#   response           : outcome column (forward mode, string) OR list (reverse mode)
#   predictors         : list of predictor columns (forward) OR single (reverse)
#   mode               : "predictor" | "response"  (default "predictor")
#   covariates         : character vector of numeric covariates (optional)
#   tip_column         : default "tip"
#   evolutionary_model : "BM" | "BM_penalized" | "OUfixedRoot" | ... (default "BM")
#   btol               : binomial-tolerance parameter for phyloglm (default 10)
#   boot               : number of parametric-bootstrap replicates for SE/CI (default 0)
#   min_count          : minimum presence AND absence count per binary column
#   center_covariates  : centre + scale covariates (default TRUE)

suppressPackageStartupMessages({
  required <- c("ape", "phylolm", "jsonlite")
  missing <- setdiff(required, rownames(installed.packages()))
  if (length(missing) > 0)
    stop("Missing R packages: ", paste(missing, collapse = ", "),
         ". Install with install.packages(c(", paste(sprintf('"%s"', missing), collapse = ", "), "))")
  invisible(lapply(required, library, character.only = TRUE))
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 4)
tree_path <- args[1]; data_path <- args[2]; args_path <- args[3]; out_path <- args[4]

params <- jsonlite::fromJSON(args_path, simplifyVector = TRUE)
mode              <- if (!is.null(params$mode)) params$mode else "predictor"
covariates        <- if (!is.null(params$covariates)) params$covariates else character(0)
tip_column        <- if (!is.null(params$tip_column)) params$tip_column else "tip"
evol_model        <- if (!is.null(params$evolutionary_model)) params$evolutionary_model else "BM"
btol              <- if (!is.null(params$btol)) params$btol else 10
boot_n            <- if (!is.null(params$boot)) params$boot else 0
min_count         <- if (!is.null(params$min_count)) params$min_count else 5
center_covariates <- if (!is.null(params$center_covariates)) params$center_covariates else TRUE

phyloglm_method <- switch(evol_model,
  BM           = "logistic_MPLE",
  BM_penalized = "logistic_IG10",
  OUfixedRoot  = "logistic_MPLE",
  "logistic_MPLE"
)

tree <- ape::read.tree(tree_path)
data <- read.delim(data_path, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)

# ---- Pre-normalisation diagnostic (always printed) ----
# Use cat(file=stderr()) rather than message() so this is visible in
# every call chain (subprocess capture, RStudio, interactive). Dump the
# raw shapes and a small sample BEFORE any normalisation so we can see
# exactly what ape and read.delim are handing us.
cat(file = stderr(),
    sprintf("[phyloglm_uni.R] tree_path=%s data_path=%s tip_column=%s\n",
            tree_path, data_path, tip_column),
    sprintf("[phyloglm_uni.R] length(tree$tip.label)=%d nrow(data)=%d\n",
            length(tree$tip.label), nrow(data)),
    sprintf("[phyloglm_uni.R] data columns (first 8): %s\n",
            paste(head(colnames(data), 8), collapse = " | ")),
    sprintf("[phyloglm_uni.R] tree tips RAW (first 5): %s\n",
            paste(head(tree$tip.label, 5), collapse = " | ")),
    sprintf("[phyloglm_uni.R] data[[tip]] RAW (first 5): %s\n",
            if (tip_column %in% colnames(data))
              paste(head(data[[tip_column]], 5), collapse = " | ")
            else "<MISSING COLUMN>")
)

# Normalise tip labels so the intersect is robust to ape's standard
# unquoted-underscore-to-space conversion on read. Force both sides to
# underscore form; this works regardless of whether dendropy wrote the
# label quoted (spaces preserved) or unquoted (underscores get converted
# to spaces by ape and then back to underscores by this gsub).
#
# IMPORTANT: do NOT strip '[species NNN]'-style bracket annotations here.
# In this dataset those brackets are meaningful identifiers that
# distinguish otherwise-identical species names (e.g. 's__foo [species
# 1]' vs 's__foo [species 2]'); stripping them would collapse distinct
# species to the same row key.
normalise_tips <- function(s) {
  s <- trimws(s)
  # Strip literal outer single quotes. Newick quote delimiters SHOULD be
  # stripped by ape::read.tree(), but some dendropy-write + ape-read
  # combinations leave them in the label string (seen on GTDB species-
  # level trees with bracketed annotations). Remove any leading/trailing
  # single quotes here so the intersect with the data TSV works either way.
  s <- gsub("^'+|'+$", "", s)
  s <- trimws(s)
  gsub(" ", "_", s, fixed = TRUE)
}
tree$tip.label <- normalise_tips(tree$tip.label)
if (tip_column %in% colnames(data)) {
  data[[tip_column]] <- normalise_tips(data[[tip_column]])
}

cat(file = stderr(),
    sprintf("[phyloglm_uni.R] tree tips NORMALISED (first 5): %s\n",
            paste(head(tree$tip.label, 5), collapse = " | ")),
    sprintf("[phyloglm_uni.R] data[[tip]] NORMALISED (first 5): %s\n",
            if (tip_column %in% colnames(data))
              paste(head(data[[tip_column]], 5), collapse = " | ")
            else "<MISSING COLUMN>")
)

# Duplicate-safety: if the data has multiple rows mapping to the same
# tip label (shouldn't happen after upstream species aggregation, but
# guard against it), keep the first and warn rather than erroring.
dup_tip_count <- sum(duplicated(data[[tip_column]]))
if (dup_tip_count > 0) {
  cat(file = stderr(),
      sprintf("[phyloglm_uni.R] WARNING: %d duplicate tip values in data; keeping first row per tip.\n",
              dup_tip_count))
  data <- data[!duplicated(data[[tip_column]]), , drop = FALSE]
}
rownames(data) <- data[[tip_column]]
kept <- intersect(tree$tip.label, data[[tip_column]])
cat(file = stderr(),
    sprintf("[phyloglm_uni.R] intersect: %d tips matched\n", length(kept))
)
if (length(kept) < 10) {
  # Show a few tree tips that DON'T appear in the data, so the mismatch
  # pattern is obvious (underscore vs space, missing prefix, etc.).
  tree_not_in_data <- setdiff(head(tree$tip.label, 20), data[[tip_column]])
  data_not_in_tree <- setdiff(head(data[[tip_column]], 20),
                               tree$tip.label)
  cat(file = stderr(),
      sprintf("[phyloglm_uni.R] tree tips NOT in data (first 5): %s\n",
              paste(head(tree_not_in_data, 5), collapse = " | ")),
      sprintf("[phyloglm_uni.R] data tips NOT in tree (first 5): %s\n",
              paste(head(data_not_in_tree, 5), collapse = " | "))
  )
  stop("Too few matched tips (", length(kept), ")")
}
tree <- ape::drop.tip(tree, setdiff(tree$tip.label, kept))
data <- data[tree$tip.label, , drop = FALSE]

# Numeric coercion + centering for covariates (done once on the whole frame)
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

# Resolve forward/reverse mode into (iter_names, resp_fn, pred_fn):
# iter_names is the list of column names iterated over; for each name,
# resp_fn gives the response column and pred_fn gives the predictor column.
if (mode == "predictor") {
  response   <- params$response      # single
  predictors <- params$predictors    # list
  stopifnot(length(response) == 1)
  iter_names <- as.character(predictors)
  resp_col_for <- function(nm) response
  pred_col_for <- function(nm) nm
} else if (mode == "response") {
  predictors <- params$predictors    # single
  responses  <- params$response      # list
  stopifnot(length(predictors) == 1)
  iter_names <- as.character(responses)
  resp_col_for <- function(nm) nm
  pred_col_for <- function(nm) predictors[1]
} else {
  stop("Unknown mode: ", mode, " (expected 'predictor' or 'response')")
}

make_skip_row <- function(nm, n, n_pos, n_neg, reason) {
  data.frame(
    test_label = nm, n_species = n,
    n_predictor_present = n_pos, n_predictor_absent = n_neg,
    phyloglm_coefficient = NA_real_, phyloglm_std_err = NA_real_,
    phyloglm_z_value = NA_real_, phyloglm_p_value = NA_real_,
    phyloglm_alpha = NA_real_,
    phyloglm_method = phyloglm_method,
    n_covariates_used = length(covariates),
    mode = mode,
    skip_reason = reason,
    stringsAsFactors = FALSE
  )
}

results <- list()
for (nm in iter_names) {
  rcol <- resp_col_for(nm)
  pcol <- pred_col_for(nm)
  if (!(rcol %in% colnames(data)) || !(pcol %in% colnames(data))) {
    results[[nm]] <- make_skip_row(nm, nrow(data), NA, NA, "column_missing")
    next
  }
  y <- data[[rcol]]
  x <- data[[pcol]]

  fit_frame <- data.frame(y = y, x = x, row.names = rownames(data))
  if (length(covariates) > 0) {
    fit_frame <- cbind(fit_frame, data[, covariates, drop = FALSE])
  }
  finite_mask <- complete.cases(fit_frame) & is.finite(fit_frame$y) &
                 is.finite(fit_frame$x)
  if (length(covariates) > 0) {
    for (c in covariates) finite_mask <- finite_mask & is.finite(fit_frame[[c]])
  }
  fit_frame <- fit_frame[finite_mask, , drop = FALSE]
  tree_this <- ape::drop.tip(tree, setdiff(tree$tip.label, rownames(fit_frame)))
  fit_frame <- fit_frame[tree_this$tip.label, , drop = FALSE]

  is_binary_pred <- all(fit_frame$x %in% c(0, 1, 0L, 1L))
  n_pos <- if (is_binary_pred) sum(fit_frame$x == 1) else NA_integer_
  n_neg <- if (is_binary_pred) sum(fit_frame$x == 0) else NA_integer_

  # phyloglm requires binary response. Also gate on min_count of both
  # presence/absence for whichever side is binary. Response is binary by
  # construction (logistic); predictor binary check gates the min_count rule.
  if (!all(fit_frame$y %in% c(0, 1, 0L, 1L))) {
    results[[nm]] <- make_skip_row(nm, nrow(fit_frame), n_pos, n_neg,
                                   "response_not_binary")
    next
  }
  if (is_binary_pred && (n_pos < min_count || n_neg < min_count)) {
    results[[nm]] <- make_skip_row(nm, nrow(fit_frame), n_pos, n_neg,
                                   "low_count")
    next
  }
  if (nrow(fit_frame) < 10) {
    results[[nm]] <- make_skip_row(nm, nrow(fit_frame), n_pos, n_neg,
                                   "insufficient_species_after_covariate_filter")
    next
  }

  rhs_terms <- c("x", covariates)
  fml <- as.formula(paste0("y ~ ", paste(rhs_terms, collapse = " + ")))

  fit <- tryCatch(
    phylolm::phyloglm(fml, data = fit_frame, phy = tree_this,
                      method = phyloglm_method, btol = btol,
                      boot = boot_n),
    error = function(e) e
  )

  if (inherits(fit, "error")) {
    results[[nm]] <- make_skip_row(nm, nrow(fit_frame), n_pos, n_neg,
                                   paste("phyloglm_error:", conditionMessage(fit)))
    next
  }

  coefs <- summary(fit)$coefficients
  row_idx <- if ("x" %in% rownames(coefs)) "x" else 2
  coef  <- coefs[row_idx, "Estimate"]
  se    <- coefs[row_idx, "StdErr"]
  zval  <- coefs[row_idx, "z.value"]
  pval  <- coefs[row_idx, "p.value"]
  alpha <- tryCatch(fit$alpha, error = function(e) NA_real_)

  results[[nm]] <- data.frame(
    test_label = nm,
    n_species = nrow(fit_frame),
    n_predictor_present = n_pos,
    n_predictor_absent = n_neg,
    phyloglm_coefficient = coef,
    phyloglm_std_err = se,
    phyloglm_z_value = zval,
    phyloglm_p_value = pval,
    phyloglm_alpha = alpha,
    phyloglm_method = phyloglm_method,
    n_covariates_used = length(covariates),
    mode = mode,
    skip_reason = NA_character_,
    stringsAsFactors = FALSE
  )
}

out <- do.call(rbind, results)
write.table(out, out_path, sep = "\t", quote = FALSE, row.names = FALSE)
