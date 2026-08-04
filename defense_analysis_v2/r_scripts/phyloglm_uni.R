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
# Pagel's lambda applied to the tree BEFORE fitting. 1 = untransformed.
# This is the only knob in this script that genuinely changes the assumed
# phylogenetic covariance structure (see the estimator note below).
lambda_rescale    <- if (!is.null(params$lambda_rescale)) as.numeric(params$lambda_rescale) else 1.0
# Which side of the fit is the defense system, so the prevalence gate can be
# applied to it in BOTH directions. "predictor" (forward: plasmid ~ defense)
# or "response" (reverse: defense ~ plasmid). Defaults to the fit mode.
defense_side      <- if (!is.null(params$defense_side)) params$defense_side else mode
# Minimum count for BOTH levels of the binary response. Previously unset:
# only the predictor was gated, so a depth-filtered rerun whose outcome had
# been driven to 97% positive was fit at near-zero power without comment.
min_count_response <- if (!is.null(params$min_count_response)) params$min_count_response else min_count

# ESTIMATOR selection. `phylolm::phyloglm`'s `method` argument chooses the
# ESTIMATOR, not the evolutionary process: there is no BM-vs-OU switch for
# binary traits. The previous mapping sent both "BM" and "OUfixedRoot" to
# logistic_MPLE, so the "OU sensitivity analysis" was bit-identical to the
# primary fit and measured nothing. Genuine covariance-structure sensitivity
# is done by rescaling the tree under Pagel's lambda -- see lambda_rescale.
phyloglm_method <- switch(evol_model,
  MPLE         = "logistic_MPLE",
  IG10         = "logistic_IG10",
  BM           = "logistic_MPLE",     # legacy alias
  BM_penalized = "logistic_IG10",     # legacy alias
  OUfixedRoot  = "logistic_MPLE",     # legacy alias; identical to MPLE
  "logistic_MPLE"
)
if (evol_model == "OUfixedRoot") {
  cat(file = stderr(), paste0(
    "[phyloglm_uni.R] NOTE: evolutionary_model='OUfixedRoot' maps to the same\n",
    "  estimator as the primary fit (logistic_MPLE) and is retained only as a\n",
    "  legacy alias. Use lambda_rescale for covariance-structure sensitivity.\n"))
}

# Sanitise a string for single-cell TSV output. Error messages routinely
# contain tabs and newlines; written verbatim they corrupt the TSV that the
# Python bridge then parses with a fixed column count.
tsv_safe <- function(s) {
  if (is.null(s) || is.na(s)) return(NA_character_)
  s <- gsub("[\t\r\n]+", " ", as.character(s))
  s <- gsub('"', "'", s, fixed = TRUE)
  substr(s, 1, 300)
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

# ---- Pagel's lambda rescaling (covariance-structure sensitivity) ----
# lambda multiplies every internal branch while holding tip heights fixed.
# lambda = 1 leaves the tree untouched; lambda -> 0 approaches a star
# phylogeny (no phylogenetic covariance). Defense systems and plasmids move
# horizontally, so BM covariance is a simplifying assumption; this is the
# axis a reviewer will push on, and it is the one that actually changes the
# model.
if (is.finite(lambda_rescale) && abs(lambda_rescale - 1) > 1e-9) {
  rescaled <- tryCatch(
    phylolm::transf.branch.lengths(tree, model = "lambda",
                                   parameters = list(lambda = lambda_rescale))$tree,
    error = function(e) NULL
  )
  if (is.null(rescaled)) {
    cat(file = stderr(), sprintf(
      "[phyloglm_uni.R] WARNING: lambda rescale to %.3f failed; using untransformed tree\n",
      lambda_rescale))
  } else {
    tree <- rescaled
    data <- data[tree$tip.label, , drop = FALSE]
    cat(file = stderr(), sprintf(
      "[phyloglm_uni.R] tree rescaled under Pagel's lambda = %.3f\n",
      lambda_rescale))
  }
}

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

make_skip_row <- function(nm, n, n_pos, n_neg, reason,
                          n_resp_pos = NA_integer_, n_resp_neg = NA_integer_,
                          n_defense_pos = NA_integer_,
                          n_defense_neg = NA_integer_) {
  data.frame(
    test_label = nm, n_species = n,
    n_predictor_present = n_pos, n_predictor_absent = n_neg,
    n_response_present = n_resp_pos, n_response_absent = n_resp_neg,
    n_defense_present = n_defense_pos, n_defense_absent = n_defense_neg,
    phyloglm_coefficient = NA_real_, phyloglm_std_err = NA_real_,
    phyloglm_z_value = NA_real_, phyloglm_p_value = NA_real_,
    phyloglm_alpha = NA_real_,
    phyloglm_method = phyloglm_method,
    phyloglm_lambda_rescale = lambda_rescale,
    n_covariates_used = length(covariates),
    mode = mode,
    phyloglm_converged = NA,
    phyloglm_degenerate = NA,
    phyloglm_fit_warning = NA_character_,
    skip_reason = tsv_safe(reason),
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

  is_binary_resp <- all(fit_frame$y %in% c(0, 1, 0L, 1L))
  n_resp_pos <- if (is_binary_resp) sum(fit_frame$y == 1) else NA_integer_
  n_resp_neg <- if (is_binary_resp) sum(fit_frame$y == 0) else NA_integer_

  # Which side carries the defense system? In forward mode ("predictor") the
  # defense system is x; in reverse mode ("response") it is y. The prevalence
  # gate must follow the defense system, not the fit position.
  #
  # This was the bug: the gate tested `x` unconditionally, so in the reverse
  # direction it checked the balance of the PLASMID column and never looked at
  # the defense system at all. A system present in 2 of 15,000 species passed,
  # separated, and emitted a coefficient and a Wald p-value into the
  # reverse-direction FDR family.
  defense_is_x <- identical(defense_side, "predictor")
  n_defense_pos <- if (defense_is_x) n_pos else n_resp_pos
  n_defense_neg <- if (defense_is_x) n_neg else n_resp_neg

  skip_args <- list(nm, nrow(fit_frame), n_pos, n_neg,
                    n_resp_pos = n_resp_pos, n_resp_neg = n_resp_neg,
                    n_defense_pos = n_defense_pos,
                    n_defense_neg = n_defense_neg)
  emit_skip <- function(reason) {
    do.call(make_skip_row, c(skip_args[1:4], list(reason),
                             skip_args[5:8]))
  }

  # phyloglm requires a binary response.
  if (!is_binary_resp) {
    results[[nm]] <- emit_skip("response_not_binary")
    next
  }
  # Gate on the defense system's prevalence, whichever side it is on.
  if (is.finite(n_defense_pos) && is.finite(n_defense_neg) &&
      (n_defense_pos < min_count || n_defense_neg < min_count)) {
    results[[nm]] <- emit_skip("low_count_defense")
    next
  }
  # Gate on the RESPONSE having both levels represented. Without this a
  # depth-filtered subset whose outcome prevalence has been driven to ~97%
  # is fit silently at near-zero power and reported as a clean null.
  if (n_resp_pos < min_count_response || n_resp_neg < min_count_response) {
    results[[nm]] <- emit_skip("low_count_response")
    next
  }
  # Non-binary predictor still needs variance.
  if (!is_binary_pred && sd(fit_frame$x) <= 0) {
    results[[nm]] <- emit_skip("predictor_no_variance")
    next
  }
  if (nrow(fit_frame) < 10) {
    results[[nm]] <- emit_skip("insufficient_species_after_covariate_filter")
    next
  }
  # Complete separation screen. If every species with x = 1 has the same y
  # (or vice versa), the MLE is at infinity; phyloglm will not error, it will
  # walk the coefficient out to btol and return a Wald p-value computed from a
  # meaningless standard error.
  if (is_binary_pred) {
    tab <- table(factor(fit_frame$x, levels = c(0, 1)),
                 factor(fit_frame$y, levels = c(0, 1)))
    if (any(rowSums(tab) == 0) || any(colSums(tab) == 0) || any(tab == 0)) {
      results[[nm]] <- emit_skip("complete_or_quasi_separation")
      next
    }
  }

  rhs_terms <- c("x", covariates)
  fml <- as.formula(paste0("y ~ ", paste(rhs_terms, collapse = " + ")))

  # Capture warnings as well as errors. `phylolm::phyloglm` does NOT stop() on
  # non-convergence or when the optimiser pins alpha at its bound -- it
  # warning()s and returns an object that looks entirely normal. Catching only
  # errors is why boundary-hit and separated fits were previously written with
  # skip_reason = NA, indistinguishable from clean fits, and then FDR-corrected
  # and exponentiated downstream.
  fit_warnings <- character(0)
  fit <- withCallingHandlers(
    tryCatch(
      phylolm::phyloglm(fml, data = fit_frame, phy = tree_this,
                        method = phyloglm_method, btol = btol,
                        boot = boot_n),
      error = function(e) e
    ),
    warning = function(w) {
      fit_warnings <<- c(fit_warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )

  if (inherits(fit, "error")) {
    results[[nm]] <- emit_skip(paste("phyloglm_error:", conditionMessage(fit)))
    next
  }

  coefs <- tryCatch(summary(fit)$coefficients, error = function(e) NULL)
  if (is.null(coefs) || !("Estimate" %in% colnames(coefs))) {
    results[[nm]] <- emit_skip("summary_unavailable")
    next
  }
  row_idx <- if ("x" %in% rownames(coefs)) "x" else 2
  coef  <- coefs[row_idx, "Estimate"]
  se    <- coefs[row_idx, "StdErr"]
  zval  <- coefs[row_idx, "z.value"]
  pval  <- coefs[row_idx, "p.value"]
  # fit$alpha is NULL for some methods; force a scalar so the data.frame below
  # always has the same column count and rbind() cannot fail on a ragged list.
  alpha <- tryCatch({
    a <- fit$alpha
    if (is.null(a) || length(a) == 0) NA_real_ else as.numeric(a)[1]
  }, error = function(e) NA_real_)

  # ---- Explicit convergence / degeneracy verdict ----
  # A fit is degenerate if any of:
  #   * the optimiser reported non-convergence
  #   * the coefficient walked out to the btol bound (separation signature;
  #     btol = 20 admits odds ratios up to e^20 ~ 5e8)
  #   * the standard error is non-finite or numerically zero
  #   * the Wald p-value is non-finite
  # Degenerate fits get an NA p-value so `apply_fdr` excludes them from the
  # family rather than ranking a fictitious 1e-300 at the top of the table.
  conv_code <- tryCatch({
    cc <- fit$convergence
    if (is.null(cc) || length(cc) == 0) NA_integer_ else as.integer(cc)[1]
  }, error = function(e) NA_integer_)
  hit_bound <- is.finite(coef) && abs(coef) >= (btol - 1e-6)
  bad_se    <- !is.finite(se) || se <= 0 || se > 1e4
  bad_p     <- !is.finite(pval)
  warned_converge <- any(grepl("converg|bound|singular|NaN|infinite",
                               fit_warnings, ignore.case = TRUE))
  degenerate <- hit_bound || bad_se || bad_p || warned_converge ||
                (is.finite(conv_code) && conv_code != 0)
  converged  <- !degenerate

  if (degenerate) {
    reason <- if (hit_bound) "coefficient_at_btol_bound"
              else if (bad_se) "nonfinite_or_extreme_std_err"
              else if (bad_p) "nonfinite_p_value"
              else if (warned_converge) "convergence_warning"
              else "nonzero_convergence_code"
    row <- emit_skip(paste0("degenerate_fit:", reason))
    # Keep the numbers for forensics, but the p-value stays NA so the row
    # cannot enter FDR or consensus.
    row$phyloglm_coefficient <- coef
    row$phyloglm_std_err     <- se
    row$phyloglm_z_value     <- zval
    row$phyloglm_alpha       <- alpha
    row$phyloglm_converged   <- FALSE
    row$phyloglm_degenerate  <- TRUE
    row$phyloglm_fit_warning <- tsv_safe(paste(fit_warnings, collapse = "; "))
    results[[nm]] <- row
    next
  }

  results[[nm]] <- data.frame(
    test_label = nm,
    n_species = nrow(fit_frame),
    n_predictor_present = n_pos,
    n_predictor_absent = n_neg,
    n_response_present = n_resp_pos,
    n_response_absent = n_resp_neg,
    n_defense_present = n_defense_pos,
    n_defense_absent = n_defense_neg,
    phyloglm_coefficient = coef,
    phyloglm_std_err = se,
    phyloglm_z_value = zval,
    phyloglm_p_value = pval,
    phyloglm_alpha = alpha,
    phyloglm_method = phyloglm_method,
    phyloglm_lambda_rescale = lambda_rescale,
    n_covariates_used = length(covariates),
    mode = mode,
    phyloglm_converged = TRUE,
    phyloglm_degenerate = FALSE,
    phyloglm_fit_warning = if (length(fit_warnings))
      tsv_safe(paste(fit_warnings, collapse = "; ")) else NA_character_,
    skip_reason = NA_character_,
    stringsAsFactors = FALSE
  )
}

out <- do.call(rbind, results)

# Fit-outcome census to stderr so a stage where most systems failed is visible
# in the log instead of looking like a stage where most systems were null.
n_fit  <- sum(is.na(out$skip_reason))
n_skip <- sum(!is.na(out$skip_reason))
cat(file = stderr(), sprintf(
  "[phyloglm_uni.R] fits: %d converged, %d skipped/degenerate (of %d)\n",
  n_fit, n_skip, nrow(out)))
if (n_skip > 0) {
  tallies <- sort(table(sub(":.*$", "", out$skip_reason[!is.na(out$skip_reason)])),
                  decreasing = TRUE)
  cat(file = stderr(), sprintf("[phyloglm_uni.R]   %s = %d\n",
                               names(tallies), as.integer(tallies)))
}

write.table(out, out_path, sep = "\t", quote = FALSE, row.names = FALSE,
            na = "NA")
