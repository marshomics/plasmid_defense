# Shared-frame support for the R scripts.
#
# The Python bridge can hand a script EITHER a per-call data TSV (the original
# behaviour) or a single shared TSV written once for the whole stage, plus two
# tiny side files:
#
#   row_filter_file  one column of key values; keep only these rows
#   override_file    a keyed table whose columns REPLACE the shared frame's
#
# This exists because writing the full ~40 MB species x feature frame for every
# one of ~4,800 R calls produced ~190 GB of concurrent temporary writes, which
# is the likely cause of the SIGBUS crashes that killed LOCO and the
# misclassification Monte Carlo. Row filters and overrides are a few hundred KB.
#
# Source this immediately after reading `params` and `data`:
#
#   source(file.path(dirname(script_path), "_shared_data.R"))
#   data <- apply_shared_filters(data, params)

apply_shared_filters <- function(data, params) {
  key <- if (!is.null(params$shared_key)) params$shared_key else "tip"

  # ---- row filter ----
  if (!is.null(params$row_filter_file) && nzchar(params$row_filter_file) &&
      file.exists(params$row_filter_file)) {
    keep <- read.delim(params$row_filter_file, sep = "\t",
                       stringsAsFactors = FALSE, check.names = FALSE)
    if (!(key %in% colnames(keep))) {
      stop("row_filter_file has no '", key, "' column; found: ",
           paste(colnames(keep), collapse = ", "))
    }
    before <- nrow(data)
    data <- data[data[[key]] %in% keep[[key]], , drop = FALSE]
    cat(file = stderr(), sprintf(
      "[shared_data] row filter: %d -> %d rows\n", before, nrow(data)))
    if (nrow(data) == 0) stop("row filter removed every row")
  }

  # ---- column overrides ----
  # Used by stages where only ONE column changes between calls (the permuted
  # outcome in the negative control, the resampled outcome in the
  # misclassification MC, the synthetic predictors in the feature control).
  if (!is.null(params$override_file) && nzchar(params$override_file) &&
      file.exists(params$override_file)) {
    ov <- read.delim(params$override_file, sep = "\t",
                     stringsAsFactors = FALSE, check.names = FALSE)
    if (!(key %in% colnames(ov))) {
      stop("override_file has no '", key, "' column; found: ",
           paste(colnames(ov), collapse = ", "))
    }
    ov_cols <- setdiff(colnames(ov), key)
    idx <- match(data[[key]], ov[[key]])
    n_matched <- sum(!is.na(idx))
    if (n_matched == 0) {
      stop("override_file keys do not match any row of the shared frame")
    }
    for (cc in ov_cols) {
      # New columns are created; existing ones are replaced in place. Rows the
      # override does not mention keep their shared-frame value, so a partial
      # override is well defined.
      if (cc %in% colnames(data)) {
        vals <- data[[cc]]
        vals[!is.na(idx)] <- ov[[cc]][idx[!is.na(idx)]]
        data[[cc]] <- vals
      } else {
        data[[cc]] <- ov[[cc]][idx]
      }
    }
    cat(file = stderr(), sprintf(
      "[shared_data] applied %d override column(s) to %d/%d rows: %s\n",
      length(ov_cols), n_matched, nrow(data),
      paste(head(ov_cols, 8), collapse = ", ")))
  }

  data
}
