#!/usr/bin/env Rscript
# Install the R packages required by the phylogenetic stages of
# defense_analysis_v2. Run once per environment:
#
#   Rscript install_r_packages.R
#
# Honors $R_MIRROR if set; defaults to the RStudio CRAN mirror. Skips any
# package that's already installed so the script is idempotent.

cran_mirror <- Sys.getenv("R_MIRROR", unset = "https://cloud.r-project.org")
options(repos = c(CRAN = cran_mirror))

required <- c(
  "ape",       # tree I/O, pruning
  "phylolm",   # phyloglm (univariate logistic regression with tree)
  "phytools",  # fitPagel (correlated-evolution test)
  "caper",     # Fritz & Purvis D statistic (phylo_d.R)
  "phyr",      # pglmm (multivariate GLMM with tree)
  "nlme",      # gls with corBrownian / corPagel (PGLS + phylo-residuals)
  "jsonlite"   # args.json parsing at the Python <-> R boundary
)

installed <- rownames(installed.packages())
missing <- setdiff(required, installed)

if (length(missing) == 0) {
  cat("All required R packages already installed.\n")
  quit(status = 0)
}

cat("Installing missing R packages from", cran_mirror, ":\n  ",
    paste(missing, collapse = ", "), "\n", sep = "")

# Ncpus: use multiple cores if available. On headless CI a single thread is
# safer, so we respect $MAKEFLAGS-style -j when set, otherwise cap at 4.
ncpus <- as.integer(Sys.getenv("NCPUS", unset = "0"))
if (!is.finite(ncpus) || ncpus < 1) {
  ncpus <- min(4L, parallel::detectCores(logical = FALSE))
}

install.packages(missing, Ncpus = ncpus, dependencies = TRUE)

# Verify install. A failed install() can exit with status 0, so we
# re-check installed.packages() and exit non-zero if anything is still
# missing. This lets install.sh detect R-side failures.
still_missing <- setdiff(required, rownames(installed.packages()))
if (length(still_missing) > 0) {
  cat("\nERROR: the following R packages failed to install:\n  ",
      paste(still_missing, collapse = ", "), "\n",
      "See output above for build errors. Common causes: missing system\n",
      "libraries (libxml2, gfortran, gsl) — install them via your OS\n",
      "package manager and retry.\n", sep = "")
  quit(status = 1)
}

cat("\nAll required R packages installed successfully.\n")
