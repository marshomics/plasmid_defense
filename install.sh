#!/usr/bin/env bash
# One-command installer for defense_analysis_v2.
#
# Installs the Python package (editable mode) and the R packages it calls
# via subprocess. Safe to re-run; each step is idempotent.
#
# Usage:
#   ./install.sh             # use the Python currently on PATH
#   ./install.sh --venv      # create & install into ./.venv
#   ./install.sh --conda     # use conda env from environment.yml
#   ./install.sh --no-r      # skip the R package install (Python-only)
#
# Requires: python >= 3.9, pip, and (unless --no-r) R >= 4.0 with a
# functioning CRAN mirror. On macOS you may need Xcode command-line tools
# for R source builds (`xcode-select --install`).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

USE_VENV=0
USE_CONDA=0
INSTALL_R=1

for arg in "$@"; do
  case "$arg" in
    --venv)   USE_VENV=1 ;;
    --conda)  USE_CONDA=1 ;;
    --no-r)   INSTALL_R=0 ;;
    -h|--help)
      sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown flag: $arg (use --help)" >&2
      exit 1
      ;;
  esac
done

if [[ $USE_VENV -eq 1 && $USE_CONDA -eq 1 ]]; then
  echo "--venv and --conda are mutually exclusive." >&2
  exit 1
fi

# ----------------------------------------------------------------------
# 1. Python + pip + package
# ----------------------------------------------------------------------
echo "==> Checking Python"
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on PATH. Install Python >= 3.9 and retry." >&2
  exit 1
fi
PYVER="$(python3 -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))')"
PYOK="$(python3 -c 'import sys; print(sys.version_info >= (3,9))')"
if [[ "$PYOK" != "True" ]]; then
  echo "Python $PYVER found; need >= 3.9." >&2
  exit 1
fi
echo "    Python $PYVER OK"

PIP=(python3 -m pip)

if [[ $USE_CONDA -eq 1 ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH but --conda was requested." >&2
    exit 1
  fi
  echo "==> Creating conda env 'defense-analysis-v2' from environment.yml"
  conda env create -f environment.yml -n defense-analysis-v2 || \
    conda env update -f environment.yml -n defense-analysis-v2
  # Switch pip to the env's python for the editable install
  PIP=(conda run -n defense-analysis-v2 python -m pip)
  # R calls below also go through conda
  RSCRIPT_BIN="conda run -n defense-analysis-v2 Rscript"
elif [[ $USE_VENV -eq 1 ]]; then
  if [[ ! -d .venv ]]; then
    echo "==> Creating virtualenv ./.venv"
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PIP=(python -m pip)
  RSCRIPT_BIN="Rscript"
else
  RSCRIPT_BIN="Rscript"
fi

echo "==> Upgrading pip"
"${PIP[@]}" install --upgrade pip setuptools wheel

echo "==> Installing defense_analysis_v2 (editable mode)"
"${PIP[@]}" install -e .

# ----------------------------------------------------------------------
# 2. R + CRAN packages
# ----------------------------------------------------------------------
if [[ $INSTALL_R -eq 1 ]]; then
  echo "==> Checking R"
  # Use `eval` so "conda run -n env Rscript" still works for --conda
  if ! eval "$RSCRIPT_BIN --version" >/dev/null 2>&1; then
    echo "Rscript not found. Install R >= 4.0 (https://cran.r-project.org)" >&2
    echo "or rerun with --no-r to skip the R-side install." >&2
    exit 1
  fi
  RVER="$(eval "$RSCRIPT_BIN -e 'cat(as.character(getRversion()))'")"
  echo "    R $RVER OK"

  echo "==> Installing R packages (ape, phylolm, phytools, caper, phyr, nlme, jsonlite)"
  eval "$RSCRIPT_BIN install_r_packages.R"
else
  echo "==> Skipping R install (--no-r)"
fi

# ----------------------------------------------------------------------
# 3. Quick smoke test
# ----------------------------------------------------------------------
echo "==> Verifying Python package import"
"${PIP[@]/pip/python}" -c "import defense_analysis_v2; \
print('    defense_analysis_v2', defense_analysis_v2.__version__, 'OK')"

cat <<'EOF'

==> Installation complete.

Next steps:
  defense-plasmid-analyze --help
  # or: python -m defense_analysis_v2.defense_plasmid_analysis --help

If you used --venv, activate it first:
  source .venv/bin/activate

If you used --conda:
  conda activate defense-analysis-v2
EOF
