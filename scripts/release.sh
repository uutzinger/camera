#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Error: required file missing: ${path}"
    exit 2
  fi
}

validate_version_format() {
  local version="$1"
  if [[ ! "${version}" =~ ^[0-9]+(\.[0-9]+){2}([a-zA-Z0-9._-]*)?$ ]]; then
    echo "Error: camera/__version__.py must provide X.Y.Z (PEP 440 compatible suffixes allowed)."
    exit 2
  fi
}

project_version() {
  python3 - <<'PY'
from pathlib import Path
namespace = {}
version_file = Path("camera/__version__.py")
exec(version_file.read_text(encoding="utf-8"), namespace)
print(namespace["__version__"])
PY
}

usage() {
  cat <<'EOF'
Usage:
  scripts/release.sh [options]

Options:
  --commit-msg "message"   Commit message (default: "release: <version>").
  --install                Install built wheel after build.
  --commit                 Stage + commit build/release files.
  --tag                    Create git tag "v<version>".
  --push                   Push commit and tags.
  --upload-testpypi        Upload artifacts in dist/ to TestPyPI via twine.
  --upload-pypi            Upload artifacts in dist/ to PyPI via twine.
  --clean                  Remove build/, dist/, *.egg-info before build.
  --isolation              Use isolated PEP517 build env (default: no isolation).
  -h, --help               Show this help.

Notes:
  - Runs from repository root and validates required packaging files.
  - Release version is read from camera/__version__.py.
  - Requires twine for upload steps.
EOF
}

COMMIT_MSG=""
DO_INSTALL=0
DO_COMMIT=0
DO_TAG=0
DO_PUSH=0
DO_UPLOAD_TESTPYPI=0
DO_UPLOAD_PYPI=0
DO_CLEAN=0
USE_ISOLATION=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit-msg) COMMIT_MSG="${2:-}"; shift 2 ;;
    --install) DO_INSTALL=1; shift ;;
    --commit) DO_COMMIT=1; shift ;;
    --tag) DO_TAG=1; shift ;;
    --push) DO_PUSH=1; shift ;;
    --upload-testpypi) DO_UPLOAD_TESTPYPI=1; shift ;;
    --upload-pypi) DO_UPLOAD_PYPI=1; shift ;;
    --clean) DO_CLEAN=1; shift ;;
    --isolation) USE_ISOLATION=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

require_file "setup.py"
require_file "README.rst"
require_file "camera/__version__.py"

PACKAGE_VERSION="$(project_version)"
if [[ -z "${PACKAGE_VERSION}" ]]; then
  echo "Error: camera/__version__.py did not provide __version__."
  exit 2
fi
validate_version_format "${PACKAGE_VERSION}"
echo "Release version: ${PACKAGE_VERSION}"

if [[ "${DO_UPLOAD_TESTPYPI}" -eq 1 && "${DO_UPLOAD_PYPI}" -eq 1 ]]; then
  echo "Error: choose only one of --upload-testpypi or --upload-pypi."
  exit 2
fi

if [[ "${DO_CLEAN}" -eq 1 ]]; then
  rm -rf build dist ./*.egg-info ./*.egg .pytest_cache
fi

if ! python3 -m build --help >/dev/null 2>&1; then
  echo "Error: python build package not found. Install with: python3 -m pip install build"
  exit 2
fi

if [[ "${USE_ISOLATION}" -eq 1 ]]; then
  python3 -m build
else
  python3 -m build --no-isolation
fi

shopt -s nullglob
WHEELS=(dist/*.whl)
shopt -u nullglob
if [[ "${#WHEELS[@]}" -eq 0 ]]; then
  echo "Error: build completed but no wheel found in dist/."
  exit 2
fi
WHEEL_PATH="$(ls -t dist/*.whl | head -n1)"
echo "Built wheel: ${WHEEL_PATH}"

if [[ "${DO_INSTALL}" -eq 1 ]]; then
  python3 -m pip install --no-deps --force-reinstall "${WHEEL_PATH}"
fi

if [[ "${DO_UPLOAD_TESTPYPI}" -eq 1 || "${DO_UPLOAD_PYPI}" -eq 1 ]]; then
  if ! python3 -m twine --help >/dev/null 2>&1; then
    echo "Error: twine not found. Install with: python3 -m pip install twine"
    exit 2
  fi
  # Upload only PyPI-compatible artifacts by default:
  # - Always include source distributions.
  # - Include wheels except local platform-tagged Linux wheels (e.g. linux_x86_64),
  #   which PyPI rejects. manylinux/musllinux wheels remain included.
  shopt -s nullglob
  upload_files=(dist/*.tar.gz)
  for whl in dist/*.whl; do
    base="$(basename "${whl}")"
    if [[ "${base}" == *"-linux_"*".whl" ]]; then
      echo "Skipping non-PyPI wheel: ${whl}"
      continue
    fi
    upload_files+=("${whl}")
  done
  shopt -u nullglob
  if [[ "${#upload_files[@]}" -eq 0 ]]; then
    echo "Error: no uploadable artifacts found in dist/."
    exit 2
  fi
  if [[ "${DO_UPLOAD_TESTPYPI}" -eq 1 ]]; then
    python3 -m twine check "${upload_files[@]}"
    python3 -m twine upload --repository testpypi "${upload_files[@]}"
  else
    python3 -m twine check "${upload_files[@]}"
    python3 -m twine upload "${upload_files[@]}"
  fi
fi

if [[ "${DO_COMMIT}" -eq 1 ]]; then
  if [[ -z "${COMMIT_MSG}" ]]; then
    COMMIT_MSG="release: ${PACKAGE_VERSION}"
  fi
  git add -A
  git commit -m "${COMMIT_MSG}"
fi

if [[ "${DO_TAG}" -eq 1 ]]; then
  git tag "v${PACKAGE_VERSION}"
fi

if [[ "${DO_PUSH}" -eq 1 ]]; then
  git push
  git push --tags
fi

echo "Release script completed."
