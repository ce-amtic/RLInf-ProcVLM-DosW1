#!/usr/bin/env bash
set -euo pipefail

ROBOTWIN_DIR="${1:-third_party/RoboTwin}"
EXPECTED_BASE="0008ae6800df9f75fc8de7098bacb01735fd8fd2"
PATCH_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/patches/robotwin-0008ae6-runtime-fixes.patch"

if [[ ! -d "${ROBOTWIN_DIR}/.git" ]]; then
  echo "RoboTwin Git checkout not found: ${ROBOTWIN_DIR}" >&2
  exit 1
fi

actual_base="$(git -C "${ROBOTWIN_DIR}" rev-parse HEAD)"
if [[ "${actual_base}" != "${EXPECTED_BASE}" ]]; then
  echo "Expected RoboTwin ${EXPECTED_BASE}, found ${actual_base}." >&2
  echo "Refusing to apply an unverified patch base." >&2
  exit 1
fi

git -C "${ROBOTWIN_DIR}" apply --check "${PATCH_FILE}"
git -C "${ROBOTWIN_DIR}" apply "${PATCH_FILE}"
echo "Applied ${PATCH_FILE} to ${ROBOTWIN_DIR}."
