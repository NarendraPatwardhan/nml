#!/usr/bin/env bash
# Publish NML's CUDA serving image entirely on a BuildBuddy remote runner.
# The runner checks out one pushed source commit, materializes the image beside
# BuildBuddy's cache, and pushes it directly to GHCR. No OCI layer traverses the
# operator machine.

set -euo pipefail

readonly REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly TOKEN_FILE="${GHCR_TOKEN_FILE:-${REPOSITORY_ROOT}/../github.packages.key}"

if [[ ! -r "${TOKEN_FILE}" ]]; then
  echo "GHCR token file is not readable: ${TOKEN_FILE}" >&2
  exit 1
fi

cd "${REPOSITORY_ROOT}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to publish: BuildBuddy publication requires a clean, pushed commit." >&2
  exit 1
fi

readonly SOURCE_COMMIT="$(git rev-parse HEAD)"
TOKEN="$(tr -d '\r\n' < "${TOKEN_FILE}")"
if [[ -z "${TOKEN}" ]]; then
  echo "GHCR token file is empty: ${TOKEN_FILE}" >&2
  exit 1
fi

# secret-env-overrides-base64 is BuildBuddy's redacted short-lived secret
# channel. Only this small header leaves the operator machine; the token is not
# a Bazel flag, action environment, cache key, repository file, or log field.
SECRET_HEADER="$(printf 'GHCR_TOKEN=%s' "${TOKEN}" | base64 | tr -d '\r\n')"
unset TOKEN

bb remote \
  --run_from_commit="${SOURCE_COMMIT}" \
  --disable_retry \
  --timeout=45m \
  --runner_exec_properties=recycle-runner=false \
  --runner_exec_properties=EstimatedFreeDiskBytes=20GB \
  --remote_run_header="x-buildbuddy-platform.secret-env-overrides-base64=${SECRET_HEADER}" \
  --script='set -eu
umask 077
auth_dir="$(mktemp -d)"
cleanup() { rm -rf "$auth_dir"; }
trap cleanup EXIT
export DOCKER_CONFIG="$auth_dir"
auth="$(printf "%s" "NarendraPatwardhan:${GHCR_TOKEN}" | base64 | tr -d "\n")"
printf "%s\n" "{\"auths\":{\"ghcr.io\":{\"auth\":\"$auth\"}}}" > "$DOCKER_CONFIG/config.json"
unset auth GHCR_TOKEN
bazel run --config=buildbuddy --config=cuda //products/serve:publish_serve_image &
publisher_pid=$!
while kill -0 "$publisher_pid" 2>/dev/null; do
  printf "Publisher still active at %s\n" "$(date -u +%H:%M:%S)"
  sleep 20
done
wait "$publisher_pid"'

unset SECRET_HEADER
