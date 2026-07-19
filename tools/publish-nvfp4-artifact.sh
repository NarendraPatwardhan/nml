#!/usr/bin/env bash
# Convert and publish the checked GPT-OSS NVFP4 artifact on a CPU-only
# BuildBuddy runner. Model bytes travel directly between Hugging Face and the
# remote worker; neither they nor a Python environment touch the operator host.

set -euo pipefail

readonly REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly TOKEN_FILE="${HF_TOKEN_FILE:-${REPOSITORY_ROOT}/../hf.gptoss.key}"

if [[ ! -r "${TOKEN_FILE}" ]]; then
  echo "Hugging Face token file is not readable: ${TOKEN_FILE}" >&2
  exit 1
fi

cd "${REPOSITORY_ROOT}"

TOKEN="$(tr -d '\r\n' < "${TOKEN_FILE}")"
if [[ -z "${TOKEN}" ]]; then
  echo "Hugging Face token file is empty: ${TOKEN_FILE}" >&2
  exit 1
fi

# The token is a redacted runner secret, never a Bazel flag, action input,
# repository file, or log field. Automatic retry is disabled because the final
# operation creates an externally visible Hugging Face commit.
SECRET_HEADER="$(printf 'HF_TOKEN=%s' "${TOKEN}" | base64 | tr -d '\r\n')"
unset TOKEN

bb remote \
  --disable_retry \
  --timeout=3h \
  --runner_exec_properties=recycle-runner=false \
  --runner_exec_properties=EstimatedCPU=24 \
  --runner_exec_properties=EstimatedFreeDiskBytes=80GB \
  --remote_run_header="x-buildbuddy-platform.secret-env-overrides-base64=${SECRET_HEADER}" \
  --script='set -eu
umask 077
work_directory="$(mktemp -d)"
token_file="$(mktemp)"
cleanup() { rm -f "$token_file"; }
trap cleanup EXIT
printf "%s" "$HF_TOKEN" > "$token_file"
unset HF_TOKEN

curl --proto "=https" --tlsv1.2 -LsSf https://astral.sh/uv/0.8.17/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12.11
uv venv --python 3.12.11 "$work_directory/venv"
uv pip install --python "$work_directory/venv/bin/python" --requirement tools/nvfp4/requirements.txt

"$work_directory/venv/bin/python" tools/nvfp4/convert.py \
  --source-manifest artifacts/gpt-oss-20b-nvfp4/source.json \
  --tensor-manifest artifacts/gpt-oss-20b-nvfp4/source-tensors.json \
  --recipe artifacts/gpt-oss-20b-nvfp4/recipe.json \
  --work-directory "$work_directory/artifact" \
  --destination-repository narendra747/gpt-oss-20b-nvfp4 \
  --hf-token-file "$token_file" \
  --resume'

unset SECRET_HEADER
