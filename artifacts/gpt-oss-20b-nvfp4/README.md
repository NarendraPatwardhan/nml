# GPT-OSS 20B NVFP4 artifact

This directory freezes the input and deterministic conversion contract for
NML's first NVFP4 product artifact. It does not vendor model payloads.

The source is `unsloth/gpt-oss-20b-BF16` at immutable revision
`cc89b3e7fd423253264883a80a4fa5abc619649f`. No sufficiently auditable
NVIDIA- or Unsloth-published GPT-OSS 20B NVFP4 checkpoint existed when the
selection was made on 2026-07-17. Community repositories were not admitted as
product inputs merely because their titles contained `NVFP4`.

[`source.json`](./source.json) records every source file and content hash.
[`source-tensors.json`](./source-tensors.json) records all 411 tensors, their
source shard, dtype, logical shape, byte extent, role, mapping, transpose, and
whether the NML artifact retains them as BF16 or converts them to NVFP4.
[`recipe.json`](./recipe.json) is the exact conversion and output contract.

The public generated artifact is
[`narendra747/gpt-oss-20b-nvfp4`](https://huggingface.co/narendra747/gpt-oss-20b-nvfp4)
at immutable revision `704c34282b2d84cc6a4e5ce7de14b6f6fc1286e9`.
[`published.json`](./published.json) records that identity and
[`artifact-manifest.json`](./artifact-manifest.json) is the byte-exact public
manifest downloaded back from that revision. Its SHA-256 is
`3c36a89cbc0f908b3e782550fe32f3b6890ef3f857232d11710bc8e0dbcea71d`.
The 20 payload/metadata files total 11,805,934,204 bytes. The separately
authenticated artifact manifest itself is 4,118 bytes.

The permanent auditor at `//tools/nvfp4:audit_published_artifact` can derive a
complete physical tensor inventory from bounded SafeTensor header reads. The
companion `//tools/nvfp4:extract_execution_fixture` target can then freeze exact
compact rows and independent decoded projections. Recipe-v1 inventories and
fixtures were deleted when recipe v2 became authoritative; generated evidence
is admitted only when it names the immutable recipe-v2 revision above.
