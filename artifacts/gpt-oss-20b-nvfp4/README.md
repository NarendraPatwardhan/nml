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
at immutable revision `729e9053f43c267636bfda3d6659c4141ff3ea1d`.
[`published.json`](./published.json) records that identity and
[`artifact-manifest.json`](./artifact-manifest.json) is the byte-exact public
manifest downloaded back from that revision. Its SHA-256 is
`ab4c8cbd4424c8fec95bf683c0efd04c9cd350ec2a26737408b5500e61003207`.
The 20 payload/metadata files total 11,805,933,892 bytes.

[`output-tensors.json`](./output-tensors.json) is the bounded-header audit of
that immutable publication. It maps all 411 logical parameters to 703 physical
records with exact dtype, shape, byte extent, shard offset, role, logical
mapping, and transpose; the physical tensor payload is 11,777,751,752 bytes.
Its SHA-256 is
`7fa49a03db85d75df2bf0db520aefe950afb2e45f3fc12e7503d175f3b56bd1d`.
The permanent auditor at `//tools/nvfp4:audit_published_artifact` regenerates or
checks this inventory using HTTP range reads rather than model downloads.

[`execution-fixture.json`](./execution-fixture.json) freezes four widely
separated rows of the first attention query projection directly from that
publication. It includes the original compact bytes, decoded F32 hashes and
boundary samples, plus independent F64 projection results. The permanent
extractor at `//tools/nvfp4:extract_execution_fixture` uses bounded range reads
and rejects any artifact, inventory, component, or HTTP-range identity drift.
