# IREE tokenizer source boundary

NML builds the tokenizer implementation from the original
[`iree-org/iree`](https://github.com/iree-org/iree) repository at commit
`4d4e97d00f099a21f38eeff26f82a6d9e3643a11`. A sparse checkout retains only
IREE's runtime source and Bazel definitions.

The three local patches are the audited tokenizer integration patch set used by
the pinned ZML reference:

- `tokenizer-only.patch` removes compiler-only Bazel loads from the deliberately
  sparse runtime repository;
- `fix-added-token-matching.patch` makes every Hugging Face added token
  participate in pre-segmentation matching;
- `match-hf-tokenizer.patch` corrects bounded BPE backtracking and boundary
  behavior so streamed results match Hugging Face tokenizers.

They are carried locally under ZML's Apache-2.0 license as permitted external
build logic under D-010. NML does not depend on a ZML source repository, Bazel
target, or IREE fork: Bazel applies the reviewed changes to the pinned original
IREE commit.
