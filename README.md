<div align="center">
  <h1>NML</h1>

  <p><strong>A focused acceleration substrate for CPU and NVIDIA CUDA.</strong></p>

  <p>
    Define tensor programs in Rust, compile them through StableHLO and XLA,
    and execute them through PJRT without making a framework own the model.
  </p>

  <p>
    <img alt="Language: Rust" src="https://img.shields.io/badge/language-Rust-b7410e">
    <img alt="Build: Bazel" src="https://img.shields.io/badge/build-Bazel-43a047">
    <img alt="Compiler: XLA" src="https://img.shields.io/badge/compiler-XLA-5c6bc0">
    <img alt="Backends: CPU and CUDA" src="https://img.shields.io/badge/backends-CPU%20%7C%20CUDA-2e7d32">
  </p>

  <p>
    <a href="#what-you-can-build">What You Can Build</a> ·
    <a href="#why-nml">Why NML</a> ·
    <a href="#quickstart">Quickstart</a> ·
    <a href="#current-capability-status">Capability Status</a> ·
    <a href="./SYSTEM.md">System Architecture</a> ·
    <a href="./TASKS.md">Engineering Ledger</a>
  </p>
</div>

NML is a Rust-native environment for authoring compiled tensor programs and
running them efficiently on CPUs and NVIDIA GPUs. It keeps the useful shape of
ZML—tagged tensors, XLA compilation, PJRT execution, persistent buffers, and
explicit sharding—while deliberately narrowing the hardware surface and
improving the architecture where NML's products require it.

The model remains ordinary Rust data. NML owns checkpoint loading, graph
construction, compilation, device placement, executable arguments, donated
state, and result buffers; it does not require a general eager runtime or
autograd engine between the model and XLA.

## What You Can Build

| Workload | What NML provides today |
|---|---|
| Transformer inference | Token embeddings, linear layers, gated MLPs, RMSNorm and LayerNorm, RoPE, masks, ordinary and paged attention, persistent KV caches, and token sampling. |
| Long-context and speculative systems | Page-table-owned KV storage, bounded blockwise paged attention, cache updates, truncation, rollback, and replay without rebuilding a dense persistent cache. |
| Dense neural networks | FP32, FP16, and BF16 matrix operations, nonlinear activations, normalization, reductions, sorting, top-k, and explicit-state random generation. |
| Mixture-of-experts models | Portable top-k routing and grouped expert execution on CPU and CUDA, plus Shardy expert partitioning and private grouped Triton projections for SM80 and newer GPUs. |
| Vision and audio models | 1D and 2D convolution, grouped and depthwise convolution, pooling, nearest/linear/bilinear/cubic resizing, FFT/IFFT, and complex tensors. |
| Recurrent and state-space models | Explicit state threading and compiled step or full-sequence Gated DeltaNet graphs. |
| Sharded model experiments | Tagged logical axes, Shardy meshes, tiled parameters and activations, host-to-shard loading, result assembly, manual computations, and typed all-reduce. |
| Checkpoint-backed products | SafeTensors discovery, typed parameter declarations, tied weights, bounded parallel loading, persistent device buffers, and reusable compiled executables. |
| Custom CUDA acceleration | PJRT GPU custom-call registration, lifecycle handlers, automatic capability dispatch, upstream FlashAttention integration, and private Triton kernels. |
| Analytic training experiments | The graph language needed to author an explicit backward computation as another compiled program, without introducing general autograd. The backward graphs themselves are not yet supplied. |

These pieces compose. A model can load sharded SafeTensors parameters, run a
convolutional or transformer front end, maintain a paged KV cache, route tokens
through experts, sample the next token, and reuse the same compiled executable
with fresh buffers.

NML is an acceleration substrate, not a turnkey model-serving service. The
Qwen3 product includes tokenizer, checkpoint, prefill, decode, and generation
integration; request scheduling, continuous batching, network APIs, and server
policy remain application concerns outside the substrate.

## Why NML

### A deliberately narrow hardware contract

NML targets CPU and NVIDIA CUDA. CPU is both the correctness reference and a
performance backend; CUDA is an additive backend rather than a replacement for
it. The product host matrix is Linux x86-64, Linux AArch64, and Apple Silicon
macOS, with CUDA available on supported Linux hosts. Unsupported NVIDIA compute
capabilities fail with a diagnostic instead of silently selecting an invalid
kernel.

### Rust owns the safe product boundary

Shapes, dtypes, tensor graphs, checkpoints, buffers, executables, placement,
and dispatch are Rust APIs. Unsafe PJRT, MLIR, XLA, CUDA, and custom-call ABIs
stay behind narrow internal crates. Backend launch and compiler ownership types
do not leak into the compact `nml` facade.

### XLA remains the compiler

NML lowers its typed graph to StableHLO, applies Shardy placement, compiles with
XLA, and executes through PJRT:

```text
Rust model and tensor program
  -> typed NML graph
  -> StableHLO plus Shardy
  -> XLA CPU or CUDA executable
  -> PJRT buffers and execution
```

This gives portable compiled semantics for every supported device while still
allowing private CUDA kernels where they provide a material advantage.

### Sharding is part of the graph

Logical axis tags and partition metadata survive reshape, transpose,
contraction, attention, and expert routing. NML supports Shardy only; it does
not expose a GSPMD/Shardy selector or maintain two competing sharding models.

### Portable and specialized paths are separate evidence

Portable StableHLO implementations are executable CPU references and CUDA
fallbacks. FlashAttention and Triton are selected only on compatible GPUs.
Compiling an SM80 or SM90 kernel is not reported as runtime evidence until it
has executed on that hardware.

### Persistent state has explicit ownership

Parameters and KV caches are persistent buffers. Donated inputs, output
aliases, cache replacement, repeated execution, and failure cleanup have
explicit contracts rather than relying on incidental reference counting or
hidden mutation.

## Quickstart

NML is built exclusively through Bazel. Install Bazel or Bazelisk, clone the
repository, and run the complete CPU product contracts:

```sh
git clone git@github.com:NarendraPatwardhan/nml.git
cd nml
bazel test --config=cpu //:cpu_contracts
```

The repository keeps Bazel's output root at `../nml-bazel-cache` so expensive
XLA, LLVM, and Rust actions are reused across invocations.

For authenticated BuildBuddy remote execution:

```sh
bb test --config=buildbuddy --config=cpu //:cpu_contracts
```

Build every CUDA device-contract binary and run the hardware-independent CUDA
contracts remotely:

```sh
bb build --config=buildbuddy --config=cuda //:cuda_contract_binaries
bb test --config=buildbuddy --config=cuda \
  //:cuda_remote_contracts //:cuda_package_contracts
```

On a supported local NVIDIA GPU, execute the real device contracts:

```sh
bb test --config=buildbuddy --config=cuda --cache_test_results=no \
  //:cuda_device_contracts
```

The CPU suite is intentionally substantial: it creates four logical CPU
devices so sharded placement and collectives execute numerically instead of
being reduced to metadata checks.

## A Model Is Ordinary Rust Data

NML derives parameter traversal from the user's model structure. Graph methods
remain on an independent graph builder, keeping model data reusable across
prefill, decode, training, and other compiled programs:

```rust
#[derive(nml::ParameterTree)]
struct Linear {
    weight: nml::Parameter,
    bias: Option<nml::Parameter>,
}

#[derive(nml::ParameterTree)]
struct Mlp {
    first: Linear,
    second: Linear,
}

let registry = nml::safetensors::TensorRegistry::from_path("model").unwrap();
let parameters = nml::io::ParameterSet::new(registry);

let first = parameters.view("first");
let second = parameters.view("second");
let shape = nml::Shape::new(nml::DataType::F16, &[4096, 4096]).unwrap();
let model = Mlp {
    first: Linear {
        weight: first.dense("weight", shape, &[]).unwrap(),
        bias: None,
    },
    second: Linear {
        weight: second.dense("weight", shape, &[]).unwrap(),
        bias: None,
    },
};

let mut graph = nml::Graph::new();
let input = graph.input(
    "input",
    nml::Shape::new(nml::DataType::F16, &[1, 4096]).unwrap(),
);
let hidden = graph.linear(input, &model.first.weight, None).unwrap();
let hidden = graph.gelu(hidden).unwrap();
let output = graph.linear(hidden, &model.second.weight, None).unwrap();
let program = graph.finish_named(&[("output".to_owned(), output)]).unwrap();
```

`ParameterSet` resolves and loads physical checkpoint components; `Graph`
constructs programs; `Platform` compiles programs and owns persistent buffers.
The resulting executable validates and binds a loaded parameter's component
manifest once, then accepts fresh activation buffers on each call.

## Current Capability Status

| Area | Current status |
|---|---|
| CPU execution | Product path; real numerical, lifecycle, sharding, collective, and performance contracts run on a four-device Linux CPU topology. |
| CUDA execution | Product path from SM75 upward; portable XLA CUDA is exercised on a GTX 1660 Ti. Unsupported GPUs fail during platform creation. |
| Ordinary attention | Portable CPU/CUDA path complete. FA2 SM80-SM89 and FA3 SM90 paths are built into the CUDA product graph. |
| Paged attention | Portable blockwise CPU/CUDA path complete; Triton optimized paths are built for compatible SM80/SM90 GPUs. |
| MoE | Portable CPU/CUDA execution complete; four-device CPU expert sharding executes numerically; grouped Triton kernels are compiled for SM80 and newer. |
| Distributed execution | Real Shardy placement and collectives are verified on CPU. Multi-GPU CUDA execution remains hardware-deferred. |
| Quantization | W4A16, W8A8, and NVFP4 are designed product goals but have not been implemented. |
| Training | No autograd engine and no supplied analytic backward graph library yet. |

The SM80/SM90 attention and grouped-MoE binaries are present and verified at
the compilation, linking, TTIR, custom-call, and dispatch layers. Their runtime
numerical and performance contracts remain explicitly deferred until matching
GPU hardware is rented; they are not counted as executed capabilities today.

## Supported Platforms

| Host | CPU target | NVIDIA CUDA target | Current runtime evidence |
|---|---:|---:|---|
| Linux x86-64 | Yes | Yes | Four-device CPU and SM75 CUDA |
| Linux AArch64 | Yes | Yes | Host and package-selection contracts; native execution pending |
| macOS Apple Silicon | Yes | No | Host contract present; native execution pending |
| Windows | No | No | Outside the product scope |
| Intel macOS | No | No | Outside the product scope |

CPU and CUDA are independent Bazel settings. `--config=cuda` enables CUDA while
retaining CPU, so one product can discover both PJRT backends without changing
the tensor-program API.

## Project Documents

| Resource | Use it for |
|---|---|
| [`crates/nml`](./crates/nml) | The deliberately compact public Rust facade |
| [`crates/nml-ir`](./crates/nml-ir) | Typed tensor-program construction and StableHLO lowering |
| [`crates/nml-runtime`](./crates/nml-runtime) | Platforms, persistent buffers, executables, arguments, results, and KV-cache ownership |
| [`crates/nml-checkpoint`](./crates/nml-checkpoint) | SafeTensors discovery, model declarations, loading, and graph-facing model construction |
| [`products/serve`](./products/serve) | Qwen model execution and the planned continuous-batching serving control plane |

## Acknowledgements

NML is an opinionated fork of [ZML](https://github.com/zml/zml). It narrows the
hardware and dtype surface, uses Rust as its core language, and makes different
architectural choices for the experiments and products we intend to build.

Most users should use ZML instead. ZML is the mature, stable, open-source
project with a broader hardware ecosystem, model integrations, and an
established community. NML remains an actively developed alternative, and we
continue to read, study, and learn from ZML's design and implementation.
