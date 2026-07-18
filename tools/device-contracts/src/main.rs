use std::process::ExitCode;

use device_contract_runner::{ContractDefinition, RunnerDefinition};

const CONTRACTS: &[ContractDefinition] = &[
    ContractDefinition {
        name: "flash_attention_device_capability",
        rlocation: env!("NML_FLASH_ATTENTION_CAPABILITY_CONTRACT"),
        arguments: &[],
        environment: &[],
    },
    ContractDefinition {
        name: "cuda_runtime",
        rlocation: env!("NML_CUDA_RUNTIME_CONTRACT"),
        arguments: &[],
        environment: &[],
    },
    ContractDefinition {
        name: "linear",
        rlocation: env!("NML_LINEAR_CONTRACT"),
        arguments: &[],
        environment: &[],
    },
    ContractDefinition {
        name: "attention",
        rlocation: env!("NML_ATTENTION_CONTRACT"),
        arguments: &[],
        environment: &[],
    },
    ContractDefinition {
        name: "neural_ops",
        rlocation: env!("NML_NEURAL_OPS_CONTRACT"),
        arguments: &[],
        environment: &[],
    },
    ContractDefinition {
        name: "execution_performance",
        rlocation: env!("NML_EXECUTION_PERFORMANCE_CONTRACT"),
        // The runner invokes the Rust test binary directly, outside Bazel's
        // wrapper. Preserve phase measurements in the captured result.
        arguments: &["--nocapture"],
        environment: &[],
    },
    ContractDefinition {
        name: "nvfp4",
        rlocation: env!("NML_NVFP4_CONTRACT"),
        arguments: &[],
        environment: &[],
    },
];

const RUNNER: RunnerDefinition = RunnerDefinition {
    service: "nml-substrate-device-contracts",
    cuda_runtime_rlocation: env!("NML_CUDA_RUNTIME"),
    contracts: CONTRACTS,
    isolated_environment: &[],
};

#[tokio::main]
async fn main() -> ExitCode {
    match device_contract_runner::serve(&RUNNER).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("run_device_contracts: {error}");
            ExitCode::FAILURE
        }
    }
}
