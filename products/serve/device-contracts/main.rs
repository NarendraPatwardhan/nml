//! GPT-OSS product acceptance runner.
//!
//! The reusable runner owns process and lease mechanics. This product target
//! owns the model-specific contract names, required mounted inputs, and exact
//! test executables. No GPT-OSS policy enters the substrate runner.

use device_contract_runner::{
    ChildEnvironmentDefinition, ContractDefinition, RunnerDefinition,
};
use std::process::ExitCode;

const MODEL_INPUT: ChildEnvironmentDefinition = ChildEnvironmentDefinition {
    name: "NML_GPT_OSS_MODEL",
    required: true,
};
const GENERATION_FIXTURE_INPUT: ChildEnvironmentDefinition = ChildEnvironmentDefinition {
    name: "NML_GPT_OSS_GENERATION_FIXTURE",
    required: true,
};

const CONTRACTS: &[ContractDefinition] = &[
    ContractDefinition {
        name: "gpt_oss_20b_nvfp4_generation",
        rlocation: env!("NML_GPT_OSS_20B_NVFP4_GENERATION_CONTRACT"),
        arguments: &["--nocapture"],
        environment: &[MODEL_INPUT],
    },
    ContractDefinition {
        name: "gpt_oss_20b_nvfp4_acceptance",
        rlocation: env!("NML_GPT_OSS_20B_NVFP4_ACCEPTANCE_CONTRACT"),
        arguments: &["--nocapture"],
        environment: &[MODEL_INPUT, GENERATION_FIXTURE_INPUT],
    },
];

const RUNNER: RunnerDefinition = RunnerDefinition {
    service: "nml-gpt-oss-device-contracts",
    cuda_runtime_rlocation: env!("NML_CUDA_RUNTIME"),
    contracts: CONTRACTS,
    isolated_environment: &["NML_GPT_OSS_MODEL", "NML_GPT_OSS_GENERATION_FIXTURE"],
};

#[tokio::main]
async fn main() -> ExitCode {
    match device_contract_runner::serve(&RUNNER).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("gpt_oss_device_contracts: {error}");
            ExitCode::FAILURE
        }
    }
}
