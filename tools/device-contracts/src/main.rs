use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    match device_contract_runner::serve().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("run_device_contracts: {error}");
            ExitCode::FAILURE
        }
    }
}
