//! End-to-end contract for the packaged CPU PJRT runtime.

#[test]
fn packaged_cpu_plugin_creates_a_real_cpu_client() {
    let plugin = nml_pjrt_cpu::load().expect("packaged CPU PJRT plugin must load and initialize");
    let version = plugin.version();
    assert_eq!(version.major, 0, "unexpected PJRT major version");

    let client = plugin
        .create_client()
        .expect("CPU PJRT client creation must succeed");
    let platform = client
        .platform_name()
        .expect("CPU platform name must be queryable");
    assert_eq!(platform.to_ascii_lowercase(), "cpu");
    assert!(
        client
            .device_count()
            .expect("CPU devices must be enumerable")
            > 0,
        "CPU PJRT must expose at least one addressable host device"
    );
}
