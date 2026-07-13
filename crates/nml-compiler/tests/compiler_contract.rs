use nml_compiler::{Error, negotiate_stablehlo_version};
use nml_pjrt::StableHloVersion;

#[test]
fn stablehlo_negotiation_enforces_the_compiler_range() {
    let fallback = negotiate_stablehlo_version(None).unwrap();
    assert_eq!(fallback.split('.').count(), 3);

    let future = negotiate_stablehlo_version(Some(StableHloVersion {
        major: i64::MAX,
        minor: 0,
        patch: 0,
    }))
    .unwrap();
    assert_ne!(future, format!("{}.0.0", i64::MAX));

    assert!(matches!(
        negotiate_stablehlo_version(Some(StableHloVersion {
            major: -1,
            minor: 0,
            patch: 0,
        })),
        Err(Error::UnsupportedStableHloVersion { .. })
    ));
}
