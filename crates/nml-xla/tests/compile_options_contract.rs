use nml_xla::{Backend, CompileOptions, Error};

#[test]
fn generated_upb_serialization_is_deterministic_and_backend_specific() {
    let cpu = CompileOptions::single_device(0, Backend::Cpu).unwrap();
    let first = cpu.serialize().unwrap();
    let second = cpu.serialize().unwrap();
    assert_eq!(first, second);
    assert!(!first.is_empty());

    let cuda = CompileOptions::single_device(0, Backend::Cuda)
        .unwrap()
        .serialize()
        .unwrap();
    assert_ne!(first, cuda);
}

#[test]
fn topology_is_rejected_before_upb_or_pjrt() {
    assert_eq!(
        CompileOptions::new(0, 1, Vec::new(), Backend::Cpu),
        Err(Error::ZeroTopology)
    );
    assert_eq!(
        CompileOptions::new(2, 2, vec![0, 1], Backend::Cpu),
        Err(Error::DeviceCount {
            expected: 4,
            actual: 2
        })
    );
    assert_eq!(
        CompileOptions::single_device(-1, Backend::Cpu),
        Err(Error::InvalidDeviceId { index: 0, id: -1 })
    );
    assert_eq!(
        CompileOptions::new(i32::MAX as u32 + 1, 1, Vec::new(), Backend::Cpu,),
        Err(Error::TopologyOverflow)
    );
}
