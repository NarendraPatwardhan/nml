"""Explicit runfiles grouping for incrementally distributable OCI binaries."""

load(
    "@rules_runfiles_group//runfiles_group:providers.bzl",
    "RunfilesGroupInfo",
    "RunfilesGroupMetadataInfo",
)

def _oci_layered_binary_impl(ctx):
    target = ctx.attr.binary
    default = target[DefaultInfo]
    source_executable = ctx.executable.binary
    executable = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.symlink(
        output = executable,
        target_file = source_executable,
        is_executable = True,
    )
    runfiles = default.default_runfiles

    # A data executable links product code and therefore changes at application
    # cadence even when its CUDA/PJRT closure does not. Give each declared
    # executable an independent group: changing one acceptance contract then
    # invalidates one small upper layer rather than a merged multi-gigabyte
    # runtime layer. DefaultInfo.files is intentional here. Pulling the data
    # target's complete runfiles would duplicate the shared runtime closure in
    # every application layer.
    application_file_set = {}
    application_groups = {}
    application_metadata = {}
    for index, application in enumerate(ctx.attr.application_runfiles):
        files = application[DefaultInfo].files.to_list()
        for file in files:
            if file in application_file_set:
                fail("application runfile {} belongs to more than one OCI layer".format(file.path))
            application_file_set[file] = None
        group = "nml_oci#application_{}".format(index)
        application_groups[group] = ctx.runfiles(files = files)
        application_metadata[group] = {
            "rank": 1,
            "do_not_merge": True,
        }

    # rules_rust includes the executable itself in default runfiles. Keep that
    # artifact and its runfiles symlink out of the dependency group so changing
    # one Rust source file does not invalidate the multi-gigabyte CUDA layer.
    dependency_files = depset([
        file
        for file in runfiles.files.to_list()
        if file != source_executable and file not in application_file_set
    ])
    dependency_symlinks = depset([
        entry
        for entry in runfiles.symlinks.to_list()
        if entry.target_file != source_executable and entry.target_file not in application_file_set
    ])
    dependency_root_symlinks = depset([
        entry
        for entry in runfiles.root_symlinks.to_list()
        if entry.target_file != source_executable and entry.target_file not in application_file_set
    ])
    if runfiles.empty_filenames.to_list():
        fail("oci_layered_binary does not support empty runfile sentinels")
    dependencies = ctx.runfiles(
        transitive_files = dependency_files,
        symlinks = dependency_symlinks,
        root_symlinks = dependency_root_symlinks,
    )
    groups = {
        "nml_oci#dependencies": dependencies,
    }
    groups.update(application_groups)
    metadata = {
        # Lower ranks are emitted first and are therefore reusable by every
        # later manifest that carries the same CUDA/PJRT contract closure.
        "nml_oci#dependencies": {
            "rank": 0,
            "do_not_merge": True,
        },
    }
    metadata.update(application_metadata)

    providers = [
        DefaultInfo(
            executable = executable,
            files = depset([executable]),
            runfiles = runfiles,
        ),
        RunfilesGroupInfo(**groups),
        RunfilesGroupMetadataInfo(groups = metadata),
    ]
    # rules_rust uses RunEnvironmentInfo for environment entries declared on
    # the binary. The wrapper must be transparent to image_from_binary; losing
    # this provider would silently drop runtime selection such as
    # NML_CUDA_RUNTIME_RLOCATION from the OCI configuration.
    if RunEnvironmentInfo in target:
        providers.append(target[RunEnvironmentInfo])
    return providers

oci_layered_binary = rule(
    implementation = _oci_layered_binary_impl,
    attrs = {
        "binary": attr.label(
            executable = True,
            cfg = "target",
            mandatory = True,
        ),
        "application_runfiles": attr.label_list(
            allow_files = False,
            doc = "Data executables isolated into independently reusable upper OCI layers.",
        ),
    },
    executable = True,
)
