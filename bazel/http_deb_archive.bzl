"""Focused hermetic Debian archive repository rule for CUDA runtime files.

NML only needs ordinary payload extraction from two pinned zlib Debian
packages. This keeps ZML's BSD-tar-based, host-independent extraction model
without importing its unrelated patch/auth/repository-overlay surface.
"""

load("@bazel_lib//lib:repo_utils.bzl", "repo_utils")

def _host_bsdtar_label(rctx):
    platform = repo_utils.platform(rctx)
    binary = "tar.exe" if platform.startswith("windows_") else "tar"
    return Label("@bsd_tar_toolchains_{}//:{}".format(platform, binary))

def _http_deb_archive_impl(rctx):
    archive = "package.deb"
    rctx.download(
        output = archive,
        sha256 = rctx.attr.sha256,
        url = rctx.attr.urls,
    )

    bsdtar = rctx.path(_host_bsdtar_label(rctx))
    result = rctx.execute([bsdtar, "-xf", archive, "--include=data.tar.*"])
    if result.return_code:
        fail("failed to extract data archive from {}:\n{}".format(rctx.name, result.stderr))

    payload = None
    for extension in ["zst", "xz", "gz"]:
        candidate = "data.tar.{}".format(extension)
        if rctx.path(candidate).exists:
            payload = candidate
            break
    if not payload:
        fail("{} contains no supported data.tar payload".format(rctx.name))

    result = rctx.execute([bsdtar, "-xf", payload])
    if result.return_code:
        fail("failed to extract payload from {}:\n{}".format(rctx.name, result.stderr))

    rctx.delete(archive)
    rctx.delete(payload)
    rctx.file("BUILD.bazel", rctx.attr.build_file_content)

http_deb_archive = repository_rule(
    implementation = _http_deb_archive_impl,
    attrs = {
        "build_file_content": attr.string(mandatory = True),
        "sha256": attr.string(mandatory = True),
        "urls": attr.string_list(mandatory = True),
    },
)
