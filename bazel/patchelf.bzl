"""Hermetic ELF surgery used by the retained CUDA runtime packaging.

The rule is intentionally the same narrow operation ZML uses: copy an input
ELF file, make the copy writable, then apply explicit SONAME, NEEDED, RPATH,
and dynamic-symbol edits with the Bzlmod-provided patchelf executable. Source
artifacts are never mutated in place.
"""

def _patchelf_impl(ctx):
    output_name = ctx.attr.soname or ctx.file.src.basename
    output = ctx.actions.declare_file("{}/{}".format(ctx.attr.name, output_name))
    renamed_symbols = ctx.actions.declare_file("{}.rename.txt".format(ctx.label.name))
    ctx.actions.write(
        renamed_symbols,
        "\n".join([
            "{} {}".format(old, new)
            for old, new in ctx.attr.rename_dynamic_symbols.items()
        ]),
    )

    commands = [
        "set -e",
        'cp -f "$2" "$3"',
        'chmod +w "$3"',
    ]
    if ctx.attr.soname:
        commands.append('"$1" --set-soname \'{}\' "$3"'.format(ctx.attr.soname))
    for library in ctx.attr.add_needed:
        commands.append('"$1" --add-needed \'{}\' "$3"'.format(library))
    for old, new in ctx.attr.replace_needed.items():
        commands.append('"$1" --replace-needed \'{}\' \'{}\' "$3"'.format(old, new))
    if ctx.attr.set_rpath:
        commands.append('"$1" --set-rpath \'{}\' --force-rpath "$3"'.format(ctx.attr.set_rpath))
    if ctx.attr.rename_dynamic_symbols:
        commands.append('"$1" --rename-dynamic-symbols \'{}\' "$3"'.format(renamed_symbols.path))

    ctx.actions.run_shell(
        inputs = [ctx.file.src, renamed_symbols],
        outputs = [output],
        arguments = [ctx.executable._patchelf.path, ctx.file.src.path, output.path],
        command = "\n".join(commands),
        tools = [ctx.executable._patchelf],
    )
    return [DefaultInfo(files = depset([output]))]

patchelf = rule(
    implementation = _patchelf_impl,
    attrs = {
        "src": attr.label(allow_single_file = True, mandatory = True),
        "soname": attr.string(),
        "add_needed": attr.string_list(),
        "replace_needed": attr.string_dict(),
        "rename_dynamic_symbols": attr.string_dict(),
        "set_rpath": attr.string(),
        "_patchelf": attr.label(
            default = "@patchelf",
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
    },
)
