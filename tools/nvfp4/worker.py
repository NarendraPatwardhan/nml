"""Remote process owner for one restart-observable NVFP4 conversion."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--status", type=Path, required=True)
    result.add_argument("--log", type=Path, required=True)
    result.add_argument("converter_arguments", nargs=argparse.REMAINDER)
    return result


def main() -> int:
    arguments = parser().parse_args()
    converter_arguments = arguments.converter_arguments
    if converter_arguments[:1] == ["--"]:
        converter_arguments = converter_arguments[1:]
    if not converter_arguments:
        raise SystemExit("worker: converter arguments are empty")

    arguments.status.unlink(missing_ok=True)
    with arguments.log.open("ab", buffering=0) as log:
        completed = subprocess.run(
            [sys.executable, "convert.py", *converter_arguments],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    temporary = arguments.status.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump({"exit_code": completed.returncode}, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, arguments.status)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
