#!/usr/bin/env python3
"""Build the project sdist twice and require byte-for-byte identity."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def build(output: Path, environment: dict[str, str]) -> Path:
    output.mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(output)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        raise SystemExit(result.returncode)
    artifacts = list(output.glob("*.tar.gz"))
    if len(artifacts) != 1:
        raise SystemExit(f"expected one sdist in {output}, found {len(artifacts)}")
    return artifacts[0]


def main() -> int:
    environment = dict(os.environ)
    environment.setdefault("SOURCE_DATE_EPOCH", "1700000000")
    try:
        epoch = int(environment["SOURCE_DATE_EPOCH"], 10)
    except ValueError as error:
        raise SystemExit("SOURCE_DATE_EPOCH must be a non-negative integer") from error
    if epoch < 0:
        raise SystemExit("SOURCE_DATE_EPOCH must be a non-negative integer")

    with tempfile.TemporaryDirectory(prefix="python-sound-generator-repro-") as root:
        first = build(Path(root, "first"), environment)
        second = build(Path(root, "second"), environment)
        first_bytes = first.read_bytes()
        second_bytes = second.read_bytes()
        if first_bytes != second_bytes:
            raise SystemExit("source distributions are not byte-identical")
        with tarfile.open(first, "r:gz") as archive:
            members = archive.getmembers()
        if members != sorted(members, key=lambda member: member.name):
            raise SystemExit("source-distribution entries are not sorted")
        if {member.mtime for member in members} != {epoch}:
            raise SystemExit("source-distribution timestamps are not canonical")
        owners = {
            (member.uid, member.gid, member.uname, member.gname)
            for member in members
        }
        if owners != {(0, 0, "", "")}:
            raise SystemExit("source-distribution ownership is not canonical")
        digest = hashlib.sha256(first_bytes).hexdigest()
        print(f"PASS reproducible sdist sha256={digest} entries={len(members)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
