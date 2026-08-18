"""Canonicalize setuptools source archives when SOURCE_DATE_EPOCH is set."""

from __future__ import annotations

import copy
import gzip
import os
from pathlib import Path
import tarfile


def source_date_epoch() -> int | None:
    value = os.environ.get("SOURCE_DATE_EPOCH")
    if value is None:
        return None
    try:
        epoch = int(value, 10)
    except ValueError as error:
        raise ValueError("SOURCE_DATE_EPOCH must be a non-negative integer") from error
    if epoch < 0:
        raise ValueError("SOURCE_DATE_EPOCH must be a non-negative integer")
    return epoch


def repack_sdist(source_path: Path, destination_path: Path, epoch: int) -> None:
    """Copy an sdist while canonicalizing archive-only metadata."""

    temporary_path = destination_path.with_name(destination_path.name + ".tmp")
    try:
        with tarfile.open(source_path, "r:gz") as source:
            members = sorted(source.getmembers(), key=lambda member: member.name)
            with temporary_path.open("wb") as raw_output:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw_output,
                    compresslevel=9,
                    mtime=epoch,
                ) as compressed_output:
                    with tarfile.open(
                        fileobj=compressed_output,
                        mode="w|",
                        format=tarfile.PAX_FORMAT,
                    ) as destination:
                        for original in members:
                            member = copy.copy(original)
                            member.mtime = epoch
                            member.uid = 0
                            member.gid = 0
                            member.uname = ""
                            member.gname = ""
                            member.pax_headers = {}
                            payload = source.extractfile(original) if original.isfile() else None
                            destination.addfile(member, payload)
        os.replace(temporary_path, destination_path)
    finally:
        temporary_path.unlink(missing_ok=True)
