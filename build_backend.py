"""Setuptools backend with reproducible SOURCE_DATE_EPOCH source archives."""

from __future__ import annotations

from pathlib import Path
import tempfile

from setuptools import build_meta as _setuptools

from reproducible_sdist import repack_sdist, source_date_epoch


build_wheel = _setuptools.build_wheel
build_editable = _setuptools.build_editable
get_requires_for_build_wheel = _setuptools.get_requires_for_build_wheel
get_requires_for_build_editable = _setuptools.get_requires_for_build_editable
prepare_metadata_for_build_wheel = _setuptools.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = (
    _setuptools.prepare_metadata_for_build_editable
)


def get_requires_for_build_sdist(config_settings=None):
    return _setuptools.get_requires_for_build_sdist(config_settings)


def build_sdist(sdist_directory, config_settings=None):
    epoch = source_date_epoch()
    if epoch is None:
        return _setuptools.build_sdist(sdist_directory, config_settings)

    output_directory = Path(sdist_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="python-sound-generator-sdist-", dir=output_directory
    ) as temporary_directory:
        filename = _setuptools.build_sdist(temporary_directory, config_settings)
        source_path = Path(temporary_directory, filename)
        destination_path = output_directory / filename
        repack_sdist(source_path, destination_path, epoch)
    return filename
