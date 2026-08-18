import gzip
import io
from pathlib import Path
import tarfile

import pytest

import reproducible_sdist


def _write_fixture(path: Path, mtime: float) -> None:
    with path.open("wb") as raw_output:
        with gzip.GzipFile(filename="changing-name.tar", mode="wb", fileobj=raw_output) as gz:
            with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                directory = tarfile.TarInfo("example-1.0")
                directory.type = tarfile.DIRTYPE
                directory.mode = 0o755
                directory.mtime = mtime
                directory.uid = 123
                directory.gid = 456
                directory.uname = "builder"
                directory.gname = "builder"
                archive.addfile(directory)

                data = b"payload remains byte-identical\n"
                member = tarfile.TarInfo("example-1.0/data.txt")
                member.size = len(data)
                member.mode = 0o640
                member.mtime = mtime
                member.uid = 123
                member.gid = 456
                member.uname = "builder"
                member.gname = "builder"
                archive.addfile(member, io.BytesIO(data))


def test_repack_sdist_is_reproducible_without_changing_payload(tmp_path):
    source_one = tmp_path / "source-one.tar.gz"
    source_two = tmp_path / "source-two.tar.gz"
    output_one = tmp_path / "output-one.tar.gz"
    output_two = tmp_path / "output-two.tar.gz"
    _write_fixture(source_one, 1000.25)
    _write_fixture(source_two, 2000.75)

    reproducible_sdist.repack_sdist(source_one, output_one, 123456789)
    reproducible_sdist.repack_sdist(source_two, output_two, 123456789)

    assert output_one.read_bytes() == output_two.read_bytes()
    with tarfile.open(output_one, "r:gz") as archive:
        assert [member.name for member in archive] == [
            "example-1.0",
            "example-1.0/data.txt",
        ]
        member = archive.getmember("example-1.0/data.txt")
        assert member.mtime == 123456789
        assert (member.uid, member.gid, member.uname, member.gname) == (0, 0, "", "")
        assert member.mode == 0o640
        assert archive.extractfile(member).read() == b"payload remains byte-identical\n"


@pytest.mark.parametrize("value", ["not-a-number", "-1"])
def test_source_date_epoch_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", value)
    with pytest.raises(ValueError, match="non-negative integer"):
        reproducible_sdist.source_date_epoch()
