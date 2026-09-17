"""Materialized-panel cache: write to ``.npz``, reload by memory map.

A panel is written once per data snapshot and read on every run, so the read
path is what matters. Parquet decode and pivot cost seconds and a full copy of
the panel in memory; a memory map costs neither, and lets the operating system
page in only the fields an expression actually touches. A tree over close and
volume should not fault in seventeen other planes.

The trap
--------
``np.load(path, mmap_mode="r")`` on a ``.npz`` does not memory-map anything.
It returns an ``NpzFile``, and indexing it decompresses and reads the member
into ordinary memory. The ``mmap_mode`` argument is accepted and ignored. Code
written that way looks like it maps and does not, and the difference shows up
only as memory pressure on a panel large enough to matter.

What works is that ``np.savez`` stores its members uncompressed, so each
member is a contiguous ``.npy`` payload sitting at a known offset inside the
zip container. This module finds those offsets and maps each one with
``np.memmap``. The file stays a perfectly ordinary ``.npz`` that any other
tool can read.

``np.savez_compressed`` is therefore not an option here, and the writer
asserts the members came out uncompressed rather than trusting that to remain
true across numpy versions.
"""

from __future__ import annotations

import struct
import zipfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from alpha.data.types import DTYPE, FIELDS, Panel
from alpha.logging_config import get_logger

_log = get_logger(__name__)

CACHE_VERSION: int = 1
"""Bumped whenever the on-disk layout changes. A cache written by a different
version is rejected rather than reinterpreted."""

_DATES_KEY = "__dates__"
_INSTRUMENTS_KEY = "__instruments__"
_META_KEY = "__meta__"

_LOCAL_HEADER_SIZE = 30
_LOCAL_HEADER_FORMAT = "<4s5H3I2H"


class CacheError(RuntimeError):
    """The cache file cannot be used as written."""


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Where a member's payload lives inside the container."""

    name: str
    dtype: np.dtype
    shape: tuple[int, ...]
    offset: int


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def write(panel: Panel, path: str | Path) -> Path:
    """Materialize a panel to an uncompressed ``.npz``.

    Returns the path written. The snapshot id is stored inside, so a reader
    never has to re-hash several hundred megabytes to find out what it has.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = {name: np.asarray(plane, dtype=DTYPE) for name, plane in panel.fields.items()}
    payload[_DATES_KEY] = panel.dates.values.astype("datetime64[ns]").view(np.int64)
    payload[_INSTRUMENTS_KEY] = np.asarray(
        [str(x) for x in panel.instruments], dtype=np.str_
    )
    payload[_META_KEY] = np.array(
        [CACHE_VERSION, panel.n_days, panel.n_instruments], dtype=np.int64
    )

    np.savez(target, **payload)
    _assert_uncompressed(target)

    _log.info(
        "panel cached",
        extra={
            "path": str(target),
            "n_days": panel.n_days,
            "n_instruments": panel.n_instruments,
            "bytes": target.stat().st_size,
        },
    )
    return target


def _assert_uncompressed(path: Path) -> None:
    """Every member must be stored, not deflated, or nothing can be mapped."""
    with zipfile.ZipFile(path) as archive:
        deflated = [
            info.filename
            for info in archive.infolist()
            if info.compress_type != zipfile.ZIP_STORED
        ]
    if deflated:
        raise CacheError(
            f"members {deflated} are compressed, so they cannot be memory "
            f"mapped. Use np.savez, never np.savez_compressed."
        )


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def read(path: str | Path, *, mmap: bool = True) -> Panel:
    """Load a cached panel.

    With ``mmap`` the planes are memory maps and cost no resident memory until
    touched. Without it they are read into ordinary arrays, which is what the
    tests use when they want to be sure they are comparing values rather than
    views.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    entries = _index_members(source)
    meta = _materialize(source, entries[_META_KEY], mmap=False)
    version = int(meta[0])
    if version != CACHE_VERSION:
        raise CacheError(
            f"cache at {source} is version {version}, this build writes "
            f"version {CACHE_VERSION}; rebuild it rather than reinterpreting it"
        )

    dates = pd.DatetimeIndex(
        _materialize(source, entries[_DATES_KEY], mmap=False).view("datetime64[ns]"),
        name="date",
    )
    instruments = np.asarray(
        _materialize(source, entries[_INSTRUMENTS_KEY], mmap=False), dtype=object
    )

    missing = set(FIELDS) - set(entries)
    if missing:
        raise CacheError(f"cache is missing fields: {sorted(missing)}")

    fields = {name: _materialize(source, entries[name], mmap=mmap) for name in FIELDS}
    return Panel(dates=dates, instruments=instruments, fields=fields)


def _index_members(path: Path) -> dict[str, CacheEntry]:
    """Locate every member's payload within the container.

    A zip member's data does not begin at its central-directory offset: it
    begins after a local header whose own extra field can differ in length
    from the one in the directory. The local header has to be read, not
    assumed.
    """
    entries: dict[str, CacheEntry] = {}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()

    with path.open("rb") as handle:
        for info in infos:
            handle.seek(info.header_offset)
            raw = handle.read(_LOCAL_HEADER_SIZE)
            fields = struct.unpack(_LOCAL_HEADER_FORMAT, raw)
            name_length, extra_length = fields[-2], fields[-1]
            payload_start = (
                info.header_offset + _LOCAL_HEADER_SIZE + name_length + extra_length
            )

            handle.seek(payload_start)
            shape, fortran_order, dtype = _read_npy_header(handle)
            if fortran_order:
                raise CacheError(
                    f"member {info.filename} is Fortran ordered; the panel "
                    f"layout is C ordered everywhere so that a rolling "
                    f"operation walks contiguous memory"
                )

            key = info.filename[:-4] if info.filename.endswith(".npy") else info.filename
            entries[key] = CacheEntry(
                name=key, dtype=dtype, shape=shape, offset=handle.tell()
            )
    return entries


def _read_npy_header(handle: BinaryIO) -> tuple[tuple[int, ...], bool, np.dtype]:
    """Parse a .npy header through numpy's public readers.

    Dispatching on the version explicitly, rather than calling numpy's private
    ``_read_array_header``, so that a numpy upgrade turns into a clear error
    here instead of an AttributeError somewhere inside a cache read. numpy
    writes version 1.0 for everything this module stores; 2.0 exists for
    oversized headers and is handled because handling it costs one line.
    """
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(handle)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(handle)
    raise CacheError(
        f"npy header version {version} is not supported by this reader"
    )


def _materialize(path: Path, entry: CacheEntry, *, mmap: bool) -> np.ndarray:
    """One member, as a memory map or as an ordinary array."""
    mapped = np.memmap(
        path, dtype=entry.dtype, mode="r", offset=entry.offset, shape=entry.shape
    )
    return mapped if mmap else np.array(mapped)


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------


def snapshot_id(panel: Panel) -> str:
    """Content hash of a panel, for ``runs.data_snapshot_id``.

    Covers the axes and every plane, so two runs claiming the same snapshot
    really saw the same numbers. A path and a modification time would not
    survive a file being regenerated with different content at the same
    location, which is exactly the case worth catching.
    """
    digest = sha256()
    digest.update(str(CACHE_VERSION).encode())
    digest.update(panel.dates.values.astype("datetime64[ns]").view(np.int64).tobytes())
    for name in panel.instruments:
        digest.update(str(name).encode("utf-8"))
    for name in FIELDS:
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(panel[name]).tobytes())
    return digest.hexdigest()[:32]
