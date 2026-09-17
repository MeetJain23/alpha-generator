"""Tests for Layer 1: panels, the synthetic generator, the cache and the adapter.

The assertions worth writing here are the ones about bias. A panel that loads
is not the property under test. A panel that cannot quietly drop a bankruptcy,
resurrect a delisted name, or carry a price past the day trading stopped, is.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from alpha.data import cache
from alpha.data.synthetic import SyntheticSpec, generate, write_layout
from alpha.data.types import DTYPE, FIELDS, CostModel, Panel, PanelError
from alpha.data.us_adapter import USAdapter

SPEC = SyntheticSpec(n_days=400, n_instruments=40)


@pytest.fixture()
def dataset():
    return generate(np.random.default_rng(20260917), SPEC)


@pytest.fixture()
def panel(dataset) -> Panel:
    return dataset.panel


def _blank_panel(n_days: int = 6, n_instruments: int = 3) -> dict[str, np.ndarray]:
    """Planes for a minimal panel where everyone is listed throughout."""
    shape = (n_days, n_instruments)
    planes = {name: np.full(shape, 1.0, dtype=DTYPE) for name in FIELDS}
    planes["is_listed"] = np.ones(shape, dtype=DTYPE)
    return planes


def _make_panel(planes: dict[str, np.ndarray]) -> Panel:
    n_days, n_instruments = planes["close"].shape
    return Panel(
        dates=pd.bdate_range("2020-01-01", periods=n_days, name="date"),
        instruments=np.array([f"A{i}" for i in range(n_instruments)], dtype=object),
        fields=planes,
    )


# --------------------------------------------------------------------------
# panel invariants
# --------------------------------------------------------------------------


def test_a_well_formed_panel_validates() -> None:
    _make_panel(_blank_panel()).validate()


def test_dates_must_be_strictly_increasing() -> None:
    planes = _blank_panel()
    bad = Panel(
        dates=pd.DatetimeIndex(["2020-01-02", "2020-01-01"] * 3, name="date"),
        instruments=np.array(["A0", "A1", "A2"], dtype=object),
        fields=planes,
    )
    with pytest.raises(PanelError, match="strictly increasing"):
        bad.validate()


def test_duplicate_instrument_ids_are_rejected() -> None:
    planes = _blank_panel()
    bad = Panel(
        dates=pd.bdate_range("2020-01-01", periods=6, name="date"),
        instruments=np.array(["A0", "A0", "A2"], dtype=object),
        fields=planes,
    )
    with pytest.raises(PanelError, match="unique"):
        bad.validate()


def test_a_missing_field_is_rejected() -> None:
    planes = _blank_panel()
    del planes["vwap"]
    with pytest.raises(PanelError, match="missing fields"):
        _make_panel(planes).validate()


def test_an_unknown_field_is_rejected() -> None:
    """The schema is fixed so an expression valid on one panel is valid on another."""
    planes = _blank_panel()
    planes["book_value"] = np.ones((6, 3), dtype=DTYPE)
    with pytest.raises(PanelError, match="unknown fields"):
        _make_panel(planes).validate()


def test_a_non_float32_plane_is_rejected() -> None:
    planes = _blank_panel()
    planes["close"] = planes["close"].astype(np.float64)
    with pytest.raises(PanelError, match="float32"):
        _make_panel(planes).validate()


def test_an_infinity_is_rejected() -> None:
    """An infinity survives every downstream operator; NaN propagates."""
    planes = _blank_panel()
    close = np.array(planes["close"])
    close[2, 1] = np.inf
    planes["close"] = close
    with pytest.raises(PanelError, match="infinity"):
        _make_panel(planes).validate()


def test_a_relisted_instrument_is_rejected() -> None:
    """Either an id was recycled or a delisting was dropped."""
    planes = _blank_panel(n_days=8)
    listed = np.ones((8, 3), dtype=DTYPE)
    listed[3:5, 1] = 0.0
    planes["is_listed"] = listed
    for name in ("open", "high", "low", "close", "vwap"):
        plane = np.array(planes[name])
        plane[3:5, 1] = np.nan
        planes[name] = plane
    with pytest.raises(PanelError, match="relist"):
        _make_panel(planes).validate()


def test_a_price_after_delisting_is_rejected() -> None:
    """This is forward filling a bankrupt company at its last quote."""
    planes = _blank_panel(n_days=8)
    listed = np.ones((8, 3), dtype=DTYPE)
    listed[5:, 1] = 0.0
    planes["is_listed"] = listed
    for name in ("open", "high", "low", "vwap"):
        plane = np.array(planes[name])
        plane[5:, 1] = np.nan
        planes[name] = plane
    # close is left carrying its last value, which is the bug.
    with pytest.raises(PanelError, match="carried forward"):
        _make_panel(planes).validate()


def test_is_listed_cannot_be_nan() -> None:
    planes = _blank_panel()
    listed = np.array(planes["is_listed"])
    listed[2, 0] = np.nan
    planes["is_listed"] = listed
    with pytest.raises(PanelError, match="never NaN"):
        _make_panel(planes).validate()


def test_a_panel_is_read_only() -> None:
    p = _make_panel(_blank_panel())
    with pytest.raises(ValueError):
        p["close"][0, 0] = 5.0


def test_an_unknown_field_lookup_lists_what_is_there() -> None:
    p = _make_panel(_blank_panel())
    with pytest.raises(KeyError, match="panel carries"):
        p["book_value"]


# --------------------------------------------------------------------------
# head, which the look-ahead check is built on
# --------------------------------------------------------------------------


def test_head_matches_slicing_the_full_panel(panel: Panel) -> None:
    k = 137
    prefix = panel.head(k)
    assert prefix.n_days == k
    assert prefix.dates.equals(panel.dates[:k])
    for name in FIELDS:
        assert np.array_equal(prefix[name], panel[name][:k], equal_nan=True)


def test_head_produces_an_independent_panel(panel: Panel) -> None:
    """A prefix must not be a view, or mutating one would reach the other."""
    prefix = panel.head(10)
    assert not np.shares_memory(prefix["close"], panel["close"])


def test_head_of_everything_is_everything(panel: Panel) -> None:
    assert panel.head(panel.n_days).n_days == panel.n_days


@pytest.mark.parametrize("k", [-1, 10_000])
def test_head_out_of_range_raises(panel: Panel, k: int) -> None:
    with pytest.raises(IndexError):
        panel.head(k)


# --------------------------------------------------------------------------
# costs
# --------------------------------------------------------------------------


def test_cost_rises_with_participation() -> None:
    model = CostModel()
    assert model.trade_cost_bps(0.10) > model.trade_cost_bps(0.01)


def test_cost_at_zero_participation_is_the_flat_terms() -> None:
    model = CostModel(commission_bps=1.0, half_spread_bps=2.5, impact_coef=10.0)
    assert float(model.trade_cost_bps(0.0)) == pytest.approx(3.5)


def test_cost_of_an_unknown_participation_is_unknown() -> None:
    """No volume means no cost estimate, not a free trade."""
    assert np.isnan(CostModel().trade_cost_bps(np.nan))


def test_negative_participation_is_refused() -> None:
    with pytest.raises(ValueError):
        CostModel().trade_cost_bps(-0.1)


# --------------------------------------------------------------------------
# synthetic generator
# --------------------------------------------------------------------------


def test_a_generated_panel_validates(panel: Panel) -> None:
    panel.validate()


def test_generation_is_reproducible_from_its_seed() -> None:
    first = generate(np.random.default_rng(4), SPEC).panel
    second = generate(np.random.default_rng(4), SPEC).panel
    assert np.array_equal(first["close"], second["close"], equal_nan=True)


def test_a_different_seed_gives_a_different_panel() -> None:
    first = generate(np.random.default_rng(4), SPEC).panel
    second = generate(np.random.default_rng(5), SPEC).panel
    assert not np.array_equal(first["close"], second["close"], equal_nan=True)


def test_the_panel_contains_delistings(dataset) -> None:
    """A generator that never kills anything tests nothing that matters."""
    assert len(dataset.delistings) > 0


def test_the_terminal_return_lands_on_the_final_valid_day(dataset) -> None:
    panel = dataset.panel
    for name, (date, terminal) in dataset.delistings.items():
        column = list(panel.instruments).index(name)
        row = list(panel.dates).index(date)
        market = dataset.market_returns[row, column]
        expected = (
            terminal if np.isnan(market) else (1.0 + market) * (1.0 + terminal) - 1.0
        )
        assert panel["returns"][row, column] == pytest.approx(expected, rel=1e-4)


def test_nothing_survives_its_own_delisting(dataset) -> None:
    panel = dataset.panel
    for name, (date, _) in dataset.delistings.items():
        column = list(panel.instruments).index(name)
        row = list(panel.dates).index(date)
        assert np.isnan(panel["close"][row + 1 :, column]).all()
        assert (panel["is_listed"][row + 1 :, column] == 0.0).all()


def test_a_terminal_return_is_never_worse_than_a_total_loss(dataset) -> None:
    for _, terminal in dataset.delistings.values():
        assert terminal >= -1.0


def test_halts_appear_and_leave_gaps(dataset) -> None:
    panel = dataset.panel
    halted = np.nan_to_num(panel["halted"]) == 1.0
    assert halted.any(), "no halts generated, so the min_periods path is untested"
    assert np.isnan(panel["close"][halted]).all()


def test_sector_is_reclassified_at_least_once(dataset) -> None:
    """Or the adapter's point-in-time check has nothing to detect."""
    assert dataset.sector_changes > 0


def test_planted_momentum_does_not_look_forward() -> None:
    """A planted effect that peeked would make every look-ahead test pass."""
    spec = SyntheticSpec(n_days=400, n_instruments=40, momentum_strength=0.05)
    seeded = generate(np.random.default_rng(1), spec).panel
    plain = generate(np.random.default_rng(1), SyntheticSpec(n_days=400, n_instruments=40)).panel
    # The first 250 rows precede any trailing window, so they must be untouched.
    early = slice(0, 250)
    assert np.array_equal(
        seeded["returns"][early], plain["returns"][early], equal_nan=True
    )


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


def test_cache_round_trips_every_plane(panel: Panel, tmp_path) -> None:
    path = cache.write(panel, tmp_path / "panel.npz")
    restored = cache.read(path)
    assert restored.dates.equals(panel.dates)
    assert list(restored.instruments) == list(panel.instruments)
    for name in FIELDS:
        assert np.array_equal(restored[name], panel[name], equal_nan=True)


def test_a_cached_panel_is_actually_memory_mapped(panel: Panel, tmp_path) -> None:
    """np.load(npz, mmap_mode=...) does not map; this must."""
    path = cache.write(panel, tmp_path / "panel.npz")
    assert isinstance(cache.read(path)["close"], np.memmap)


def test_reading_without_mmap_gives_ordinary_arrays(panel: Panel, tmp_path) -> None:
    path = cache.write(panel, tmp_path / "panel.npz")
    assert not isinstance(cache.read(path, mmap=False)["close"], np.memmap)


def test_a_cached_panel_still_validates(panel: Panel, tmp_path) -> None:
    path = cache.write(panel, tmp_path / "panel.npz")
    cache.read(path).validate()


def test_the_cache_is_a_readable_npz(panel: Panel, tmp_path) -> None:
    """The container stays ordinary, whatever this module does with offsets."""
    path = cache.write(panel, tmp_path / "panel.npz")
    with np.load(path) as archive:
        assert np.array_equal(archive["close"], panel["close"], equal_nan=True)


def test_members_are_stored_uncompressed(panel: Panel, tmp_path) -> None:
    path = cache.write(panel, tmp_path / "panel.npz")
    with zipfile.ZipFile(path) as archive:
        assert all(
            info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()
        )


def test_a_cache_from_another_version_is_refused(panel: Panel, tmp_path, monkeypatch) -> None:
    path = cache.write(panel, tmp_path / "panel.npz")
    monkeypatch.setattr(cache, "CACHE_VERSION", cache.CACHE_VERSION + 1)
    with pytest.raises(cache.CacheError, match="version"):
        cache.read(path)


def test_snapshot_id_is_stable_for_the_same_content(panel: Panel, tmp_path) -> None:
    path = cache.write(panel, tmp_path / "panel.npz")
    assert cache.snapshot_id(cache.read(path)) == cache.snapshot_id(panel)


def test_snapshot_id_changes_when_one_number_changes(panel: Panel) -> None:
    """A path and an mtime would not catch a file regenerated in place."""
    planes = {name: np.array(plane) for name, plane in panel.fields.items()}
    planes["close"][0, 0] = np.float32(1234.5)
    mutated = Panel(
        dates=panel.dates, instruments=panel.instruments.copy(), fields=planes
    )
    assert cache.snapshot_id(mutated) != cache.snapshot_id(panel)


def test_reading_a_missing_cache_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        cache.read(tmp_path / "absent.npz")


# --------------------------------------------------------------------------
# adapter
# --------------------------------------------------------------------------


@pytest.fixture()
def source(dataset, tmp_path):
    return write_layout(dataset, tmp_path / "us"), dataset


def test_the_adapter_reconstructs_the_panel_it_was_given(source) -> None:
    """The round trip through parquet must be lossless, or nothing else holds."""
    root, dataset = source
    built = USAdapter(root).panel(dataset.panel.dates[0], dataset.panel.dates[-1])
    assert built.shape == dataset.panel.shape
    for name in FIELDS:
        np.testing.assert_allclose(
            np.nan_to_num(built[name], nan=-9e9),
            np.nan_to_num(dataset.panel[name], nan=-9e9),
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"field {name} did not survive the round trip",
        )


def test_the_adapter_applies_every_terminal_return(source) -> None:
    root, dataset = source
    adapter = USAdapter(root)
    adapter.panel(dataset.panel.dates[0], dataset.panel.dates[-1])
    assert adapter.diagnostics is not None
    assert adapter.diagnostics.delist_returns_applied == len(dataset.delistings)


def test_the_adapter_reports_what_it_measured(source) -> None:
    root, dataset = source
    adapter = USAdapter(root)
    adapter.panel(dataset.panel.dates[0], dataset.panel.dates[-1])
    diagnostics = adapter.diagnostics
    assert diagnostics is not None
    assert diagnostics.n_delisted == len(dataset.delistings)
    assert 0.0 < diagnostics.listed_fraction <= 1.0


def test_a_bar_outside_the_calendar_is_refused(source) -> None:
    """The calendar is authoritative; a bar that never traded is a pipeline bug."""
    root, dataset = source
    sessions = pd.read_parquet(root / "calendar" / "sessions.parquet")
    sessions = sessions.iloc[:-5]
    sessions.to_parquet(root / "calendar" / "sessions.parquet", index=False)
    with pytest.raises(PanelError, match="not trading sessions"):
        USAdapter(root).panel(dataset.panel.dates[0], dataset.panel.dates[-1])


def test_duplicate_rows_are_refused(source) -> None:
    root, dataset = source
    part = next((root / "bars").rglob("*.parquet"))
    frame = pd.read_parquet(part)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_parquet(part, index=False)
    with pytest.raises(PanelError, match="duplicate"):
        USAdapter(root).panel(dataset.panel.dates[0], dataset.panel.dates[-1])


def test_bars_after_a_delisting_date_are_refused(source) -> None:
    """The case where a source pads its rows forward at the last price."""
    root, dataset = source
    delistings = pd.read_parquet(root / "reference" / "delistings.parquet")
    delistings["delist_date"] = dataset.panel.dates[5]
    delistings.to_parquet(root / "reference" / "delistings.parquet", index=False)
    with pytest.raises(PanelError, match="after its delisting"):
        USAdapter(root).panel(dataset.panel.dates[0], dataset.panel.dates[-1])


def test_restated_values_are_refused(source) -> None:
    root, dataset = source
    part = next((root / "bars").rglob("*.parquet"))
    frame = pd.read_parquet(part)
    frame["knowledge_date"] = frame["date"] + pd.Timedelta(days=30)
    frame.to_parquet(part, index=False)
    with pytest.raises(PanelError, match="look-ahead"):
        USAdapter(root).panel(dataset.panel.dates[0], dataset.panel.dates[-1])


def test_an_empty_date_range_is_refused(source) -> None:
    root, _ = source
    with pytest.raises(PanelError, match="no bars"):
        USAdapter(root).panel("1990-01-01", "1990-12-31")


def test_a_backwards_date_range_is_refused(source) -> None:
    root, _ = source
    with pytest.raises(ValueError, match="after end"):
        USAdapter(root).panel("2020-01-01", "2019-01-01")


def test_costs_come_from_the_default_row(source) -> None:
    root, _ = source
    model = USAdapter(root).costs()
    assert model.commission_bps == pytest.approx(1.0)
    assert model.half_spread_bps == pytest.approx(2.5)


def test_a_cost_table_without_a_default_is_refused(source) -> None:
    """A per-instrument table with no fallback prices unknown names at zero."""
    root, _ = source
    path = root / "costs" / "costs.parquet"
    frame = pd.read_parquet(path)
    frame["instrument_id"] = "SYN00000"
    frame.to_parquet(path, index=False)
    with pytest.raises(Exception, match="no default row"):
        USAdapter(root).costs()


def test_the_calendar_is_read_whole(source) -> None:
    root, dataset = source
    assert len(USAdapter(root).calendar()) == dataset.panel.n_days


def test_a_missing_root_is_refused(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        USAdapter(tmp_path / "nothing-here")


def test_the_cache_short_circuits_the_second_load(source, tmp_path) -> None:
    root, dataset = source
    adapter = USAdapter(root, cache_dir=tmp_path / "cache")
    first = adapter.panel(dataset.panel.dates[0], dataset.panel.dates[-1])
    second = adapter.panel(dataset.panel.dates[0], dataset.panel.dates[-1])
    for name in FIELDS:
        assert np.array_equal(second[name], first[name], equal_nan=True)
    assert isinstance(second["close"], np.memmap)


def test_a_sub_range_loads_only_that_range(source) -> None:
    root, dataset = source
    start, end = dataset.panel.dates[50], dataset.panel.dates[150]
    built = USAdapter(root).panel(start, end)
    assert built.dates[0] == start
    assert built.dates[-1] == end
