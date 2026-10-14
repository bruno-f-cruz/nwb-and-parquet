"""Compare two reader layers over the same NWB Zarr v3 store.

NWBZarrIO (hdmf-zarr) reconstructs PyNWB objects. zarr opens arrays and JSON
attributes. Both use the Python Zarr implementation underneath, so these are
reader-layer consistency checks, not independent codec implementations.
"""

import numpy as np
import pandas as pd
import pytest
import zarr
from hdmf.common import DynamicTable
from hdmf_zarr import NWBZarrIO
from pynwb import TimeSeries, validate

from .helpers import (nwb_file, nwb_roundtrip, open_raw_zarr,
                      parquet_roundtrip, read_nwb, write_nwb, zarr_reference)


@pytest.mark.parametrize("dtype", ["int8", "int64", "uint64", "float32", "float64", "bool"])
def test_zarr_numeric_arrays_match_nwb_reader_and_source(tmp_path, dtype):
    """Given a numeric DataFrame, compare raw Zarr and reconstructed NWB columns.

    Expected: source, zarr.Array, and NWB VectorData have identical values,
    dtype, and shape. Integer ids also agree and NWB schema validation passes.
    Why: Zarr v3 zarr.json records dtype, shape, and chunk layout;
    HDMF adds DynamicTable semantics without changing the scalar payload.
    Parquet uses physical types plus logical integer annotations, whereas NWB
    columns are typed arrays. Neither requires lossy float conversion here.
    Specs:
    https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html
    https://hdmf-common-schema.readthedocs.io/en/stable/format.html
    """
    source = pd.DataFrame({"value": np.array([0, 1, 1], dtype=dtype)},
                          index=pd.Index([5, 8, 11], name="id"))
    path = tmp_path / "source.nwb.zarr"
    recovered = nwb_roundtrip(source, path, backend="zarr")
    pd.testing.assert_frame_equal(source, recovered)
    root = open_raw_zarr(path, mode="r")
    raw = root["processing/comparison/source"]
    assert root.attrs["nwb_version"]
    assert raw.attrs["neurodata_type"] == "DynamicTable"
    with NWBZarrIO(str(path), "r") as io:
        table = io.read().processing["comparison"]["source"]
        for name, expected in [("id", source.index.to_numpy()),
                               ("value", source.value.to_numpy())]:
            high = table.id.data if name == "id" else table[name].data
            assert raw[name].shape == high.shape == expected.shape
            assert raw[name].dtype == high.dtype == expected.dtype
            np.testing.assert_array_equal(raw[name][:], high[:], strict=True)
            np.testing.assert_array_equal(raw[name][:], expected, strict=True)


def test_zarr_large_uint64_nan_and_signed_zero_match_nwb_reader(tmp_path):
    """Given uint64 extremes and float special values, compare both reader layers.

    Expected: no integer rounding; NaN, infinities, and negative zero survive.
    Why: dtype-preserving chunk storage does not route uint64 through float64.
    Zarr fill_value describes unwritten chunks; it is not a pandas null bitmap.
    Parquet has distinct null definition levels, even for floating columns.
    Specs:
    https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html
    https://parquet.apache.org/docs/file-format/nulls/
    """
    source = pd.DataFrame({"integer": np.array([0, 2**53 + 1, 2**64 - 1, 1], dtype="uint64"),
                          "floating": [np.nan, np.inf, -np.inf, -0.0]})
    path = tmp_path / "special.nwb.zarr"
    recovered = nwb_roundtrip(source, path, backend="zarr")
    raw = open_raw_zarr(path, mode="r")["processing/comparison/source"]
    for name in source:
        np.testing.assert_array_equal(raw[name][:], recovered[name].to_numpy(), strict=True)
        np.testing.assert_array_equal(raw[name][:], source[name].to_numpy(), strict=True)
    assert np.signbit(raw["floating"][-1])
    assert np.signbit(recovered.floating.iloc[-1])


def test_zarr_unicode_codec_values_match_nwb_reader(tmp_path):
    """Given accented, CJK, and empty strings, compare decoded text in both readers.

    Expected: decoded labels equal the source, but the array
    dtype in v3 differs: raw StringDType versus HDMF's object-dtype string view.
    Why: the HDMF-Zarr mapping uses a variable-length UTF-8 codec for strings.
    Parquet annotates BYTE_ARRAY as STRING; Python object/string dtype is a
    reader choice, so this checks text values rather than requiring that dtype.
    Specs: https://hdmf-zarr.readthedocs.io/en/stable/hdmf_zarr.html
    https://parquet.apache.org/docs/file-format/types/logicaltypes/
    """
    source = pd.DataFrame({"label": pd.Series(["neurón", "神経", ""], dtype=object)})
    path = tmp_path / "text.nwb.zarr"
    recovered = nwb_roundtrip(source, path, backend="zarr")
    raw = open_raw_zarr(path, mode="r")["processing/comparison/source/label"]
    with NWBZarrIO(str(path), "r") as io:
        column = io.read().processing["comparison"]["source"]["label"]
        assert raw.shape == column.data.shape
        assert raw.dtype == np.dtypes.StringDType()
        assert column.data.dtype == np.dtype(object)
        assert raw[:].tolist() == column.data[:].tolist() == source.label.tolist()
    assert recovered.label.tolist() == source.label.tolist()


def test_zarr_ragged_offsets_reconstruct_the_same_rows_as_nwb(tmp_path):
    """Given ragged cells, reconstruct rows manually from raw Zarr arrays.

    Expected: raw flattened values and exclusive cumulative endpoints match
    NWB VectorData/VectorIndex, and slicing them reproduces every source row.
    Why: HDMF VectorIndex stores one end position per row, including repeated
    endpoints for empty rows. Raw zarr does not perform this reconstruction.
    Parquet LIST uses a nested schema and repetition/definition levels instead.
    The _REFERENCE wrapper stores source/path only and resolves the target.
    The target dataset's own object_id still matches the reconstructed object.
    Specs: https://hdmf-common-schema.readthedocs.io/en/stable/format.html
    https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#lists
    """
    source = pd.DataFrame({"spikes": [[0.1, 0.2], [], [0.5]]})
    path = tmp_path / "ragged.nwb.zarr"
    recovered = nwb_roundtrip(source, path, backend="zarr", ragged=("spikes",))
    raw = open_raw_zarr(path, mode="r")["processing/comparison/source"]
    values, ends = raw["spikes"][:], raw["spikes_index"][:]
    np.testing.assert_array_equal(ends, [2, 2, 3])
    with NWBZarrIO(str(path), "r") as io:
        index = io.read().processing["comparison"]["source"]["spikes"]
        assert raw["spikes"].shape == index.target.data.shape
        assert raw["spikes_index"].shape == index.data.shape
        np.testing.assert_array_equal(values, index.target.data[:], strict=True)
        np.testing.assert_array_equal(ends, index.data[:], strict=True)
        target = zarr_reference(raw["spikes_index"].attrs["target"])
        assert target["path"] == "/processing/comparison/source/spikes"
        assert target["source"] == "."
        assert "object_id" not in target
        assert raw["spikes"].attrs["object_id"] == index.target.object_id
    starts = np.concatenate(([0], ends[:-1])).astype("int64")
    for row, (start, end) in enumerate(zip(starts, ends, strict=True)):
        np.testing.assert_array_equal(values[start:int(end)], source.spikes.iloc[row])
        np.testing.assert_array_equal(values[start:int(end)], recovered.spikes.iloc[row])


def test_zarr_matrix_shape_matches_nwb_reader(tmp_path):
    """Given equal-length list cells, compare the default NWB matrix representation.

    Expected: both readers expose a (2, 2) payload without a VectorIndex;
    NWB DataFrame rows match the source lists numerically.
    Why: VectorData may have trailing dimensions. The default HDMF inference
    chooses a matrix here; a ragged hint would instead flatten and add endpoints.
    This is an adapter choice, not a Zarr versus HDF5 value discrepancy.
    Specs: https://hdmf-common-schema.readthedocs.io/en/stable/format.html
    """
    source = pd.DataFrame({"samples": [[1.0, 2.0], [3.0, 4.0]]})
    path = tmp_path / "matrix.nwb.zarr"
    recovered = nwb_roundtrip(source, path, backend="zarr")
    raw = open_raw_zarr(path, mode="r")["processing/comparison/source"]
    assert "samples_index" not in raw
    with NWBZarrIO(str(path), "r") as io:
        data = io.read().processing["comparison"]["source"]["samples"].data
        assert raw["samples"].shape == data.shape == (2, 2)
        np.testing.assert_array_equal(raw["samples"][:], data[:], strict=True)
    for before, after in zip(source.samples, recovered.samples, strict=True):
        np.testing.assert_array_equal(before, after)


def test_zarr_categorical_payload_agrees_but_does_not_restore_schema(tmp_path):
    """Given ordered categories with an unused label, compare payload and schema.

    Expected: both Zarr reader layers keep labels but neither restores category
    ordering or the unused label; Parquet via Arrow retains pandas category dtype.
    Why: our HDMF adapter emits plain VectorData, not an EnumData representation
    or a dtype sidecar. Parquet dictionary encoding alone also does not guarantee
    pandas semantics; Arrow's stored schema/pandas metadata enables this recovery.
    Specs: https://hdmf-common-schema.readthedocs.io/en/stable/format.html
    https://arrow.apache.org/docs/python/pandas.html#categorical-types
    """
    source = pd.DataFrame({"condition": pd.Categorical(
        ["drug", "control"], categories=["drug", "control", "unused"], ordered=True)})
    path = tmp_path / "categories.nwb.zarr"
    recovered = nwb_roundtrip(source, path, backend="zarr")
    raw = open_raw_zarr(path, mode="r")["processing/comparison/source/condition"]
    assert raw[:].tolist() == recovered.condition.tolist() == source.condition.tolist()
    assert not isinstance(recovered.condition.dtype, pd.CategoricalDtype)
    pd.testing.assert_frame_equal(source, parquet_roundtrip(source, tmp_path / "source.parquet"))


def test_zarr_timeseries_data_timestamps_and_units_match_nwb_reader(tmp_path):
    """Given a DataFrame with voltage units, explicitly build a TimeSeries.

    Expected: raw arrays and their unit attrs equal the NWB reader's payload
    and unit properties; schema validation succeeds and source values survive.
    Why: NWB defines data units and seconds for timestamps. Zarr retains these
    as attributes but does not itself interpret scientific units. Parquet can
    store custom metadata; it has no built-in NWB TimeSeries object type.
    Specs: https://nwb-schema.readthedocs.io/en/stable/format.html#timeseries
    https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html
    """
    source = pd.DataFrame({"time": [0.0, 0.1], "voltage": [2.0, 3.0]})
    source.attrs = {"units": {"time": "seconds", "voltage": "volts"}}
    file = nwb_file()
    file.add_acquisition(TimeSeries(name="voltage", data=source.voltage.to_numpy(),
                                   timestamps=source.time.to_numpy(), unit="volts"))
    path = tmp_path / "series.nwb.zarr"
    with NWBZarrIO(str(path), "w") as io:
        io.write(file)
    raw = open_raw_zarr(path, mode="r")["acquisition/voltage"]
    with NWBZarrIO(str(path), "r") as io:
        assert validate(io=io) == []
        series = io.read().acquisition["voltage"]
        assert raw["data"].attrs["unit"] == series.unit == "volts"
        assert raw["timestamps"].attrs["unit"] == series.timestamps_unit == "seconds"
        for name, data, expected in [("data", series.data, source.voltage.to_numpy()),
                                     ("timestamps", series.timestamps, source.time.to_numpy())]:
            assert raw[name].dtype == data.dtype
            assert raw[name].shape == data.shape
            np.testing.assert_array_equal(raw[name][:], data[:], strict=True)
            np.testing.assert_array_equal(raw[name][:], expected, strict=True)


def test_zarr_table_reference_positions_and_target_match_nwb_reader(tmp_path):
    """Given related DataFrames, explicitly map row positions to DynamicTableRegion.

    Expected: raw positions equal NWB positions, and HDMF's stored target link
    resolves to the same target ids and quality values exposed by raw Zarr.
    Why: HDMF-Zarr encodes object references in attributes; generic zarr exposes
    those attributes without turning them into PyNWB objects. Parquet integer
    columns alone do not contain an NWB object reference or enforce integrity.
    Specs: https://hdmf-zarr.readthedocs.io/en/stable/hdmf_zarr.html
    https://hdmf-common-schema.readthedocs.io/en/stable/format.html
    """
    units_source = pd.DataFrame({"quality": [0.9, 0.8]}, index=pd.Index([10, 20], name="id"))
    events_source = pd.DataFrame({"unit_row": [1, 0], "amplitude": [0.1, 0.2]})
    units = DynamicTable.from_dataframe(units_source, name="units_source")
    events = DynamicTable.from_dataframe(events_source[["amplitude"]], name="events")
    events.add_column(name="unit_row", description="Target row positions", table=units,
                      data=events_source.unit_row.to_numpy())
    file = nwb_file()
    module = file.create_processing_module("comparison", "Related tables")
    module.add(units)
    module.add(events)
    path = tmp_path / "references.nwb.zarr"
    with NWBZarrIO(str(path), "w") as io:
        io.write(file)
    raw = open_raw_zarr(path, mode="r")["processing/comparison"]
    with NWBZarrIO(str(path), "r") as io:
        assert validate(io=io) == []
        reference = io.read().processing["comparison"]["events"]["unit_row"]
        np.testing.assert_array_equal(raw["events/unit_row"][:], reference.data[:], strict=True)
        np.testing.assert_array_equal(reference.data[:], events_source.unit_row.to_numpy())
        np.testing.assert_array_equal(raw["units_source/id"][:], reference.table.id.data[:], strict=True)
        np.testing.assert_array_equal(reference.table.id.data[:], units_source.index.to_numpy())
        np.testing.assert_array_equal(raw["units_source/quality"][:], reference.table["quality"].data[:])
        # Assert the raw link's identity as well as the already resolved object.
        target = zarr_reference(raw["events/unit_row"].attrs["table"])
        assert target["source"] == "."
        assert target["path"] == "/processing/comparison/units_source"
        assert "object_id" not in target
        assert raw["units_source"].attrs["object_id"] == reference.table.object_id


def test_hdf5_and_zarr_backends_recover_the_same_table(tmp_path):
    """Given mixed numeric/text columns, compare NWB HDF5 and Zarr recovery.

    Expected: both stores recover identical DataFrames and source values.
    HDMF/pandas infer StringDtype from object text; numeric dtypes stay unchanged.
    Why: NWB's logical DynamicTable schema is shared; HDF5 datasets and
    Zarr chunks are distinct storage mappings of that schema. Agreement here
    does not imply identical files, codecs, or pandas extension dtype support.
    Specs: https://hdmf-zarr.readthedocs.io/en/stable/tutorials/plot_nwb_zarrio.html
    """
    source = pd.DataFrame({"count": np.array([1, 2], dtype="int16"),
                          "score": np.array([0.25, 0.5], dtype="float32"),
                          "label": pd.Series(["control", "drug"], dtype=object)},
                          index=pd.Index([0, 1], name="id"))
    hdf5 = nwb_roundtrip(source, tmp_path / "source.nwb", backend="hdf5")
    zarr_frame = nwb_roundtrip(source, tmp_path / "source.nwb.zarr", backend="zarr")
    pd.testing.assert_frame_equal(hdf5, zarr_frame)
    pd.testing.assert_frame_equal(source[["count", "score"]], zarr_frame[["count", "score"]])
    assert source.label.tolist() == zarr_frame.label.tolist()
    assert source.label.dtype == np.dtype(object)
    assert isinstance(zarr_frame.label.dtype, pd.StringDtype)


@pytest.mark.parametrize("consolidated", [False, True], ids=["ordinary-metadata", "consolidated-metadata"])
def test_zarr_metadata_modes_match_nwb_reader(tmp_path, consolidated):
    """Given a DataFrame, write with and without consolidated Zarr metadata.

    Expected: ordinary zarr reads match NWBZarrIO values and table metadata;
    a consolidated snapshot, when written, agrees with ordinary metadata too.
    Why: v3 embedded consolidated metadata caches child metadata.
    Consolidation changes metadata access, not array values or the NWB schema.
    Reader parity must therefore include attributes as well as numeric payloads.
    References: https://hdmf-zarr.readthedocs.io/en/stable/hdmf_zarr.html
    https://zarr.readthedocs.io/en/stable/user-guide/consolidated_metadata/
    """
    source = pd.DataFrame({"value": np.array([0.25, 0.5], dtype="float32")})
    file = nwb_file()
    file.create_processing_module("comparison", "Comparison").add(
        DynamicTable.from_dataframe(source.rename_axis("id"), name="source",
                                    table_description="Source DataFrame"))
    path = tmp_path / "metadata.nwb.zarr"
    with NWBZarrIO(str(path), "w") as io:
        io.write(file, consolidate_metadata=consolidated)
    raw = open_raw_zarr(path, mode="r")["processing/comparison/source"]
    with NWBZarrIO(str(path), "r") as io:
        assert validate(io=io) == []
        table = io.read().processing["comparison"]["source"]
        np.testing.assert_array_equal(raw["value"][:], table["value"].data[:], strict=True)
        assert raw.attrs["description"] == table.description
        assert raw.attrs["colnames"] == list(table.colnames)
    if consolidated:
        snapshot = zarr.open_consolidated(str(path), mode="r")["processing/comparison/source"]
        assert snapshot.attrs.asdict() == raw.attrs.asdict()
        np.testing.assert_array_equal(snapshot["value"][:], raw["value"][:], strict=True)
    pd.testing.assert_frame_equal(source.rename_axis("id"), read_nwb(path, backend="zarr"),
                                  check_index_type=False)


def test_stale_consolidated_zarr_metadata_can_disagree_with_nwb_reader(tmp_path):
    """Given a store, edit table description without refreshing its metadata snapshot.

    Expected: ordinary zarr sees the edit; default NWBZarrIO and consolidated
    zarr see the old description. NWBZarrIO r- also sees the snapshot because
    it delegates to zarr.open with automatic consolidated reading.
    Reconsolidation restores agreement.
    The array payload never changes.
    Why: NWBZarrIO prefers consolidated metadata in mode r. Our raw reader
    explicitly bypasses it (use_consolidated=False in Zarr 3). Stale snapshots
    (embedded consolidated metadata in v3) produce a real
    metadata mismatch, not numeric deserialization loss or a different NWB schema.
    This explains why reader parity is conditional on metadata consistency.
    References: https://hdmf-zarr.readthedocs.io/en/stable/tutorials/plot_nwb_zarrio.html#consolidating-metadata
    https://zarr.readthedocs.io/en/stable/user-guide/consolidated_metadata/
    """
    source = pd.DataFrame({"value": [1, 2]})
    path = write_nwb(source, tmp_path / "stale.nwb.zarr", backend="zarr")
    raw = open_raw_zarr(path, mode="r+")["processing/comparison/source"]
    original = raw.attrs["description"]
    raw.attrs["description"] = "Updated description"
    with NWBZarrIO(str(path), "r") as io:
        table = io.read().processing["comparison"]["source"]
        assert table.description == original
        assert table.description != raw.attrs["description"]
        np.testing.assert_array_equal(table["value"].data[:], raw["value"][:], strict=True)
    snapshot = zarr.open_consolidated(str(path), mode="r")["processing/comparison/source"]
    assert snapshot.attrs["description"] == original
    with NWBZarrIO(str(path), "r-") as io:
        expected = original
        assert io.read().processing["comparison"]["source"].description == expected
    zarr.consolidate_metadata(str(path))
    with NWBZarrIO(str(path), "r") as io:
        assert validate(io=io) == []
        assert io.read().processing["comparison"]["source"].description == raw.attrs["description"]
    refreshed = zarr.open_consolidated(str(path), mode="r")["processing/comparison/source"]
    assert refreshed.attrs["description"] == raw.attrs["description"]
