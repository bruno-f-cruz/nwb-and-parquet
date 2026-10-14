"""Container, schema, and scientific meaning differ even when values match."""

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hdmf.common import DynamicTable, DynamicTableRegion, VectorData
from pynwb import NWBHDF5IO, TimeSeries, validate

from .helpers import nwb_file, parquet_roundtrip, write_nwb


def test_readers_cannot_cross_read_the_other_file_format(tmp_path):
    """Hand each file to the opposite reader. Parquet's magic bytes and NWB's
    HDF5 container differ; both readers reject the incompatible file.

    Fidelity (this test and adapter):
        Parquet: CONVENTION/INCOMPATIBILITY: Parquet reader cannot read the HDF5 container; not a lossy conversion.
        NWB-HDF5: CONVENTION/INCOMPATIBILITY: HDF5 reader cannot read Parquet; not a lossy conversion.
        NWB-Zarr: NOT TESTED in this test.

    Why (specification versus adapter):
        This specifically compares HDF5-backed NWB and Parquet framing/footer.
        NWB-Zarr needs another reader too; renaming files cannot convert containers.

    References:
        https://parquet.apache.org/docs/file-format/
        https://hdmf-zarr.readthedocs.io/en/stable/tutorials/plot_nwb_zarrio.html
    """
    source = pd.DataFrame({"value": [1, 2]})
    nwb_path = write_nwb(source, tmp_path / "source.nwb")
    parquet_path = tmp_path / "source.parquet"
    parquet_roundtrip(source, parquet_path)
    assert h5py.is_hdf5(nwb_path)
    assert not h5py.is_hdf5(parquet_path)
    assert parquet_path.read_bytes()[:4] == b"PAR1"
    with pytest.raises(pa.ArrowInvalid, match="(?i)(magic|parquet)"):
        pq.read_table(nwb_path)
    with pytest.raises(OSError, match="(?i)(signature|open)"):
        with NWBHDF5IO(str(parquet_path), "r") as io:
            io.read()


def test_hdmf_named_index_can_write_hdf5_that_fails_nwb_schema(tmp_path, nwb_roundtrip):
    """Use raw HDMF conversion with an index named trial. HDF5 writing succeeds
    but NWB validation fails because id is required; the normal adapter fixes it.

    Fidelity (this test and adapter):
        Parquet: NOT TESTED in this test.
        NWB-HDF5: INVALID SCHEMA for raw trial-named ids; normal adapter preserves ids but loses their original name.
        NWB-Zarr: LOSSY METADATA in the normal adapter: index name normalized to id; invalid raw Zarr write is not tested.

    Why (specification versus adapter):
        DynamicTable requires an id dataset. Raw from_dataframe uses the supplied
        index name, so a storage write can succeed without meeting NWB's schema.
        Successful serialization and schema validation are distinct checks.

    References:
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
    """
    source = pd.DataFrame({"value": [1, 2]}, index=pd.Index([7, 9], name="trial"))
    table = DynamicTable.from_dataframe(source, name="source")
    assert table.id.name == "trial"
    file = nwb_file()
    file.create_processing_module("comparison", "Comparison").add(table)
    path = tmp_path / "nonstandard_id.nwb"
    with NWBHDF5IO(str(path), "w") as io:
        io.write(file)
    errors = validate(path=str(path))
    assert errors
    assert any("id" in str(error) for error in errors)
    # Our normal adapter renames the index to id before writing.
    recovered = nwb_roundtrip(source, tmp_path / "valid.nwb")
    assert recovered.index.tolist() == source.index.tolist()


def test_nwb_has_session_context_parquet_does_not_infer_it(tmp_path):
    """Compare container context. The NWB adapter explicitly supplies session
    identity, start time, and table description; Parquet does not infer them.

    Fidelity (this test and adapter):
        Parquet: CONVENTION: no inferred NWB session context; none existed in the source, so this is not data loss.
        NWB-HDF5: CONVENTION: adapter supplies NWB session context explicitly; context is recovered.
        NWB-Zarr: NOT TESTED in this test.

    Why (specification versus adapter):
        NWBFile defines session description, identifier, and start time, which we
        explicitly supply rather than derive from cells. Parquet allows custom
        metadata but defines no mandatory NWB session context.

    References:
        https://nwb-schema.readthedocs.io/en/stable/format.html
        https://parquet.apache.org/docs/file-format/metadata/
    """
    source = pd.DataFrame({"value": [1.0, 2.0]})
    nwb_path = write_nwb(source, tmp_path / "source.nwb")
    parquet_path = tmp_path / "source.parquet"
    parquet_roundtrip(source, parquet_path)
    with NWBHDF5IO(str(nwb_path), "r") as io:
        file = io.read()
        assert file.identifier == "dataframe-comparison"
        assert file.session_start_time is not None
        assert file.processing["comparison"]["source"].description == "Source DataFrame"
    metadata = pq.read_schema(parquet_path).metadata or {}
    assert b"session_start_time" not in metadata
    assert b"identifier" not in metadata


def test_units_in_dataframe_attrs_need_explicit_nwb_mapping(tmp_path, nwb_roundtrip):
    """Compare units stored in DataFrame.attrs. Parquet preserves these attrs,
    the generic NWB table loses them, and an explicit TimeSeries mapping keeps units.

    Fidelity (this test and adapter):
        Parquet: LOSSLESS unit attrs with this PyArrow metadata mapping.
        NWB-HDF5: LOSSY METADATA in generic table; LOSSLESS units/values with explicit TimeSeries mapping.
        NWB-Zarr: LOSSY METADATA in generic table; explicit Zarr TimeSeries is tested separately.

    Why (specification versus adapter):
        NWB TimeSeries specifies data units and seconds for timestamps. HDMF does
        not map arbitrary DataFrame.attrs into those fields. Arrow preserves JSON
        attrs as application metadata, which is not a TimeSeries scientific schema.

    References:
        https://nwb-schema.readthedocs.io/en/stable/format.html#timeseries
        https://arrow.apache.org/docs/python/pandas.html
    """
    source = pd.DataFrame({"time": [0.0, 0.1], "voltage": [2.0, 3.0]})
    source.attrs = {"units": {"time": "seconds", "voltage": "volts"}}
    # Raw Arrow from_pandas preserves JSON attrs in pandas metadata.
    parquet = parquet_roundtrip(source, tmp_path / "source.parquet")
    assert parquet.attrs == source.attrs
    generic = nwb_roundtrip(source, tmp_path / "table.nwb")
    assert generic.attrs == {}
    file = nwb_file()
    file.add_acquisition(TimeSeries(
        name="voltage", data=source.voltage.to_numpy(),
        timestamps=source.time.to_numpy(), unit=source.attrs["units"]["voltage"],
    ))
    path = tmp_path / "timeseries.nwb"
    with NWBHDF5IO(str(path), "w") as io:
        io.write(file)
    assert validate(path=str(path)) == []
    with NWBHDF5IO(str(path), "r") as io:
        series = io.read().acquisition["voltage"]
        assert series.unit == "volts"
        assert series.timestamps_unit == "seconds"
        np.testing.assert_array_equal(series.data[:], source.voltage)
        np.testing.assert_array_equal(series.timestamps[:], source.time)


    # Parquet attrs are application metadata; they do not create a TimeSeries.


def test_equal_length_lists_have_different_nwb_schemas_with_same_values(tmp_path):
    """Write equal-length lists to NWB with and without a ragged hint. Both are
    valid, but one stores a matrix and the other flattened values plus VectorIndex.

    Fidelity (this test and adapter):
        Parquet: NOT TESTED in this test.
        NWB-HDF5: CONVENTION: matrix versus ragged schema; both preserve the asserted row values.
        NWB-Zarr: NOT TESTED in this test.

    Why (specification versus adapter):
        VectorData allows trailing dimensions; VectorIndex instead supplies ragged
        row endpoints. The hint selects between two valid NWB representations.
        Their differing layouts do not imply a difference in recovered row values.

    References:
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
    """
    source = pd.DataFrame({"samples": [[1.0, 2.0], [3.0, 4.0]]})
    plain_path = write_nwb(source, tmp_path / "matrix.nwb")
    ragged_path = write_nwb(source, tmp_path / "ragged.nwb", ragged=("samples",))
    for path in [plain_path, ragged_path]:
        assert validate(path=str(path)) == []
    with NWBHDF5IO(str(plain_path), "r") as io:
        table = io.read().processing["comparison"]["source"]
        assert "samples_index" not in table
        assert table["samples"].data.shape == (2, 2)
    with NWBHDF5IO(str(ragged_path), "r") as io:
        table = io.read().processing["comparison"]["source"]
        assert "samples_index" in table
        assert table["samples"].target.data.shape == (4,)
        for before, after in zip(source.samples, table.to_dataframe().samples, strict=True):
            np.testing.assert_array_equal(before, after)


def test_plain_integer_foreign_key_is_not_an_nwb_table_reference(tmp_path):
    """Round-trip an integer column named unit_id. Values survive, but neither
    its name nor dtype establishes a target table or an NWB object reference.

    Fidelity (this test and adapter):
        Parquet: LOSSLESS ids; CONVENTION: a column name does not create a relationship.
        NWB-HDF5: CONVENTION: ordinary VectorData does not infer a target table; no supplied reference is lost.
        NWB-Zarr: NOT TESTED in this test.

    Why (specification versus adapter):
        DynamicTableRegion includes a target table; VectorData does not. Neither
        an integer dtype nor the name unit_id creates a reference. Parquet plain
        integer fields likewise define no NWB target-table semantics.

    References:
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
    """
    source = pd.DataFrame({"unit_id": [10, 20], "amplitude": [0.1, 0.2]})
    nwb_path = write_nwb(source, tmp_path / "source.nwb")
    with NWBHDF5IO(str(nwb_path), "r") as io:
        table = io.read().processing["comparison"]["source"]
        assert type(table["unit_id"]) is VectorData
        assert not hasattr(table["unit_id"], "table")
    parquet_path = tmp_path / "source.parquet"
    recovered = parquet_roundtrip(source, parquet_path)
    assert recovered.unit_id.tolist() == source.unit_id.tolist()
    assert pa.types.is_int64(pq.read_schema(parquet_path).field("unit_id").type)


    # A column name alone defines neither referential integrity nor a target table.


def test_explicit_nwb_reference_loses_target_semantics_in_plain_parquet(tmp_path):
    """Build an NWB table reference from two DataFrames, then export positions
    to Parquet. Positions survive; the linked target table and its ids do not.

    Fidelity (this test and adapter):
        Parquet: LOSSY SEMANTICS relative to the linked NWB object: export drops the target link before Parquet writing.
        NWB-HDF5: LOSSLESS for the asserted target reference and row positions before export.
        NWB-Zarr: NOT TESTED in this test.

    Why (specification versus adapter):
        DynamicTableRegion stores row positions plus a link to a target table.
        Exporting just positions discards the link before writing Parquet; retaining
        the ids and relationships needs an explicit export contract.

    References:
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
    """
    units_source = pd.DataFrame({"quality": [0.9, 0.8]}, index=pd.Index([10, 20], name="id"))
    events_source = pd.DataFrame({"unit_row": [1, 0], "amplitude": [0.1, 0.2]})
    units = DynamicTable.from_dataframe(units_source, name="units_source")
    events = DynamicTable.from_dataframe(events_source[["amplitude"]], name="events")
    events.add_column(name="unit_row", description="Row position in units_source",
                      data=events_source.unit_row.to_numpy(), table=units)
    file = nwb_file()
    module = file.create_processing_module("comparison", "Related tables")
    module.add(units)
    module.add(events)
    path = tmp_path / "references.nwb"
    with NWBHDF5IO(str(path), "w") as io:
        io.write(file)
    assert validate(path=str(path)) == []
    with NWBHDF5IO(str(path), "r") as io:
        table = io.read().processing["comparison"]["events"]
        reference = table["unit_row"]
        assert isinstance(reference, DynamicTableRegion)
        assert reference.table.id[:].tolist() == units_source.index.tolist()
        exported = table.to_dataframe(index=True)
        assert exported.unit_row.tolist() == events_source.unit_row.tolist()
    recovered = parquet_roundtrip(exported, tmp_path / "references.parquet")
    assert recovered.unit_row.tolist() == [1, 0]
    assert recovered.attrs == {}


    # Positions survive; the target table, its ids, and the link do not.
