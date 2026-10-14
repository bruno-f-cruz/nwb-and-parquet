"""Separate native Zarr dtype capability from the HDMF-Zarr adapter."""

import numpy as np
import pandas as pd
import pytest
from hdmf_zarr import NWBZarrIO
from pynwb import validate

from .helpers import open_raw_zarr, parquet_roundtrip, write_nwb


def test_native_zarr_datetime_keeps_units_but_nwb_mapping_loses_them(tmp_path):
    """Given a nanosecond datetime DataFrame, compare native Zarr and NWB-Zarr.

    Fidelity (this test and adapter):
        Parquet: LOSSLESS timestamps; native Python Zarr is also lossless with explicit units.
        NWB-HDF5: NOT TESTED in this test.
        NWB-Zarr: LOSSY METADATA: timestamp units lost despite schema validation; DataFrame recovery fails.

    Expected: native Zarr and Parquet retain timestamp units. NWB-Zarr validates
    generic datetime64: ticks survive but units are lost, so DataFrame recovery
    fails. Raw Zarr and NWB reader arrays agree despite that semantic loss.
    Why: native Python Zarr supports unit-qualified datetime as an extension dtype. HDMF passes a list of scalars whose
    dtype inference in this backend loses that unit, producing generic <M8.
    This is an adapter limitation, not proof that Zarr cannot store datetime.
    NWB semantic encoding (for example explicit integer epochs or ISO text)
    must also be specified, independently of native array dtype support.
    Parquet TIMESTAMP supplies epoch/unit semantics; PyArrow maps pandas to it.
    References:
    https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html
    https://hdmf-zarr.readthedocs.io/en/stable/hdmf_zarr.html
    https://parquet.apache.org/docs/file-format/types/logicaltypes/
    """
    source = pd.DataFrame({"time": pd.to_datetime(["2026-01-01T00:00:00.000000123",
                                                 "2026-01-02T00:00:00.000000456"])})
    values = source.time.to_numpy(dtype="datetime64[ns]")
    root = open_raw_zarr(tmp_path / "native.zarr", mode="w")
    root.create_array("time", data=values)
    raw = open_raw_zarr(tmp_path / "native.zarr")["time"]
    np.testing.assert_array_equal(raw[:], values, strict=True)
    pd.testing.assert_frame_equal(source, parquet_roundtrip(source, tmp_path / "source.parquet"))
    path = tmp_path / "source.nwb.zarr"
    write_nwb(source, path, backend="zarr")
    raw = open_raw_zarr(path)["processing/comparison/source/time"]
    assert np.datetime_data(raw.dtype) == ("generic", 1)
    np.testing.assert_array_equal(raw[:].view("int64"), values.view("int64"), strict=True)
    with NWBZarrIO(str(path), "r") as io:
        assert validate(io=io) == []
        table = io.read().processing["comparison"]["source"]
        assert table["time"].data.dtype == raw.dtype
        np.testing.assert_array_equal(table["time"].data[:].view("int64"),
                                      raw[:].view("int64"), strict=True)
        with pytest.raises(TypeError, match="datetime64 values must have a unit specified"):
            table.to_dataframe()
