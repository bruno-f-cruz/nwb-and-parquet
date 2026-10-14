"""Explicit adapters, deliberately without dtype/attrs restoration sidecars."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from hdmf.common import DynamicTable
from hdmf_zarr import NWBZarrIO
from pynwb import NWBFile, NWBHDF5IO, validate

def open_raw_zarr(path, *, mode="r"):
    """Read live metadata explicitly; Zarr 3 otherwise prefers consolidation."""
    return zarr.open_group(str(path), mode=mode, use_consolidated=False)


def zarr_reference(attribute):
    """Assert and decode the latest HDMF-Zarr v3 reference wrapper."""
    assert set(attribute) == {"_REFERENCE"}
    return attribute["_REFERENCE"]


def nwb_file():
    return NWBFile(
        session_description="DataFrame fidelity experiment",
        identifier="dataframe-comparison",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def nwb_io(backend):
    """Select storage while retaining the same PyNWB/HDMF object model."""
    return {"hdf5": NWBHDF5IO, "zarr": NWBZarrIO}[backend]


def write_nwb(frame, path, *, ragged=(), backend="hdf5"):
    """Map the DataFrame index to NWB integer ids, with explicit ragged hints."""
    columns = [
        {"name": name, "description": f"Source column {name}", "index": True}
        for name in ragged
    ]
    table = DynamicTable.from_dataframe(
        frame.rename_axis("id"), name="source", table_description="Source DataFrame",
        columns=columns or None,
    )
    file = nwb_file()
    file.create_processing_module("comparison", "Tabular comparison").add(table)
    with nwb_io(backend)(str(path), "w") as io:
        io.write(file)
    return path


def read_nwb(path, *, backend="hdf5"):
    with nwb_io(backend)(str(path), "r") as io:
        # Materialize while the backend is open. Ragged cells are copied too.
        frame = io.read().processing["comparison"]["source"].to_dataframe()
        for name in frame:
            if frame[name].dtype == object:
                frame[name] = frame[name].map(
                    lambda value: value.copy() if isinstance(value, np.ndarray) else value
                )
        return frame


def nwb_roundtrip(frame, path, *, ragged=(), backend="hdf5"):
    write_nwb(frame, path, ragged=ragged, backend=backend)
    # Passing an IO avoids the path validator's HDF5 backend inference.
    with nwb_io(backend)(str(path), "r") as io:
        errors = validate(io=io)
    assert errors == [], f"Written NWB must satisfy its schema: {errors}"
    return read_nwb(path, backend=backend)


def parquet_roundtrip(frame, path, **write_options):
    pq.write_table(pa.Table.from_pandas(frame), path, **write_options)
    return pq.read_table(path).to_pandas()


def assert_values_equal(left, right):
    """Explicitly weaker than DataFrame equality: ignore dtypes/index metadata."""
    pd.testing.assert_frame_equal(
        left.reset_index(drop=True), right.reset_index(drop=True),
        check_dtype=False, check_categorical=False,
    )
