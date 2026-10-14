"""Conversion chains are distinct from handing a file to the wrong reader."""

import numpy as np
import pandas as pd
import pytest

from .helpers import assert_values_equal, nwb_roundtrip, parquet_roundtrip


def test_numeric_values_survive_conversion_nwb_then_parquet(tmp_path, nwb_roundtrip):
    """Convert DataFrame through NWB, then PARQUET.
    Numeric float32 values and dtype survive both conversion steps.

    Fidelity (this test and adapter):
        Parquet: LOSSLESS values/dtype at this step and across the full chain.
        NWB-HDF5: LOSSLESS values/dtype at this step and across the full chain.
        NWB-Zarr: LOSSLESS values/dtype at this step and across the full chain.

    Why (specification versus adapter):
        float32 maps to Parquet FLOAT and NWB typed VectorData. Both retain this
        payload; ids already named id avoid renaming. The second format cannot
        infer additional semantics absent from the intermediate DataFrame.

    References:
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
    """
    order = ("nwb", "parquet")
    source = pd.DataFrame({"value": np.array([0.25, 0.5], dtype="float32")},
                          index=pd.Index([5, 8], name="id"))
    recovered = source
    for format_name in order:
        adapter = nwb_roundtrip if format_name == "nwb" else parquet_roundtrip
        recovered = adapter(recovered, tmp_path / f"step.{format_name}")
    pd.testing.assert_frame_equal(source, recovered)


def test_numeric_values_survive_conversion_parquet_then_nwb(tmp_path, nwb_roundtrip):
    """Convert DataFrame through PARQUET, then NWB.
    Numeric float32 values and dtype survive both conversion steps.

    Fidelity (this test and adapter):
        Parquet: LOSSLESS values/dtype at this step and across the full chain.
        NWB-HDF5: LOSSLESS values/dtype at this step and across the full chain.
        NWB-Zarr: LOSSLESS values/dtype at this step and across the full chain.

    Why (specification versus adapter):
        float32 maps to Parquet FLOAT and NWB typed VectorData. Both retain this
        payload; ids already named id avoid renaming. The second format cannot
        infer additional semantics absent from the intermediate DataFrame.

    References:
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
    """
    order = ("parquet", "nwb")
    source = pd.DataFrame({"value": np.array([0.25, 0.5], dtype="float32")},
                          index=pd.Index([5, 8], name="id"))
    recovered = source
    for format_name in order:
        adapter = nwb_roundtrip if format_name == "nwb" else parquet_roundtrip
        recovered = adapter(recovered, tmp_path / f"step.{format_name}")
    pd.testing.assert_frame_equal(source, recovered)


def test_parquet_cannot_restore_category_schema_lost_nwb_then_parquet(tmp_path, nwb_roundtrip):
    """Convert DataFrame through NWB, then PARQUET.
    Labels survive, but categorical ordering and unused categories are lost
    at the NWB step and cannot be recovered by a subsequent Parquet write.

    Fidelity (this test and adapter):
        Parquet: LOSSLESS for its input frame; cannot restore category metadata discarded at the NWB step.
        NWB-HDF5: LOSSY METADATA: category dtype, ordering, and unused categories are lost; labels survive.
        NWB-Zarr: LOSSY METADATA: category dtype, ordering, and unused categories are lost; labels survive.

    Why (specification versus adapter):
        Plain NWB VectorData stores observed labels without ordered-category
        metadata or unused levels. Arrow metadata written afterward describes the
        already simplified dtype; it cannot reconstruct categories lost earlier.

    References:
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
        https://arrow.apache.org/docs/python/pandas.html
    """
    order = ("nwb", "parquet")
    source = pd.DataFrame({"condition": pd.Categorical(
        ["drug", "control"], categories=["drug", "control", "unused"], ordered=True
    )})
    recovered = source
    for format_name in order:
        adapter = nwb_roundtrip if format_name == "nwb" else parquet_roundtrip
        recovered = adapter(recovered, tmp_path / f"step.{format_name}")
    assert_values_equal(source, recovered)
    assert not isinstance(recovered.condition.dtype, pd.CategoricalDtype)


def test_parquet_cannot_restore_category_schema_lost_parquet_then_nwb(tmp_path, nwb_roundtrip):
    """Convert DataFrame through PARQUET, then NWB.
    Labels survive, but categorical ordering and unused categories are lost
    at the final NWB step, despite being preserved by the first Parquet write.

    Fidelity (this test and adapter):
        Parquet: LOSSLESS for its input frame; cannot restore category metadata discarded at the NWB step.
        NWB-HDF5: LOSSY METADATA: category dtype, ordering, and unused categories are lost; labels survive.
        NWB-Zarr: LOSSY METADATA: category dtype, ordering, and unused categories are lost; labels survive.

    Why (specification versus adapter):
        Plain NWB VectorData stores observed labels without ordered-category
        metadata or unused levels. Arrow preserves these at the first step, but
        the final NWB adapter does not map them into its table representation.

    References:
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
        https://arrow.apache.org/docs/python/pandas.html
    """
    order = ("parquet", "nwb")
    source = pd.DataFrame({"condition": pd.Categorical(
        ["drug", "control"], categories=["drug", "control", "unused"], ordered=True
    )})
    recovered = source
    for format_name in order:
        adapter = nwb_roundtrip if format_name == "nwb" else parquet_roundtrip
        recovered = adapter(recovered, tmp_path / f"step.{format_name}")
    assert_values_equal(source, recovered)
    assert not isinstance(recovered.condition.dtype, pd.CategoricalDtype)


def test_ragged_values_survive_conversion_nwb_then_parquet(tmp_path, nwb_roundtrip):
    """Convert DataFrame through NWB, then PARQUET.
    Ragged row values, including empty rows, survive both conversion steps.
    Python lists are recovered as NumPy arrays.

    Fidelity (this test and adapter):
        Parquet: LOSSLESS row values; CONVENTION: list cells are recovered as NumPy arrays.
        NWB-HDF5: LOSSLESS row values; CONVENTION: flattened VectorData plus VectorIndex, recovered as arrays.
        NWB-Zarr: LOSSLESS row values; CONVENTION: flattened VectorData plus VectorIndex, recovered as arrays.

    Why (specification versus adapter):
        NWB VectorIndex endpoints and Parquet LIST nesting are distinct layouts.
        The DataFrame bridge reconstructs rows between writes, allowing values
        and empty rows to survive both conversion directions.

    References:
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
    """
    order = ("nwb", "parquet")
    source = pd.DataFrame({"spikes": [[0.1, 0.2], [], [0.5]]})
    recovered = source
    for format_name in order:
        adapter = nwb_roundtrip if format_name == "nwb" else parquet_roundtrip
        recovered = adapter(recovered, tmp_path / f"step.{format_name}")
    for before, after in zip(source.spikes, recovered.spikes, strict=True):
        np.testing.assert_array_equal(before, after)


def test_ragged_values_survive_conversion_parquet_then_nwb(tmp_path, nwb_roundtrip):
    """Convert DataFrame through PARQUET, then NWB.
    Ragged row values, including empty rows, survive both conversion steps.
    Python lists are recovered as NumPy arrays.

    Fidelity (this test and adapter):
        Parquet: LOSSLESS row values; CONVENTION: list cells are recovered as NumPy arrays.
        NWB-HDF5: LOSSLESS row values; CONVENTION: flattened VectorData plus VectorIndex, recovered as arrays.
        NWB-Zarr: LOSSLESS row values; CONVENTION: flattened VectorData plus VectorIndex, recovered as arrays.

    Why (specification versus adapter):
        NWB VectorIndex endpoints and Parquet LIST nesting are distinct layouts.
        The DataFrame bridge reconstructs rows between writes, allowing values
        and empty rows to survive both conversion directions.

    References:
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
    """
    order = ("parquet", "nwb")
    source = pd.DataFrame({"spikes": [[0.1, 0.2], [], [0.5]]})
    recovered = source
    for format_name in order:
        adapter = nwb_roundtrip if format_name == "nwb" else parquet_roundtrip
        recovered = adapter(recovered, tmp_path / f"step.{format_name}")
    for before, after in zip(source.spikes, recovered.spikes, strict=True):
        np.testing.assert_array_equal(before, after)


def test_parquet_timestamp_still_needs_encoding_before_nwb(tmp_path):
    """Write a timestamp through Parquet before NWB. Parquet preserves it, but
    the subsequent NWB write still fails without an explicit datetime encoding.

    Fidelity (this test and adapter):
        Parquet: LOSSLESS at the first timestamp round trip.
        NWB-HDF5: UNSUPPORTED at the next write; the conversion chain does not complete.
        NWB-Zarr: NOT TESTED in this test.

    Why (specification versus adapter):
        Restoring Parquet TIMESTAMP produces datetime64 again; it does not teach
        h5py to encode it. This chain targets HDF5; Zarr's native datetime payload
        and its NWB validation are checked separately.

    References:
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
    """
    source = pd.DataFrame({"time": pd.to_datetime(["2026-01-01T00:00:00.000000123"])})
    intermediate = parquet_roundtrip(source, tmp_path / "first.parquet")
    pd.testing.assert_frame_equal(source, intermediate)
    with pytest.raises(Exception, match="Could not create dataset time"):
        nwb_roundtrip(intermediate, tmp_path / "second.nwb")


def test_explicit_timestamp_integer_encoding_survives_but_requires_a_contract(tmp_path, nwb_roundtrip):
    """Encode datetime as integer nanoseconds, then pass through NWB and Parquet.
    Explicit decoding restores the source; generic readers only see integers.

    Fidelity (this test and adapter):
        Parquet: LOSSLESS encoded integers; CONVENTION: explicit epoch/unit decoding restores timestamps.
        NWB-HDF5: LOSSLESS encoded integers; CONVENTION: explicit epoch/unit decoding restores timestamps.
        NWB-Zarr: LOSSLESS encoded integers; CONVENTION: explicit epoch/unit decoding restores timestamps.

    Why (specification versus adapter):
        Plain NWB integer columns do not imply an epoch or timestamp unit. Parquet
        also sees integers after this NWB step. Explicit decoding supplies a
        nanosecond contract that neither generic reader can infer.

    References:
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
    """
    source = pd.DataFrame({"time": pd.to_datetime(["2026-01-01T00:00:00.000000123"])})
    encoded = pd.DataFrame({"time_ns": source.time.to_numpy(dtype="datetime64[ns]").view("int64")})
    intermediate = nwb_roundtrip(encoded, tmp_path / "first.nwb")
    recovered = parquet_roundtrip(intermediate, tmp_path / "second.parquet")
    decoded = pd.DataFrame({"time": pd.to_datetime(recovered.time_ns.to_numpy(), unit="ns")})
    pd.testing.assert_frame_equal(source, decoded)
    assert recovered.time_ns.dtype == np.dtype("int64")


    # Neither generic reader knows that the integer means UTC nanoseconds.
