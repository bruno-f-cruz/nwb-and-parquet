"""Direct recovery: values, pandas schema, and unsupported representations."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from .helpers import assert_values_equal, nwb_roundtrip, parquet_roundtrip


def test_numpy_is_newer_than_two():
    """Require NumPy > 2.0 so these observations use the requested array stack.

    Why (specification versus adapter):
        This is an environment precondition, not a storage-format rule. NumPy dtypes
        are the starting point for Arrow physical/logical types and NWB typed arrays.

    References:
        https://arrow.apache.org/docs/python/pandas.html
    """
    assert np.lib.NumpyVersion(np.__version__) > "2.0.0"


def test_int8_values_and_dtype_survive_both_formats(tmp_path, nwb_roundtrip):
    """Round-trip a NumPy int8 column. Both formats preserve its values and
    dtype; NWB normalizes the row index to id.

    Why (specification versus adapter):
        Parquet has integer physical types with width/signedness annotations,
        BOOLEAN, FLOAT, and DOUBLE. HDMF stores typed VectorData; Zarr records
        dtype in array metadata. Arrow's schema restores narrow pandas dtypes
        that cannot be inferred from Parquet's physical type alone.

    References:
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
        https://arrow.apache.org/docs/python/pandas.html
    """
    dtype = "int8"
    source = pd.DataFrame({"value": np.array([0, 1, 1], dtype=dtype)})
    parquet = parquet_roundtrip(source, tmp_path / "source.parquet")
    nwb = nwb_roundtrip(source, tmp_path / "source.nwb")
    pd.testing.assert_frame_equal(source, parquet)
    pd.testing.assert_frame_equal(source.rename_axis("id"), nwb, check_index_type=False)


def test_int64_values_and_dtype_survive_both_formats(tmp_path, nwb_roundtrip):
    """Round-trip a NumPy int64 column. Both formats preserve its values and
    dtype; NWB normalizes the row index to id.

    Why (specification versus adapter):
        Parquet has integer physical types with width/signedness annotations,
        BOOLEAN, FLOAT, and DOUBLE. HDMF stores typed VectorData; Zarr records
        dtype in array metadata. Arrow's schema restores narrow pandas dtypes
        that cannot be inferred from Parquet's physical type alone.

    References:
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
        https://arrow.apache.org/docs/python/pandas.html
    """
    dtype = "int64"
    source = pd.DataFrame({"value": np.array([0, 1, 1], dtype=dtype)})
    parquet = parquet_roundtrip(source, tmp_path / "source.parquet")
    nwb = nwb_roundtrip(source, tmp_path / "source.nwb")
    pd.testing.assert_frame_equal(source, parquet)
    pd.testing.assert_frame_equal(source.rename_axis("id"), nwb, check_index_type=False)


def test_uint64_values_and_dtype_survive_both_formats(tmp_path, nwb_roundtrip):
    """Round-trip a NumPy uint64 column. Both formats preserve its values and
    dtype; NWB normalizes the row index to id.

    Why (specification versus adapter):
        Parquet has integer physical types with width/signedness annotations,
        BOOLEAN, FLOAT, and DOUBLE. HDMF stores typed VectorData; Zarr records
        dtype in array metadata. Arrow's schema restores narrow pandas dtypes
        that cannot be inferred from Parquet's physical type alone.

    References:
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
        https://arrow.apache.org/docs/python/pandas.html
    """
    dtype = "uint64"
    source = pd.DataFrame({"value": np.array([0, 1, 1], dtype=dtype)})
    parquet = parquet_roundtrip(source, tmp_path / "source.parquet")
    nwb = nwb_roundtrip(source, tmp_path / "source.nwb")
    pd.testing.assert_frame_equal(source, parquet)
    pd.testing.assert_frame_equal(source.rename_axis("id"), nwb, check_index_type=False)


def test_float32_values_and_dtype_survive_both_formats(tmp_path, nwb_roundtrip):
    """Round-trip a NumPy float32 column. Both formats preserve its values and
    dtype; NWB normalizes the row index to id.

    Why (specification versus adapter):
        Parquet has integer physical types with width/signedness annotations,
        BOOLEAN, FLOAT, and DOUBLE. HDMF stores typed VectorData; Zarr records
        dtype in array metadata. Arrow's schema restores narrow pandas dtypes
        that cannot be inferred from Parquet's physical type alone.

    References:
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
        https://arrow.apache.org/docs/python/pandas.html
    """
    dtype = "float32"
    source = pd.DataFrame({"value": np.array([0, 1, 1], dtype=dtype)})
    parquet = parquet_roundtrip(source, tmp_path / "source.parquet")
    nwb = nwb_roundtrip(source, tmp_path / "source.nwb")
    pd.testing.assert_frame_equal(source, parquet)
    pd.testing.assert_frame_equal(source.rename_axis("id"), nwb, check_index_type=False)


def test_float64_values_and_dtype_survive_both_formats(tmp_path, nwb_roundtrip):
    """Round-trip a NumPy float64 column. Both formats preserve its values and
    dtype; NWB normalizes the row index to id.

    Why (specification versus adapter):
        Parquet has integer physical types with width/signedness annotations,
        BOOLEAN, FLOAT, and DOUBLE. HDMF stores typed VectorData; Zarr records
        dtype in array metadata. Arrow's schema restores narrow pandas dtypes
        that cannot be inferred from Parquet's physical type alone.

    References:
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
        https://arrow.apache.org/docs/python/pandas.html
    """
    dtype = "float64"
    source = pd.DataFrame({"value": np.array([0, 1, 1], dtype=dtype)})
    parquet = parquet_roundtrip(source, tmp_path / "source.parquet")
    nwb = nwb_roundtrip(source, tmp_path / "source.nwb")
    pd.testing.assert_frame_equal(source, parquet)
    pd.testing.assert_frame_equal(source.rename_axis("id"), nwb, check_index_type=False)


def test_bool_values_and_dtype_survive_both_formats(tmp_path, nwb_roundtrip):
    """Round-trip a NumPy bool column. Both formats preserve its values and
    dtype; NWB normalizes the row index to id.

    Why (specification versus adapter):
        Parquet has integer physical types with width/signedness annotations,
        BOOLEAN, FLOAT, and DOUBLE. HDMF stores typed VectorData; Zarr records
        dtype in array metadata. Arrow's schema restores narrow pandas dtypes
        that cannot be inferred from Parquet's physical type alone.

    References:
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
        https://arrow.apache.org/docs/python/pandas.html
    """
    dtype = "bool"
    source = pd.DataFrame({"value": np.array([0, 1, 1], dtype=dtype)})
    parquet = parquet_roundtrip(source, tmp_path / "source.parquet")
    nwb = nwb_roundtrip(source, tmp_path / "source.nwb")
    pd.testing.assert_frame_equal(source, parquet)
    pd.testing.assert_frame_equal(source.rename_axis("id"), nwb, check_index_type=False)


def test_large_unsigned_integers_do_not_pass_through_float(tmp_path, nwb_roundtrip):
    """Round-trip uint64 values above 2**53 through both formats. Both retain
    exact values and uint64 dtype, including 2**64 - 1.

    Why (specification versus adapter):
        Parquet annotates INT64 with unsigned 64-bit semantics; NWB stores uint64
        arrays in either backend. Float64 cannot represent all these values, but
        neither selected mapping needs an intermediate floating conversion.

    References:
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
    """
    source = pd.DataFrame({"value": np.array([2**53 + 1, 2**64 - 1], dtype="uint64")})
    for recovered in [parquet_roundtrip(source, tmp_path / "a.parquet"),
                      nwb_roundtrip(source, tmp_path / "a.nwb")]:
        np.testing.assert_array_equal(source.value.to_numpy(), recovered.value.to_numpy())
        assert recovered.value.dtype == np.dtype("uint64")


def test_nan_infinities_and_signed_zero_survive(tmp_path, nwb_roundtrip):
    """Round-trip float NaN, infinities, and negative zero. Both readers preserve
    these values, including the sign bit of negative zero.

    Why (specification versus adapter):
        IEEE floating payloads encode these values. Arrow may map pandas NaN to
        Parquet null, so matching recovered NaN does not prove identical physical
        null representation. Signed zero requires a separate signbit check.

    References:
        https://parquet.apache.org/docs/file-format/nulls/
    """
    source = pd.DataFrame({"value": [np.nan, np.inf, -np.inf, -0.0]})
    for recovered in [parquet_roundtrip(source, tmp_path / "a.parquet"),
                      nwb_roundtrip(source, tmp_path / "a.nwb")]:
        np.testing.assert_array_equal(source.value, recovered.value)
        assert np.signbit(recovered.value.iloc[-1])


def test_nwb_loses_index_name_but_keeps_integer_ids(tmp_path, nwb_roundtrip):
    """Compare a named integer index. Parquet keeps its name and values; the NWB
    adapter renames the index to id for schema validity while keeping its values.

    Why (specification versus adapter):
        DynamicTable requires integer ElementIdentifiers named id. Our adapter
        renames the axis without a restoration sidecar. A pandas index name is
        not a Parquet core field; Arrow's pandas metadata restores it.

    References:
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
        https://arrow.apache.org/docs/python/pandas.html
    """
    source = pd.DataFrame({"value": [10, 20]}, index=pd.Index([7, 11], name="trial"))
    pd.testing.assert_frame_equal(source, parquet_roundtrip(source, tmp_path / "a.parquet"))
    recovered = nwb_roundtrip(source, tmp_path / "a.nwb")
    assert recovered.index.name == "id"
    pd.testing.assert_frame_equal(source.rename_axis("id"), recovered)


def test_categorical_labels_survive_nwb_but_category_schema_does_not(tmp_path, nwb_roundtrip):
    """Compare an ordered categorical with an unused category. Parquet restores
    the full categorical schema; NWB keeps labels but loses categories and order.

    Why (specification versus adapter):
        Our NWB adapter emits ordinary VectorData labels, not EnumData or a dtype
        sidecar. Parquet dictionary encoding alone does not imply ordered pandas
        categories; Arrow schema/pandas metadata preserve those semantics.

    References:
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
        https://arrow.apache.org/docs/python/pandas.html
    """
    source = pd.DataFrame({"condition": pd.Categorical(
        ["control", "drug", "control"], categories=["drug", "control", "unused"], ordered=True
    )})
    pd.testing.assert_frame_equal(source, parquet_roundtrip(source, tmp_path / "a.parquet"))
    recovered = nwb_roundtrip(source, tmp_path / "a.nwb")
    assert_values_equal(source, recovered)
    assert not isinstance(recovered.condition.dtype, pd.CategoricalDtype)


def test_nullable_integer_without_nulls_loses_extension_dtype_in_nwb(tmp_path, nwb_roundtrip):
    """Compare pandas Int64 without nulls. Parquet restores the extension dtype;
    NWB restores NumPy int64 with the same values.

    Why (specification versus adapter):
        pandas Int64 is an extension dtype, distinct from NumPy int64. VectorData
        stores integers without a pandas dtype descriptor; Arrow's metadata enables
        Int64 restoration. Losing the wrapper does not lose these values.

    References:
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
        https://arrow.apache.org/docs/python/pandas.html
    """
    source = pd.DataFrame({"value": pd.array([1, 2], dtype="Int64")})
    pd.testing.assert_frame_equal(source, parquet_roundtrip(source, tmp_path / "a.parquet"))
    recovered = nwb_roundtrip(source, tmp_path / "a.nwb")
    assert_values_equal(source, recovered)
    assert recovered.value.dtype == np.dtype("int64")


def test_nullable_integer_with_null_is_rejected_by_default_nwb_mapping(tmp_path, nwb_roundtrip):
    """Compare nullable Int64 with a missing value. Parquet preserves the column;
    the default NWB mapping rejects the mixed integer/NA representation.

    Why (specification versus adapter):
        Parquet definition levels represent nulls separately from integer values.
        Plain VectorData has no validity-mask column in our mapping and cannot
        encode pandas.NA in its integer array. A custom NWB encoding could do so.

    References:
        https://parquet.apache.org/docs/file-format/nulls/
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
    """
    source = pd.DataFrame({"value": pd.array([1, None], dtype="Int64")})
    pd.testing.assert_frame_equal(source, parquet_roundtrip(source, tmp_path / "a.parquet"))
    with pytest.raises((TypeError, ValueError), match="(?i)(type|dtype|object)"):
        nwb_roundtrip(source, tmp_path / "a.nwb")


def test_string_index_is_supported_by_parquet_but_not_nwb_integer_ids(tmp_path, nwb_roundtrip):
    """Compare string row labels. Parquet restores them; the NWB integer-id
    mapping rejects them instead of silently replacing the labels.

    Why (specification versus adapter):
        ElementIdentifiers requires integers. String labels need a separate NWB
        column plus integer ids. Arrow stores string index values as a column
        and records the index role in pandas metadata; Parquet has no native index.

    References:
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
        https://arrow.apache.org/docs/python/pandas.html
    """
    source = pd.DataFrame({"value": [1, 2]}, index=pd.Index(["a", "b"], name="label"))
    pd.testing.assert_frame_equal(source, parquet_roundtrip(source, tmp_path / "a.parquet"))
    with pytest.raises((TypeError, ValueError), match="(?i)(int|dtype|convert)"):
        nwb_roundtrip(source, tmp_path / "a.nwb")


def test_datetime_column_needs_an_explicit_nwb_encoding(tmp_path):
    """Compare a datetime column. Parquet restores it; NWB's HDF5 write fails
    because this default mapping has no numpy.datetime64 encoding.

    Why (specification versus adapter):
        This specifically targets h5py's missing numpy.datetime64 mapping. Parquet
        TIMESTAMP defines epoch-based units; VectorData does not automatically
        supply a pandas datetime encoding. Native Zarr datetime support is tested
        separately and does not by itself establish NWB schema validity.

    References:
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
    """
    source = pd.DataFrame({"time": pd.to_datetime(["2026-01-01", "2026-01-02"])})
    pd.testing.assert_frame_equal(source, parquet_roundtrip(source, tmp_path / "a.parquet"))
    # HDMF wraps the underlying h5py TypeError in a plain Exception.
    with pytest.raises(Exception, match="Could not create dataset time") as error:
        nwb_roundtrip(source, tmp_path / "a.nwb")
    assert isinstance(error.value.__cause__, TypeError)


def test_complex_column_rejected_by_both_default_mappings(tmp_path, nwb_roundtrip):
    """Attempt a complex128 column. HDMF and PyArrow both reject the default
    representation; HDMF wraps its underlying ValueError.

    Why (specification versus adapter):
        Parquet defines no standard complex logical type, so Arrow rejects this
        direct column. HDMF rejects complex values before either backend writes;
        native HDF5/Zarr complex support alone cannot fix that build-layer rule.

    References:
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
    """
    source = pd.DataFrame({"value": np.array([1 + 2j, 3 - 4j], dtype="complex128")})
    with pytest.raises(Exception, match="Complex numbers are not supported") as error:
        nwb_roundtrip(source, tmp_path / "a.nwb")
    assert isinstance(error.value.__cause__, ValueError)
    with pytest.raises(pa.ArrowNotImplementedError, match="(?i)(numpy|complex|unsupported)"):
        parquet_roundtrip(source, tmp_path / "a.parquet")


def test_ragged_values_survive_both_with_explicit_nwb_vector_index(tmp_path, nwb_roundtrip):
    """Round-trip lists of unequal lengths, with an explicit NWB ragged hint.
    Both retain values and empty cells but recover list cells as NumPy arrays.

    Why (specification versus adapter):
        HDMF uses flattened VectorData with cumulative-end VectorIndex. Parquet
        LIST uses nested fields and repetition/definition levels. Different layouts
        reconstruct equal row values; list versus ndarray is a reader choice.

    References:
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
    """
    source = pd.DataFrame({"spikes": [[0.1, 0.2], [], [0.5]]})
    for recovered in [parquet_roundtrip(source, tmp_path / "a.parquet"),
                      nwb_roundtrip(source, tmp_path / "a.nwb", ragged=("spikes",))]:
        for before, after in zip(source.spikes, recovered.spikes, strict=True):
            np.testing.assert_array_equal(before, after)
        assert isinstance(source.spikes.iloc[0], list)
        assert isinstance(recovered.spikes.iloc[0], np.ndarray)


def test_unequal_length_lists_are_automatically_detected_as_ragged(tmp_path, nwb_roundtrip):
    """Round-trip unequal-length lists without hints. HDMF automatically creates
    a ragged representation and preserves each row's values.

    Why (specification versus adapter):
        VectorData/VectorIndex specify ragged storage; inferring it from list
        lengths is HDMF from_dataframe behavior. HDF5 and Zarr then store the same
        flattened values and exclusive cumulative end positions.

    References:
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
    """
    source = pd.DataFrame({"spikes": [[0.1, 0.2], [], [0.5]]})
    recovered = nwb_roundtrip(source, tmp_path / "a.nwb")
    for before, after in zip(source.spikes, recovered.spikes, strict=True):
        np.testing.assert_array_equal(before, after)


def test_parquet_struct_inference_adds_missing_keys(tmp_path):
    """Round-trip dictionaries with different keys through Parquet. Struct
    inference inserts absent keys with null values, changing the dictionaries.

    Why (specification versus adapter):
        Arrow infers a fixed struct schema using the union of keys. Parquet stores
        optional child fields; absent keys and explicit nulls are indistinguishable
        in this mapping. Dictionary changes occur during inference/recovery.

    References:
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
        https://parquet.apache.org/docs/file-format/nulls/
    """
    source = pd.DataFrame({"settings": [{"gain": 2}, {"offset": 3}]})
    recovered = parquet_roundtrip(source, tmp_path / "a.parquet")
    assert recovered.settings.iloc[0] == {"gain": 2.0, "offset": None}
    assert recovered.settings.iloc[1] == {"gain": None, "offset": 3.0}
    assert source.settings.iloc[0] != recovered.settings.iloc[0]


def test_none_and_nan_collapse_to_one_parquet_null_representation(tmp_path):
    """Round-trip an object column distinguishing None from NaN. Arrow maps both
    to null; recovery uses float64 NaN and loses the original distinction.

    Why (specification versus adapter):
        Arrow maps both Python None and NaN to null in this pandas conversion.
        Parquet records definition levels, not the originating Python sentinel.
        A tag column could preserve the distinction; NaN loss is not intrinsic.

    References:
        https://parquet.apache.org/docs/file-format/nulls/
        https://arrow.apache.org/docs/python/pandas.html
    """
    source = pd.DataFrame({"value": pd.Series([None, np.nan, 1.0], dtype=object)})
    assert source.value.iloc[0] is None
    recovered = parquet_roundtrip(source, tmp_path / "a.parquet")
    assert recovered.value.isna().tolist() == [True, True, False]
    assert recovered.value.iloc[0] is not None
    assert recovered.value.dtype == np.dtype("float64")


def test_timestamp_truncation_is_rejected_unless_explicitly_authorized(tmp_path):
    """Compare exact and microsecond-coerced Parquet timestamps. Exact writes
    preserve nanoseconds; coercion rejects loss unless opted in, then loses 123 ns.

    Why (specification versus adapter):
        TIMESTAMP supports millisecond, microsecond, and nanosecond units. Explicit
        PyArrow coercion changes units and rejects discarded nonzero digits by
        default. Opting into truncation changes the source value.

    References:
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
    """
    source = pd.DataFrame({"time": pd.to_datetime(["2026-01-01T00:00:00.000000123"])})
    pd.testing.assert_frame_equal(source, parquet_roundtrip(source, tmp_path / "exact.parquet"))
    with pytest.raises(pa.ArrowInvalid, match="(?i)(lose|truncat)"):
        parquet_roundtrip(source, tmp_path / "reject.parquet", coerce_timestamps="us")
    recovered = parquet_roundtrip(source, tmp_path / "loss.parquet",
                                  coerce_timestamps="us", allow_truncated_timestamps=True)
    assert source.time.iloc[0] - recovered.time.iloc[0] == pd.Timedelta(123, unit="ns")


def test_parquet_without_pandas_metadata_loses_nullable_dtype_and_index_role(tmp_path):
    """Remove pandas metadata before writing Parquet. The index returns as an
    ordinary column and nullable Int64 becomes float64, though ids survive.

    Why (specification versus adapter):
        Parquet columns do not define a pandas index role or extension dtype.
        Removing metadata keeps physical values and nulls but removes Arrow's
        restoration instructions; generic to_pandas then chooses float64.

    References:
        https://arrow.apache.org/docs/python/pandas.html
        https://parquet.apache.org/docs/file-format/nulls/
    """
    source = pd.DataFrame({"value": pd.array([1, None], dtype="Int64")},
                          index=pd.Index([7, 9], name="trial"))
    path = tmp_path / "generic.parquet"
    pq.write_table(pa.Table.from_pandas(source).replace_schema_metadata(None), path)
    recovered = pq.read_table(path).to_pandas()
    assert recovered.columns.tolist() == ["value", "trial"]
    assert recovered.index.equals(pd.RangeIndex(2))
    assert recovered.value.dtype == np.dtype("float64")
    assert recovered.trial.tolist() == source.index.tolist()


def test_float_workaround_for_nullable_integers_loses_large_integer_precision(tmp_path, nwb_roundtrip):
    """Cast nullable integers to float to permit NWB writing. The adapter already
    rounds 2**53 + 1 to 2**53; NWB retains that rounded float, not the source integer.

    Why (specification versus adapter):
        IEEE float64 has 53 bits of integer precision. This preprocessing rounds
        before serialization; NWB retains the rounded payload. Integer VectorData
        plus a validity-mask column would avoid this adapter's precision loss.

    References:
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
        https://parquet.apache.org/docs/file-format/nulls/
    """
    source = pd.DataFrame({"value": pd.array([2**53 + 1, None], dtype="Int64")})
    pd.testing.assert_frame_equal(source, parquet_roundtrip(source, tmp_path / "exact.parquet"))
    encoded = source.astype("float64")
    recovered = nwb_roundtrip(encoded, tmp_path / "float.nwb")
    assert int(recovered.value.iloc[0]) == 2**53
    assert int(recovered.value.iloc[0]) != int(source.value.iloc[0])
    assert np.isnan(recovered.value.iloc[1])


    # This loss happens in the chosen adapter before NWB serialization.


def test_timezone_and_nanosecond_timestamp_survive_parquet(tmp_path):
    """Round-trip timezone-aware nanosecond timestamps through Parquet. Values,
    precision, and America/Los_Angeles timezone are retained.

    Why (specification versus adapter):
        TIMESTAMP records precision and UTC-adjusted semantics, not an IANA zone
        identifier. PyArrow's stored Arrow schema restores America/Los_Angeles.
        This recovery is not guaranteed for every Parquet reader.

    References:
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
        https://arrow.apache.org/docs/python/pandas.html
    """
    source = pd.DataFrame({"time": pd.to_datetime(
        ["2026-01-01T00:00:00.000000123Z", "2026-06-01T00:00:00.000000456Z"]
    ).tz_convert("America/Los_Angeles")})
    pd.testing.assert_frame_equal(source, parquet_roundtrip(source, tmp_path / "a.parquet"))


def test_multiindex_survives_parquet_but_needs_remapping_for_nwb(tmp_path, nwb_roundtrip):
    """Compare a two-level index. Parquet restores it; the NWB adapter fails
    during its single-axis id normalization, before serialization.

    Why (specification versus adapter):
        Arrow serializes index levels as columns and records pandas metadata. NWB
        DynamicTable has one integer id axis. Our rename_axis normalization fails
        before writing; mapping index levels into columns needs an explicit adapter.

    References:
        https://hdmf-common-schema.readthedocs.io/en/stable/format.html
        https://arrow.apache.org/docs/python/pandas.html
    """
    source = pd.DataFrame({"value": [1, 2]}, index=pd.MultiIndex.from_tuples(
        [("animal_a", 0), ("animal_b", 1)], names=["animal", "trial"]
    ))
    pd.testing.assert_frame_equal(source, parquet_roundtrip(source, tmp_path / "a.parquet"))
    # The adapter's single id-axis normalization cannot represent two axes.
    with pytest.raises(TypeError, match="Must pass list-like as `names`"):
        nwb_roundtrip(source, tmp_path / "a.nwb")


def test_unicode_text_values_survive_both_formats(tmp_path, nwb_roundtrip):
    """Round-trip accented, CJK, and empty text. Both formats preserve the text
    values; this check does not require identical pandas string dtypes.

    Why (specification versus adapter):
        Parquet STRING annotates UTF-8 BYTE_ARRAY. NWB text uses the backend string
        mapping, including variable-length UTF-8 codecs in HDMF-Zarr. Equal text
        values need not imply the same Python or pandas string dtype.

    References:
        https://parquet.apache.org/docs/file-format/types/logicaltypes/
        https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html
    """
    source = pd.DataFrame({"label": pd.Series(["neurón", "神経", ""], dtype=object)})
    for recovered in [parquet_roundtrip(source, tmp_path / "a.parquet"),
                      nwb_roundtrip(source, tmp_path / "a.nwb")]:
        assert recovered.label.tolist() == source.label.tolist()


