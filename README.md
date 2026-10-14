# DataFrame fidelity: NWB and Parquet

Pytest examples comparing Parquet, NWB-HDF5, and NWB-Zarr, with DataFrames as the
source of truth and **NumPy > 2.0**. Expected loss and rejection are asserted, so
a passing suite means the documented behavior was observed, not that every
conversion is lossless. Each function's docstring explains its scenario and
the observed preservation, loss, or rejection, followed by the relevant
specification rules, adapter choices, and links to primary documentation.

Every test docstring has a **Fidelity** annotation for Parquet, NWB-HDF5, and
NWB-Zarr. The labels apply to that test's inputs, adapter, options, and assertions:

- **LOSSLESS**: the named values or metadata survive; this is not a claim about
  every property of a DataFrame.
- **LOSSY VALUES / METADATA / STRUCTURE / SEMANTICS**: identifies information
  that cannot be recovered, with the loss attributed to its specific step.
- **CONVENTION**: storage layout or reader representation differs while the
  named information survives, such as lists becoming arrays.
- **UNSUPPORTED / INVALID SCHEMA**: conversion is rejected or the written
  object is invalid; a failed conversion is not labelled as a lossy round trip.
- **NOT TESTED / NOT APPLICABLE**: no fidelity claim for that format in the test.

A test can have multiple labels: retained values can coexist with lost dtype
metadata, and convention changes can accompany loss of the original index name.
The stale-metadata example uses **CACHE MISMATCH** because the information still
exists on disk but a reader selects an outdated snapshot. `--describe` displays
these annotations together with the existing explanations and source links.

## Run

```powershell
uv sync --locked
uv run --locked pytest -v
uv run --locked pytest -v --describe
```

`pyproject.toml` declares dependencies; `uv.lock` records exact direct and
transitive versions and distribution hashes. `.python-version` selects Python
3.13.7, and uv manages `.venv` automatically. Keep `uv.lock` in version control.
The `--locked` flag rejects an outdated lockfile instead of changing versions.
See [uv's locking and syncing documentation](https://docs.astral.sh/uv/concepts/projects/sync/).

Use `uv add <package>` and `uv remove <package>` for dependency changes.
Run `uv lock --upgrade` to resolve the latest stable releases, then
`uv sync --locked` and rerun the examples. `.uv-cache/` stays in the workspace.

## Latest dependency stack

There is one dependency list, with no legacy profile or upper version caps.
Only NumPy retains `>2`, the requested source-data requirement. The lockfile
records exact direct/transitive versions and hashes after `uv lock --upgrade`.
Locked runs reproduce those versions; upgrades refresh them deliberately.

Verified package versions: NumPy 2.5.4, pandas 3.0.6, PyArrow 26.0.0,
h5py 3.16.0, PyNWB 4.2.0, HDMF 6.2.0, hdmf-zarr 0.14.0, Zarr 3.4.1,
and pytest 9.1.1. uv 0.13.0 was used and is the minimum required tool version.
Update uv with `winget upgrade --id astral-sh.uv --exact` for a WinGet install,
or `uv self update` for a standalone install.
Verification: **87 passed**, with no skips or expected-failure marks. Upstream
consolidation and generic-datetime warnings remain visible.

Python 3.13.7 is pinned for reproducibility; the project supports Python >= 3.12.
Package dependencies are resolved to the latest stable versions independently
of this interpreter pin.

Pytest runs all cases by default and reports failures without stopping at the
first. A pass means the documented behavior occurred, including expected loss
or rejection. Use `--describe` to print each test's docstring, or read the
docstring in the test file. Failed reports also include that explanation.
`uv run --locked python -m tests` remains an alias for pytest. Temporary stores
use pytest's `tmp_path` fixture inside the fixed workspace folder `.test-run/`.
The suite clears that folder before running and deletes it afterward, including
failed runs. Cleanup retries transient Windows locks; if deletion still fails,
it reports a warning and leaves that same folder for cleanup on the next run.
The suite manages `--basetemp` automatically, so no per-run paths are needed.

## Structure

- `tests/test_roundtrips.py`: direct DataFrame → file → DataFrame recovery.
- `tests/test_cross_deserialization.py`: DataFrame → NWB → DataFrame → Parquet
  → DataFrame, and the reverse order. Also an explicit datetime encoding.
- `tests/test_format_semantics.py`: wrong readers, schema validation, session
  context, physical units, matrix/ragged representations, and table references.
- `tests/test_zarr_readers.py`: raw `zarr` versus `NWBZarrIO` arrays, metadata,
  ragged reconstruction, units, references, and stale consolidated metadata.
- `tests/test_zarr_datetime.py`: native Zarr datetime storage versus the
  default HDMF-Zarr DataFrame mapping.
- `tests/helpers.py`: small, visible adapters shared by the examples.
- `tests/conftest.py`: HDF5/Zarr adapter fixture and docstring reporting.
- `tests/__main__.py`: optional pytest entry point.

Shared round-trip and conversion functions use a parametrized `nwb_roundtrip`
fixture: `[hdf5]` and `[zarr]` in test ids identify the backend. HDF5-specific
reader and datetime failures are separate from Zarr-specific checks. The
numeric raw-Zarr comparison is also parametrized by dtype. Expected rejection
uses `pytest.raises`; no skips or expected-failure marks hide unsupported cases.

This structure separates **value fidelity**, **pandas schema fidelity**,
**unsupported mappings**, and **scientific meaning**. Those are different claims:
equal cell values do not imply equal dtypes, equal schemas, or equal semantics.
Each source is a DataFrame; session metadata required to construct an NWB file
is explicitly supplied by the adapter. Relationship and unit examples also
start with DataFrames, then explicitly map their meaning into NWB objects.

## Observed cases

The NWB column below applies to shared HDF5/Zarr mappings unless a backend is
explicitly named. The Zarr datetime test documents its distinct error.

| Source or operation | Parquet / PyArrow | NWB / HDMF mapping |
| --- | --- | --- |
| Numeric widths, bool, large uint64 | Values and dtype retained | Values and dtype retained |
| NaN, infinity, signed zero | Retained | Retained |
| Unicode text | Values retained | Values retained |
| Named integer index | Name and ids retained | Adapter keeps ids, renames index to `id` |
| String index / MultiIndex | Retained | String ids rejected; MultiIndex needs adapter remapping |
| Ordered categorical with unused category | Schema retained via metadata | Labels retained; categorical schema lost |
| Nullable `Int64`, no missing values | Extension dtype retained | Converted to NumPy int64 |
| Nullable `Int64` with missing values | Retained | Default mapping rejected |
| Nullable large integer cast to float | Cast loses precision before writing | Same adapter loss; NWB retains rounded float |
| Naive datetime | Retained | HDF5 rejects dtype; NWB-Zarr writes and validates but loses units and cannot recover a DataFrame |
| Timezone + nanosecond timestamp | Retained | No default datetime mapping demonstrated |
| Explicit datetime → int64 nanoseconds | Integers retained | Integers retained; decoding contract needed |
| Complex values | Arrow mapping rejected | HDMF mapping rejected |
| Unequal-length lists | List values retained; recovered as arrays | Automatically inferred ragged; values retained as arrays |
| Equal-length lists | List representation | Matrix by default; explicit hint gives VectorIndex |
| Dictionaries with different keys | Struct inference inserts null keys | Not exercised as a dictionary mapping |
| Object column containing `None` and `NaN` | Both become null, float recovery | Not exercised as a mixed object mapping |
| Nanoseconds coerced to microseconds | Rejected by default; opt-in truncation loses 123 ns | Not applicable to default datetime mapping |
| Pandas metadata removed | Index becomes a column; nullable dtype lost | No pandas metadata sidecar in adapter |
| JSON-compatible units in `DataFrame.attrs` | Retained as pandas metadata | Generic table loses attrs; explicit TimeSeries mapping preserves units |
| Explicit table relationship | Plain export retains positions, loses target link | DynamicTableRegion retains target and positions |
| Opposite format reader | Cannot read NWB | Cannot read Parquet |
| Raw HDMF `from_dataframe` with index named `trial` | Valid named-index representation | Writes HDF5 but fails NWB schema validation |

## Zarr reader agreement and its limits

`NWBZarrIO` reconstructs PyNWB objects from Zarr v3 stores using the latest
hdmf-zarr backend. Only this current stack is tested; older stores and migration
are outside this suite's scope.

The parity tests open the **same store** with `zarr.open_group` and `NWBZarrIO`.
Raw reads explicitly pass `use_consolidated=False` in Zarr 3 so they see live
metadata rather than automatically selecting a cached snapshot.
They compare numeric array values, dtypes, shapes, ids, text, ragged data and
endpoints, unit attributes, and target-table reference metadata. They also
compare with the source DataFrame and validate the NWB store. Ragged rows are
manually sliced from raw flattened arrays to verify NWB's reconstruction.
References use `_REFERENCE` source/path wrappers. Target paths and the target dataset/group object ids are checked
against the reconstructed objects. Raw strings in the latest stack have
`StringDType`, while HDMF exposes an object-dtype string view; values match.
Latest HDMF/pandas also recover plain object-text columns as pandas StringDtype.

Raw `zarr` provides arrays and attributes; it does not automatically construct
DynamicTable, TimeSeries, or target-table objects. `NWBZarrIO` uses the same
Python Zarr implementation underneath, so this verifies agreement between
reader layers rather than two independent codec implementations.

There is a real metadata mismatch case: editing per-group attrs without
refreshing `.zmetadata` makes ordinary `zarr.open_group` see new metadata while
default `NWBZarrIO` sees the stale consolidated snapshot. The test demonstrates
that `NWBZarrIO(mode="r-")` still reads the snapshot because its internal `zarr.open` call uses
automatic consolidation; the test explicitly documents this discrepancy.
Reconsolidation restores agreement. Numeric payloads match.
Consolidation in v3 is currently a Python Zarr extension and emits an upstream
warning. See the
[HDMF-Zarr storage mapping](https://hdmf-zarr.readthedocs.io/en/stable/hdmf_zarr.html)
and [current backend API](https://hdmf-zarr.readthedocs.io/en/stable/hdmf_zarr.html).

Every successful NWB example is schema-validated, except the example that
deliberately demonstrates an invalid file. The default NWB adapter renames the
index to `id` to avoid that schema failure. Other dtype/attrs restoration is not
performed. Unequal-length lists are automatically detected by HDMF; explicit
ragged hints are useful for equal-length lists where intent cannot be inferred.

## Interpreting the results

These compare **specific adapters and dependency versions**, not everything
either format can represent. NWB here uses HDF5 and Zarr v3 backends and a generic
DynamicTable, with separate explicit mappings for TimeSeries and references.
NWB extensions, EnumData, validity-mask columns, or restoration metadata could
change the results. Likewise, custom Arrow schemas or conversion options can
change Parquet recovery. Parquet can carry application metadata; it does not
automatically interpret it as an NWB session or object relationship.

Cross-deserialization has two meanings here: file conversion through a
DataFrame bridge, and using the other format's reader. The former can work
with losses; the latter fails because these containers are different.

For more cases, keep the same four questions per example: did writing succeed,
did the schema validate, did values survive, and did metadata/meaning survive?
Add explicit adapters as paired examples rather than silently repairing the
source, so the costs of each encoding remain visible.

The relevant upstream behavior is documented in [Arrow's pandas integration](https://arrow.apache.org/docs/python/pandas.html),
[Parquet writing options](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html),
and [HDMF's DynamicTable guide](https://hdmf.readthedocs.io/en/stable/tutorials/plot_dynamictable_howto.html).
