"""Exercise shared DataFrame mappings with both NWB storage backends."""

from functools import partial
import inspect
from importlib.metadata import version
import gc
from pathlib import Path
import shutil
import time
import warnings

import pytest

from .helpers import nwb_roundtrip as roundtrip

WORKSPACE = Path(__file__).resolve().parent.parent
TEST_ARTIFACTS = WORKSPACE / ".test-run"


def remove_test_artifacts():
    """Remove only the fixed workspace folder, retrying transient Windows locks."""
    if TEST_ARTIFACTS.resolve() != WORKSPACE / ".test-run":
        raise pytest.UsageError("The test artifact folder must stay inside the workspace")
    gc.collect()
    for attempt in range(10):
        try:
            shutil.rmtree(TEST_ARTIFACTS)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == 9:
                raise
            time.sleep(0.25)


def pytest_configure(config):
    # Let pytest's normal tmp_path fixture isolate cases inside a single root.
    if TEST_ARTIFACTS.resolve() != WORKSPACE / ".test-run":
        raise pytest.UsageError("The test artifact folder must stay inside the workspace")
    config.option.basetemp = str(TEST_ARTIFACTS)


def pytest_sessionstart(session):
    remove_test_artifacts()


def pytest_sessionfinish(session, exitstatus):
    try:
        remove_test_artifacts()
    except OSError as error:
        warnings.warn(f"Could not delete {TEST_ARTIFACTS}: {error}. "
                      "The next run will retry cleanup in the same folder.", RuntimeWarning)


def pytest_report_header():
    packages = ("numpy", "pandas", "pyarrow", "pynwb", "hdmf", "hdmf-zarr", "zarr")
    return "; ".join(f"{package} {version(package)}" for package in packages)


@pytest.fixture(params=["hdf5", "zarr"])
def nwb_roundtrip(request):
    """Inject a backend-specific adapter; test ids identify the storage backend."""
    return partial(roundtrip, backend=request.param)


def pytest_addoption(parser):
    parser.addoption("--describe", action="store_true",
                     help="Display each test's scenario and specification rationale")


def pytest_runtest_setup(item):
    if item.config.getoption("--describe"):
        reporter = item.config.pluginmanager.getplugin("terminalreporter")
        if reporter is not None:
            reporter.write_sep("-", item.nodeid)
            reporter.write_line(inspect.getdoc(item.obj) or "Missing test explanation")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.failed:
        report.sections.append(("Scenario and specification", inspect.getdoc(item.obj) or ""))
