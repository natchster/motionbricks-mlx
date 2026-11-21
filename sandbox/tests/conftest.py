import os

import pytest

# `cudashim` is importable as the installed workspace member (uv run --all-packages).

# MOTIONBRICKS_REQUIRE_FIXTURES=1 turns every skip in sandbox/tests (missing upstream checkout,
# checkpoints, golden fixtures, ONNX export, MPS, or the optional CuMetal toolchain) into a
# failure, so a full run cannot pass by skipping. Mirrors model/tests/conftest.py.
REQUIRE_FIXTURES = os.environ.get("MOTIONBRICKS_REQUIRE_FIXTURES", "").strip().lower() not in ("", "0", "false", "no")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if REQUIRE_FIXTURES and report.skipped and not hasattr(report, "wasxfail"):
        reason = report.longrepr[-1] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        report.outcome = "failed"
        report.longrepr = f"skipped, but MOTIONBRICKS_REQUIRE_FIXTURES is set: {reason}"
