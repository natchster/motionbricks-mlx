import os

import pytest

from mlx_helpers import MODELS

# MOTIONBRICKS_REQUIRE_FIXTURES=1 turns every skip in model/tests (missing checkpoints, golden or
# runtime fixtures, clips, the Metal GPU) into a failure, so a full run cannot pass by skipping.
REQUIRE_FIXTURES = os.environ.get("MOTIONBRICKS_REQUIRE_FIXTURES", "").strip().lower() not in ("", "0", "false", "no")


@pytest.fixture(params=MODELS)
def model_name(request):
    return request.param


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if REQUIRE_FIXTURES and report.skipped and not hasattr(report, "wasxfail"):
        reason = report.longrepr[-1] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        report.outcome = "failed"
        report.longrepr = f"skipped, but MOTIONBRICKS_REQUIRE_FIXTURES is set: {reason}"
