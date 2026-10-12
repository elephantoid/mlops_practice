"""The serving log config: app INFO reaches stderr, library INFO does not.

Without a config, uvicorn handles only its own loggers. Every ``src.*`` record then falls to
``logging.lastResort``, which passes WARNING and above with no level prefix -- so the
"model loaded" and "visible CPUs" lines never reached Cloud Run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from uvicorn.config import LOGGING_CONFIG

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_CONFIG = "src/api/logging.json"

# Applied in a subprocess: in the pytest process, dictConfig would leave the root logger at
# WARNING with an extra stderr handler for every later test. caplog reattaches its own handler
# per test, but a test reading a non-src INFO record without caplog.at_level would see nothing.
_EMIT = """
import json, logging, logging.config, sys
with open(sys.argv[1]) as f:
    logging.config.dictConfig(json.load(f))
logging.getLogger("src.api.main").info("app-info-line")
logging.getLogger("mlflow").info("library-info-line")
logging.getLogger("mlflow").warning("library-warning-line")
"""


def test_the_serving_log_config_emits_app_info_and_holds_library_info_back():
    result = subprocess.run(
        [sys.executable, "-c", _EMIT, str(PROJECT_ROOT / LOG_CONFIG)],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stderr.splitlines()

    app = [line for line in lines if "app-info-line" in line]
    assert len(app) == 1, f"app INFO must print exactly once, got {app}"
    assert app[0].startswith("INFO:"), app[0]
    assert not any("library-info-line" in line for line in lines), lines
    warning = [line for line in lines if "library-warning-line" in line]
    assert len(warning) == 1 and warning[0].startswith("WARNING:"), warning


def test_the_image_starts_uvicorn_with_the_serving_log_config():
    """A dropped flag would only show up as missing lines on the live service."""
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()
    # The last CMD: the runtime stage is the final one; the dev stage's CMD comes earlier.
    cmd = [line for line in dockerfile.splitlines() if line.startswith("CMD ")][-1]

    assert f"--log-config {LOG_CONFIG}" in cmd, cmd
    assert (PROJECT_ROOT / LOG_CONFIG).is_file()


def test_the_serving_log_config_is_uvicorns_default_plus_src_and_root():
    """The file copies uvicorn's default; an upgrade that changes the default must not leave it
    stale without notice. Reconcile by copying the new default and re-adding the two entries."""
    config = json.loads((PROJECT_ROOT / LOG_CONFIG).read_text())

    assert config.pop("root") == {"handlers": ["default"], "level": "WARNING"}
    assert config["loggers"].pop("src") == {"level": "INFO"}
    assert config == LOGGING_CONFIG
