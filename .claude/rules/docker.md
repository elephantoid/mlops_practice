---
paths:
  - "Dockerfile"
  - "docker-compose.yml"
  - ".dockerignore"
---

# Image packaging

- **No `libomp`.** LightGBM needs it on macOS only; Linux base images ship `libgomp`.
  Adding it cargo-cults a host-specific fix into the image.
- **No dev dependencies.** `pytest`, `ruff`, and `pre-commit` have no place in a serving
  image.
- **`uv` only** — never `pip install` (project-wide constraint).
- **`MODEL_VERSION` must be baked in alongside `MODEL_URI=/app/model`.** With a local-path
  model URI the API cannot query the registry for a version, so `/health` reports
  `"unknown"` — in exactly the deployment where knowing the live version matters most.
  `src/models/export.py` emits the resolved version for this reason.
- The registry lives in `mlflow.db` + `mlruns/`, both gitignored and both excluded from the
  build context. The export step therefore runs **on the host**, before the build.
