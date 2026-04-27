# `include/`

Shared interface and schema definitions consumed by code under `../src/ssfl_project/`.

Python doesn't have header files in the C/C++ sense, so this folder is the
project's home for the closest equivalents:

- **Type stubs** (`.pyi`) for modules whose runtime implementation lives
  elsewhere but whose static interface needs to be pinned down.
- **Schemas**: JSON / YAML / TOML descriptions of message payloads,
  on-disk artefacts (e.g. partition manifests), and CLI configs.
- **Protocol / `typing.Protocol` definitions** that describe duck-typed
  contracts between strategy, client, and server modules.
- **Constant tables** that more than one module imports (e.g. the canonical
  N-BaIoT class label list) — only when keeping them inside `config.py`
  starts to bloat that module.

Anything that must be importable at runtime should still live under
`src/ssfl_project/`. Files here are reference material; if you add a runtime
import path, also expose it through `config.py` so the rest of the code
keeps a single import surface.
