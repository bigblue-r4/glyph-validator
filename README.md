# glyph-validator

Lightweight CJK glyph geometric coherence validator for Python.

A strict, pre-processing gateway that checks character geometry for structural
coherence before data is passed downstream. Deterministic — no LLM, no network,
no linguistic processing. Pure NumPy / OpenCV / Pillow mathematics.

[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/glyph-validator.svg)](https://pypi.org/project/glyph-validator)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)

## Contents

- [What It Does](#what-it-does)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modes](#modes)
- [CLI](#cli)
- [API](#api)
- [Font Configuration](#font-configuration)
- [Dependencies](#dependencies)
- [License](#license)

---

## What It Does

For each CJK character (U+4E00–U+9FFF) found in an input string:

1. **Rasterize** — renders the character to a 128×128 binary matrix via a reference font (Noto Sans CJK or equivalent).
2. **Transform** — applies horizontal mirror, vertical mirror, and 180° rotation.
3. **Interference** — computes `cv2.absdiff(original, transform)` for each pair.
4. **Evaluate** — checks stroke density, tofu-box detection, connected-component count, and interference statistics against strict geometric thresholds.

Execution halts at the first anomalous character. Input is never forwarded downstream.

---

## Installation

```bash
pip install glyph-validator
```

Requires a local CJK font (NotoSansCJK recommended):

```bash
# Debian/Ubuntu
sudo apt install fonts-noto-cjk

# macOS — PingFang.ttc is bundled with the OS
# Windows — msyh.ttc (Microsoft YaHei) is bundled with the OS
```

---

## Quick Start

```python
from glyph_validator import validate_payload

# No CJK — immediate pass
validate_payload("Hello world")
# → {"status": "pass", "code": 200}

# Valid CJK ideographs
validate_payload("中文汉字")
# → {"status": "pass", "code": 200}

# Anomalous character (font fallback / missing glyph / degenerate geometry)
validate_payload("鿿")
# → {"status": "fail", "code": 403, "flagged_hex": "U+9FFF"}
```

---

## Modes

### Threshold mode (default)

Evaluates each character against parametric coherence bounds:

| Check | What it catches |
|-------|----------------|
| Stroke density `[0.03, 0.60]` | Empty renders, solid blocks |
| Border fraction `< 0.40` | Missing-glyph tofu rectangles |
| Component count `≤ 32` | Noisy / fragmented renders |
| Interference floor `0.002` | Implausible tri-axis symmetry (circle fallbacks) |
| Interference ceiling `0.48` | Degenerate stroke geometry |

### Strict mode — per-character SHA-256 fingerprints

Build a reference profile once from your trusted font installation:

```bash
python3 -m glyph_validator --build-profiles profiles.npz
# renders all 20,992 CJK characters → ~8 s → 685 KB compressed
```

Load at runtime for pixel-exact comparison:

```python
from glyph_validator import load_profiles, validate_payload

load_profiles("profiles.npz")   # switches to strict mode automatically

validate_payload("中文")        # → pass (matches stored fingerprints)
validate_payload("鿿")      # → fail (zero-digest entry)
```

Any pixel-level deviation from the reference — caused by font file substitution,
render-parameter tampering, or injected fallback glyphs — triggers an immediate fail.

---

## CLI

```bash
# Single string
glyph-validator "input text"

# Build reference profiles (one-time)
glyph-validator --build-profiles profiles.npz

# Strict mode
glyph-validator --profiles profiles.npz "input text"

# Batch mode — one JSON object per line in, one result per line out
echo '{"id": "r1", "text": "中文"}' | glyph-validator --batch

# Strict + batch
cat requests.jsonl | glyph-validator --profiles profiles.npz --batch
```

Batch input/output format:
```jsonl
{"id": "r1", "text": "中文测试"}
{"id": "r2", "text": "Hello world"}
```
```jsonl
{"status": "pass", "code": 200, "id": "r1"}
{"status": "pass", "code": 200, "id": "r2"}
```

---

## API

```python
from glyph_validator import validate_payload, load_profiles, build_reference_profiles

# Core gateway — returns dict, never raises on bad input
result = validate_payload(input_string: str) -> dict
# {"status": "pass", "code": 200}
# {"status": "fail", "code": 403, "flagged_hex": "U+XXXX"}

# Strict mode setup
build_reference_profiles(output_path: str, verbose: bool = True) -> dict
load_profiles(profile_path: str) -> None
```

---

## Font Configuration

Edit the module-level constants before importing:

```python
import glyph_validator
glyph_validator.FONT_PATH  = "/path/to/your/CJK.ttf"
glyph_validator.FONT_INDEX = 0   # TTC collection index
glyph_validator.FONT_SIZE  = 96  # render point size
```

Default search order: `FONT_PATH` → NotoSansCJK fallbacks → NotoSerifCJK →
PingFang.ttc (macOS) → msyh.ttc (Windows).

---

## Dependencies

- `numpy >= 1.21`
- `opencv-python-headless >= 4.5`
- `Pillow >= 8.0`

No network access. No LLM layers. No external linguistic resources.

---

## License

MIT — see [LICENSE](LICENSE).
