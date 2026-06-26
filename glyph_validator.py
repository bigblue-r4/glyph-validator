"""
glyph_validator.py
------------------
Lightweight CJK glyph geometric coherence validator.

For each CJK character (U+4E00–U+9FFF) in an input string:
  1. Rasterize to a 128×128 binary matrix via a reference font.
  2. Apply horizontal mirror, vertical mirror, and 180° rotation.
  3. Compute absdiff interference matrices for each (original, transform) pair.
  4. Evaluate stroke density, tofu-box detection, connected-component count,
     and interference statistics against strict geometric thresholds.
  5. Optionally compare against a pre-built per-character SHA-256 profile for
     pixel-exact strict mode (catches font substitution and rendering tampering).

Execution halts at the first anomalous character; input is never forwarded.

Dependencies: numpy, opencv-python (or opencv-python-headless), pillow (>=8.0)
No network access. No LLM layers. Entirely deterministic.

Usage:
  # Validate a single string
  python3 glyph_validator.py "input text"

  # Build reference profiles (one-time, ~8s for the full CJK block)
  python3 glyph_validator.py --build-profiles profiles.npz

  # Strict mode: exact pixel fingerprint comparison
  python3 glyph_validator.py --profiles profiles.npz "input text"

  # Batch mode: one JSON object per stdin line, results to stdout
  echo '{"id":"r1","text":"中文"}' | python3 glyph_validator.py --batch

  # Strict + batch
  cat input.jsonl | python3 glyph_validator.py --profiles profiles.npz --batch
"""

from __future__ import annotations

import sys
import json
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


# ═══════════════════════════════════════════════════════════════════════════════
# Font Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Primary font path — set this to a local CJK TrueType/OpenType font.
FONT_PATH: str = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# Fallback paths tried in order when FONT_PATH is absent.
_FONT_FALLBACKS: list[str] = [
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJKsc-Regular.otf",
    "/System/Library/Fonts/PingFang.ttc",       # macOS
    "/System/Library/Fonts/STHeiti Light.ttc",  # macOS fallback
    "C:/Windows/Fonts/msyh.ttc",                # Windows (Microsoft YaHei)
    "C:/Windows/Fonts/simsun.ttc",              # Windows (SimSun)
]

# TrueType collection index. For NotoSansCJK: 0 = SC, 1 = TC, 2 = JP, 3 = KR.
FONT_INDEX: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Processing Constants
# ═══════════════════════════════════════════════════════════════════════════════

GLYPH_SIZE: int = 128          # Square canvas side length in pixels.
FONT_SIZE: int  = 96           # Render point size — leaves ~16 px margin each side.
BINARIZE_THRESH: int = 127     # Grayscale cutoff: > threshold → stroke pixel (1).

# CJK Unified Ideographs block (Basic block, BMP plane).
CJK_START: int = 0x4E00        # 一
CJK_END:   int = 0x9FFF        # 鿿


# ═══════════════════════════════════════════════════════════════════════════════
# Coherence Thresholds (threshold mode)
# ═══════════════════════════════════════════════════════════════════════════════

# Fraction of pixels set in the binary matrix.
STROKE_RATIO_MIN: float = 0.03
STROKE_RATIO_MAX: float = 0.60

# Fraction of stroke pixels on the outermost canvas border.
BORDER_FRACTION_MAX: float = 0.40

# Number of 8-connected foreground regions.
COMPONENT_MAX: int = 32

# Interference mean bounds (all-below = implausible symmetry; any-above = degenerate).
INTERFERENCE_MEAN_FLOOR: float = 0.002
INTERFERENCE_MEAN_MAX:   float = 0.48


# ═══════════════════════════════════════════════════════════════════════════════
# Internal caches
# ═══════════════════════════════════════════════════════════════════════════════

_font_cache: Optional[ImageFont.FreeTypeFont] = None

# Strict-mode profile: maps code_point (int) → 32-byte SHA-256 digest (bytes).
# Populated by load_profiles() or build_reference_profiles().
_profile: dict[int, bytes] = {}


# ═══════════════════════════════════════════════════════════════════════════════
# Font Loader
# ═══════════════════════════════════════════════════════════════════════════════

def _load_font() -> ImageFont.FreeTypeFont:
    global _font_cache
    if _font_cache is not None:
        return _font_cache
    for path_str in [FONT_PATH] + _FONT_FALLBACKS:
        p = Path(path_str)
        if p.exists():
            try:
                _font_cache = ImageFont.truetype(str(p), FONT_SIZE, index=FONT_INDEX)
                return _font_cache
            except Exception:
                continue
    raise FileNotFoundError(
        "glyph_validator: no CJK font found.\n"
        f"Set FONT_PATH to a local NotoSansCJK (or equivalent) font.\n"
        f"Tried: {[FONT_PATH] + _FONT_FALLBACKS}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CJK Scanner
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_cjk(text: str) -> list[str]:
    """Return unique CJK characters from text in first-appearance order."""
    seen: set[str] = set()
    out:  list[str] = []
    for ch in text:
        if CJK_START <= ord(ch) <= CJK_END and ch not in seen:
            seen.add(ch)
            out.append(ch)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Rasterizer
# ═══════════════════════════════════════════════════════════════════════════════

def _rasterize(char: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    """
    Render `char` to a GLYPH_SIZE×GLYPH_SIZE binary uint8 array (0=bg, 1=stroke).
    Glyph is centered via textbbox; binarized at BINARIZE_THRESH.
    """
    canvas = Image.new("L", (GLYPH_SIZE, GLYPH_SIZE), color=0)
    draw   = ImageDraw.Draw(canvas)
    bbox   = draw.textbbox((0, 0), char, font=font)
    ink_w  = bbox[2] - bbox[0]
    ink_h  = bbox[3] - bbox[1]
    x_off  = (GLYPH_SIZE - ink_w) // 2 - bbox[0]
    y_off  = (GLYPH_SIZE - ink_h) // 2 - bbox[1]
    draw.text((x_off, y_off), char, fill=255, font=font)
    gray = np.array(canvas, dtype=np.uint8)
    _, binary = cv2.threshold(gray, BINARIZE_THRESH, 1, cv2.THRESH_BINARY)
    return binary


def _matrix_hash(matrix: np.ndarray) -> bytes:
    """SHA-256 of the raw flattened binary matrix bytes."""
    return hashlib.sha256(matrix.tobytes()).digest()


# ═══════════════════════════════════════════════════════════════════════════════
# Optical Transform & Superposition Suite
# ═══════════════════════════════════════════════════════════════════════════════

def _transforms(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Return the three deterministic spatial transforms of a binary matrix."""
    return {
        "flip_h":     np.fliplr(matrix),
        "flip_v":     np.flipud(matrix),
        "rotate_180": np.rot90(matrix, k=2),
    }


def _interference_suite(
    original:   np.ndarray,
    t_suite:    dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Compute cv2.absdiff interference matrices between original and each transform.
    For binary {0,1} arrays: interference[i,j] = 1 iff pixels differ.
    """
    return {name: cv2.absdiff(original, t) for name, t in t_suite.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Coherence Evaluator (threshold mode)
# ═══════════════════════════════════════════════════════════════════════════════

def _border_pixel_count(matrix: np.ndarray) -> int:
    return int(
        matrix[0, :].sum()
        + matrix[-1, :].sum()
        + matrix[1:-1, 0].sum()
        + matrix[1:-1, -1].sum()
    )


def _check_coherence(char: str, matrix: np.ndarray) -> Optional[str]:
    """
    Geometric coherence suite. Returns None on pass, "U+XXXX" on fail.

    Checks (short-circuit, cheapest first):
      1. Stroke density  — empty/solid renders.
      2. Tofu-box        — missing-glyph border rectangle.
      3. Component count — noisy/fragmented renders.
      4. Interference    — implausible symmetry or degenerate geometry.
    """
    total        = matrix.size
    stroke       = int(matrix.sum())
    stroke_ratio = stroke / total
    hex_code     = f"U+{ord(char):04X}"

    if not (STROKE_RATIO_MIN <= stroke_ratio <= STROKE_RATIO_MAX):
        return hex_code

    border_fraction = _border_pixel_count(matrix) / stroke
    if border_fraction > BORDER_FRACTION_MAX:
        return hex_code

    num_labels, _ = cv2.connectedComponents(matrix, connectivity=8)
    if not (1 <= num_labels - 1 <= COMPONENT_MAX):
        return hex_code

    t_suite = _transforms(matrix)
    i_suite = _interference_suite(matrix, t_suite)
    means   = [float(m.mean()) for m in i_suite.values()]

    if all(m < INTERFERENCE_MEAN_FLOOR for m in means):
        return hex_code
    if any(m > INTERFERENCE_MEAN_MAX for m in means):
        return hex_code

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Strict Mode — Reference Profile
# ═══════════════════════════════════════════════════════════════════════════════

def build_reference_profiles(output_path: str, verbose: bool = True) -> dict:
    """
    Render every character in the CJK block using the reference font, compute the
    SHA-256 fingerprint of each 128×128 binary matrix, and save to a compressed
    .npz archive.

    This is a one-time operation (~8 s).  The resulting file locks in the exact
    pixel-level rendering of every CJK code point under this font configuration.
    In strict mode, any deviation — caused by font substitution, rendering-parameter
    tampering, or injected fallback glyphs — is detected immediately.

    Args:
        output_path: Destination path for the .npz profile archive.
        verbose:     Print progress to stderr.

    Returns:
        {"built": N, "skipped": K, "path": output_path}
        where N = characters successfully fingerprinted,
              K = characters that failed coherence (written with a zero digest —
                  strict mode will flag any payload containing them).
    """
    font = _load_font()
    code_points: list[int] = []
    digests:     list[bytes] = []
    skipped = 0

    total_chars = CJK_END - CJK_START + 1

    for i, cp in enumerate(range(CJK_START, CJK_END + 1)):
        if verbose and i % 2000 == 0:
            pct = i / total_chars * 100
            print(f"\r  building profiles: {i}/{total_chars}  ({pct:.0f}%)", end="", file=sys.stderr)

        char   = chr(cp)
        matrix = _rasterize(char, font)

        # Characters that fail coherence get a zero digest — strict mode will
        # flag any payload containing them, matching the threshold-mode behavior.
        if _check_coherence(char, matrix) is not None:
            digest = b"\x00" * 32
            skipped += 1
        else:
            digest = _matrix_hash(matrix)

        code_points.append(cp)
        digests.append(digest)

    if verbose:
        print(f"\r  building profiles: done.  {total_chars} processed, {skipped} flagged.", file=sys.stderr)

    cp_array  = np.array(code_points, dtype=np.int32)
    # Stack 32-byte digests into shape (N, 32) uint8
    dig_array = np.frombuffer(b"".join(digests), dtype=np.uint8).reshape(len(digests), 32)

    out = Path(output_path)
    np.savez_compressed(str(out), code_points=cp_array, digests=dig_array)

    built = total_chars - skipped
    if verbose:
        print(f"  saved → {out}  ({out.stat().st_size // 1024} KB)", file=sys.stderr)

    return {"built": built, "skipped": skipped, "path": str(out)}


def load_profiles(profile_path: str) -> None:
    """
    Load a profile archive built by build_reference_profiles() into the module
    cache.  Once loaded, validate_payload() switches to strict mode automatically.

    Args:
        profile_path: Path to a .npz file produced by build_reference_profiles().
    """
    global _profile
    data         = np.load(profile_path)
    code_points  = data["code_points"].tolist()            # list[int]
    digests_flat = data["digests"]                         # shape (N, 32), uint8
    _profile = {
        cp: digests_flat[i].tobytes()
        for i, cp in enumerate(code_points)
    }


def _check_strict(char: str, matrix: np.ndarray) -> Optional[str]:
    """
    Strict profile comparison.  Computes the SHA-256 of the current render and
    compares it byte-for-byte against the stored fingerprint.

    Returns None on exact match, "U+XXXX" on any deviation or missing entry.
    Zero-digest entries (characters that failed at profile-build time) always fail.
    """
    cp       = ord(char)
    hex_code = f"U+{cp:04X}"

    stored = _profile.get(cp)
    if stored is None:
        return hex_code                 # not in profile — reject unknown entries

    if stored == b"\x00" * 32:
        return hex_code                 # was incoherent at build time — always fail

    current = _matrix_hash(matrix)
    if current != stored:
        return hex_code                 # pixel mismatch — font or render changed

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Public Gateway
# ═══════════════════════════════════════════════════════════════════════════════

def validate_payload(input_string: str) -> dict:
    """
    Validate all CJK characters in `input_string` for geometric coherence.

    Mode selection (automatic):
      - If a profile has been loaded via load_profiles(), strict mode is active:
        each character's render must match its stored SHA-256 fingerprint exactly.
        This detects font substitution, rendering-environment tampering, and any
        pixel-level deviation from the trusted reference build.
      - Otherwise, threshold mode applies the five coherence checks against
        parametric bounds (stroke density, tofu, component count, interference).

    Execution halts at the first anomalous character.  The input string is never
    forwarded downstream in either failure path.

    Args:
        input_string: Arbitrary text payload to inspect.

    Returns:
        {"status": "pass", "code": 200}
            No CJK characters present, or all characters pass coherence evaluation.

        {"status": "fail", "code": 403, "flagged_hex": "U+XXXX"}
            First anomalous code point detected.

    Raises:
        FileNotFoundError: No CJK font found (see FONT_PATH / _FONT_FALLBACKS).
    """
    targets = _extract_cjk(input_string)
    if not targets:
        return {"status": "pass", "code": 200}

    font        = _load_font()
    strict_mode = bool(_profile)

    for char in targets:
        matrix = _rasterize(char, font)

        if strict_mode:
            anomaly = _check_strict(char, matrix)
        else:
            anomaly = _check_coherence(char, matrix)

        if anomaly is not None:
            return {"status": "fail", "code": 403, "flagged_hex": anomaly}

    return {"status": "pass", "code": 200}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="glyph_validator",
        description="CJK glyph geometric coherence validator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 glyph_validator.py "Hello 中文"
  python3 glyph_validator.py --build-profiles profiles.npz
  python3 glyph_validator.py --profiles profiles.npz "中文"
  echo '{"id":"r1","text":"中文"}' | python3 glyph_validator.py --batch
  cat input.jsonl | python3 glyph_validator.py --profiles profiles.npz --batch
""",
    )
    parser.add_argument(
        "input", nargs="?", default=None,
        help="Text string to validate (omit when using --batch or --build-profiles).",
    )
    parser.add_argument(
        "--build-profiles", metavar="OUTPUT.npz",
        help="Render every CJK character and save SHA-256 fingerprints to OUTPUT.npz.",
    )
    parser.add_argument(
        "--profiles", metavar="PROFILES.npz",
        help="Load profile archive for strict pixel-exact comparison mode.",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help=(
            "Read one JSON object per line from stdin, write one result per line to "
            'stdout. Each input line must have a "text" field; an optional "id" field '
            "is echoed in the output."
        ),
    )

    args = parser.parse_args()

    # ── Build profiles ────────────────────────────────────────────────────────
    if args.build_profiles:
        result = build_reference_profiles(args.build_profiles, verbose=True)
        print(json.dumps(result))
        sys.exit(0)

    # ── Load profiles for strict mode ─────────────────────────────────────────
    if args.profiles:
        p = Path(args.profiles)
        if not p.exists():
            print(f"error: profile file not found: {p}", file=sys.stderr)
            sys.exit(2)
        load_profiles(str(p))

    # ── Batch mode ────────────────────────────────────────────────────────────
    if args.batch:
        for line_no, raw in enumerate(sys.stdin, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(
                    json.dumps({"error": f"invalid JSON on line {line_no}: {exc}"}),
                    flush=True,
                )
                continue

            text = obj.get("text", "")
            if not isinstance(text, str):
                print(
                    json.dumps({"error": f"'text' must be a string (line {line_no})"}),
                    flush=True,
                )
                continue

            result = validate_payload(text)
            if "id" in obj:
                result["id"] = obj["id"]
            print(json.dumps(result, ensure_ascii=False), flush=True)

        sys.exit(0)

    # ── Single-string mode ────────────────────────────────────────────────────
    if args.input is None:
        parser.print_help(sys.stderr)
        sys.exit(1)

    result = validate_payload(args.input)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result["code"] == 200 else 1)


if __name__ == "__main__":
    _cli()
