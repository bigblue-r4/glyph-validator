"""Smoke test for an INSTALLED glyph-validator.

Run this from a directory that is not the repo, against a wheel that has
already been pip-installed:

    cd "$RUNNER_TEMP"
    cp "$GITHUB_WORKSPACE/smoke_test.py" .
    python smoke_test.py

The working directory matters. Python puts the script's own directory on
sys.path, so running this out of the checkout would import glyph_validator.py
from source and prove nothing about the packaged artifact — the console script,
the declared numpy/opencv/Pillow dependencies and the packaging itself would all
go unexercised. Copying the file out and running it there is what keeps the test
honest.

Lives in one file because both workflows run it: ci.yml on every push and pull
request, release.yml before it publishes to PyPI. The assertions existed only
inside release.yml before CI was added, and duplicating them into a second
workflow would have left two copies to drift apart — with the copy that gates
publishing being the one nobody notices going stale.

validate_payload() rasterises each CJK glyph, so a real CJK face must be
installed. On Debian/Ubuntu that is fonts-noto-cjk, which lands exactly where
glyph_validator.FONT_PATH looks first.
"""

import sys
from importlib.metadata import version

import glyph_validator as g

PASS = {"status": "pass", "code": 200}


def main() -> int:
    print("installed version:", version("glyph-validator"))
    print("module file      :", g.__file__)

    # Guard the isolation property this script depends on: if the module
    # resolved to the repo checkout, everything below is testing source rather
    # than the artifact, and the run should fail loudly instead of passing for
    # the wrong reason.
    if "site-packages" not in g.__file__:
        print(
            "ERROR: glyph_validator was imported from source, not from the "
            f"installed wheel ({g.__file__}). Run this from outside the repo.",
            file=sys.stderr,
        )
        return 1

    # No CJK in the payload: returns before a font is ever loaded, so this half
    # of the API works with no font installed at all.
    assert g.validate_payload("hello world") == PASS

    # Real CJK: rasterises, runs the five coherence checks, passes.
    r = g.validate_payload("安全性能")
    assert r == PASS, r

    # Mixed script is the case this package exists for.
    r = g.validate_payload("login 安全")
    assert r == PASS, r

    # Empty and whitespace must not raise — they reach the validator from real
    # payloads more often than anything interesting does.
    assert g.validate_payload("") == PASS
    assert g.validate_payload("   ") == PASS

    print("api OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
