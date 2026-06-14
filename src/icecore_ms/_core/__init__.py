"""
Python wrapper around the C++ native signal processing extension.

This module delegates heavy numerical work to the compiled native library
while providing idiomatic Python interfaces with NumPy interoperability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

try:
    from .. import _icecore_native as _native  # type: ignore[attr-defined]

    _HAS_NATIVE = True
except ImportError:  # pragma: no cover
    _native = None  # type: ignore[assignment]
    _HAS_NATIVE = False


def has_native_extension() -> bool:
    """Check whether the compiled C++ extension is available."""
    return _HAS_NATIVE


def require_native_extension() -> None:
    """Raise if the compiled C++ extension is not available."""
    if not _HAS_NATIVE:
        raise ImportError(
            "The icecore-ms native C++ extension (_icecore_native) is not built. "
            "Please build the package first via `pip install -e .` or "
            "`python setup.py build_ext --inplace`."
        )


__all__ = ["has_native_extension", "require_native_extension", "_native"]
