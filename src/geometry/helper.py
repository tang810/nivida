"""Helper utilities for the geometry module.

Includes csg_curve_naming and _sympy_sdf_to_sdf helpers
mirroring physicsnemo.sym.geometry.helper.
"""

import numpy as np


def csg_curve_naming(dim: int) -> list:
    """Return standard curve parameter names for a given dimension.

    2D → ['h'], 3D → ['h', 'r', 'theta']
    """
    if dim == 2:
        return ["h"]
    elif dim == 3:
        return ["h", "r", "theta"]
    return []


def _sympy_sdf_to_sdf(sympy_fn):
    """Convert a sympy-based SDF expression to a numpy-callable function.

    If the input is already a callable returning numpy, it is returned as-is.
    """
    if callable(sympy_fn):
        return sympy_fn
    # If it's a sympy expression, try to lambdify
    try:
        import sympy
        if hasattr(sympy_fn, "free_symbols"):
            symbols = sorted(sympy_fn.free_symbols, key=lambda s: s.name)
            import sympy
            return sympy.lambdify(symbols, sympy_fn, "numpy")
    except ImportError:
        pass
    return sympy_fn


def _sdf_derivatives(sdf_func, points: dict, dims: int, eps: float = 1e-6) -> dict:
    """Compute SDF spatial derivatives via finite differences.

    Args:
        sdf_func: callable(points_dict) → (N,1) sdf values
        points: dict of coord arrays {'x': (N,1), 'y': (N,1), ...}
        dims: number of spatial dimensions
        eps: finite-difference step

    Returns:
        dict with keys 'sdf', 'sdf__x', 'sdf__y', 'sdf__z' (as applicable)
    """
    dim_names = ["x", "y", "z"][:dims]
    sdf0 = np.asarray(sdf_func(points), dtype=np.float64)

    result = {"sdf": sdf0}

    for i, d in enumerate(dim_names):
        pts_plus = {k: v.copy() for k, v in points.items()}
        pts_plus[d] = pts_plus[d] + eps
        sdf_plus = np.asarray(sdf_func(pts_plus), dtype=np.float64)
        result[f"sdf__{d}"] = (sdf_plus - sdf0) / eps

    return result
