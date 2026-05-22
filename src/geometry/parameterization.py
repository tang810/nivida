"""Parameterization support for geometry primitives.

Mirrors physicsnemo.sym.geometry.parameterization API.
"""

import numpy as np


class Parameter:
    """A named parameter that can vary geometry properties."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"Parameter({self.name!r})"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Parameter):
            return self.name == other.name
        return False


class Bounds:
    """Axis-aligned bounding box defined by per-dimension (low, high) ranges."""

    def __init__(self, bounds_dict: dict):
        self.bounds = bounds_dict  # {str: (low, high)}

    def __repr__(self):
        return f"Bounds({self.bounds!r})"

    def as_array(self, dims):
        """Return bounds as (2, ndim) array [low, high] per dim."""
        arr = np.zeros((2, len(dims)))
        for i, d in enumerate(dims):
            lo, hi = self.bounds.get(str(d), self.bounds.get(d, (-1.0, 1.0)))
            arr[0, i] = lo
            arr[1, i] = hi
        return arr


class Parameterization:
    """Maps named Parameters to their numeric ranges."""

    def __init__(self, param_ranges: dict = None):
        # param_ranges: {Parameter: (low, high)} or {Parameter: float}
        self.param_ranges = param_ranges or {}

    def __repr__(self):
        return f"Parameterization({self.param_ranges!r})"

    def sample(self, nr_points: int) -> dict:
        """Draw `nr_points` random values for each parameter uniformly from its range."""
        result = {}
        for param, rng in self.param_ranges.items():
            if isinstance(rng, (tuple, list)):
                lo, hi = rng
                result[param.name] = np.random.uniform(lo, hi, (nr_points, 1))
            else:
                # Fixed scalar value
                result[param.name] = np.full((nr_points, 1), rng, dtype=np.float64)
        return result
