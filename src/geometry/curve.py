"""Parametric curve definitions for boundary sampling.

Mirrors physicsnemo.sym.geometry.curve API — SympyCurve and Curve.
"""

import numpy as np


class SympyCurve:
    """A parametric surface/side defined by symbolic expressions.

    Each curve maps one or two parameters (u, [v]) → 3D position,
    with an associated normal vector and differential area.
    """

    def __init__(self, parametric_pos: dict, normal: dict, area: float,
                 bounds: dict, dims: int = 3):
        """
        Args:
            parametric_pos: dict mapping dim name → callable(param_dict) returning (N,1) array
            normal: dict mapping dim name → callable(param_dict) returning (N,1) array
            area: scalar area of this face
            bounds: dict mapping param name → (low, high)
            dims: 2 or 3
        """
        self.parametric_pos = parametric_pos
        self.normal = normal
        self.area = float(area)
        self.bounds = bounds
        self.dims = dims

    def sample(self, nr_points: int) -> dict:
        """Sample nr_points on this curve surface.

        Returns dict with 'x','y','z' coords, 'normal_x','normal_y','normal_z', and 'area'.
        """
        # Sample parameter values uniformly
        param_vals = {}
        for pname, (lo, hi) in self.bounds.items():
            param_vals[pname] = np.random.uniform(lo, hi, (nr_points, 1))

        result = {"area": np.full((nr_points, 1), self.area / max(nr_points, 1))}

        for dim, fn in self.parametric_pos.items():
            result[dim] = fn(param_vals)

        for dim, fn in self.normal.items():
            result[f"normal_{dim}"] = fn(param_vals)

        return result


class Curve:
    """A collection of SympyCurve objects forming a complete boundary."""

    def __init__(self, curves: list, dims: int = 3):
        self.curves = curves
        self.dims = dims
        self._total_area = sum(c.area for c in curves) if curves else 0.0

    def sample(self, nr_points: int) -> dict:
        """Distribute nr_points across curves proportional to area, then sample each."""
        if not self.curves:
            dim_names = ["x", "y", "z"][:self.dims]
            result = {"area": np.zeros((nr_points, 1))}
            for d in dim_names:
                result[d] = np.zeros((nr_points, 1))
                result[f"normal_{d}"] = np.zeros((nr_points, 1))
            return result

        if self._total_area <= 0:
            # Equal distribution
            per_curve = nr_points // len(self.curves)
            remainder = nr_points % len(self.curves)
            all_samples = []
            for i, curve in enumerate(self.curves):
                n = per_curve + (1 if i < remainder else 0)
                if n > 0:
                    all_samples.append(curve.sample(n))
        else:
            all_samples = []
            remaining = nr_points
            for i, curve in enumerate(self.curves):
                if i == len(self.curves) - 1:
                    n = remaining
                else:
                    n = max(1, int(nr_points * curve.area / self._total_area))
                    n = min(n, remaining - (len(self.curves) - i - 1))
                if n > 0:
                    all_samples.append(curve.sample(n))
                    remaining -= n

        # Concatenate
        keys = all_samples[0].keys()
        return {k: np.concatenate([s[k] for s in all_samples], axis=0) for k in keys}
