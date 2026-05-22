"""Base Geometry class supporting CSG operations, transforms, and sampling.

Mirrors the physicsnemo.sym.geometry.geometry.Geometry API.
"""

from __future__ import annotations

import numpy as np

from .curve import Curve
from .helper import _sdf_derivatives
from .parameterization import Parameterization, Bounds


class Geometry:
    """Base class for all geometric primitives and CSG combinations.

    Supports:
    - Boolean CSG: + (union), & (intersection), - (subtraction)
    - Transforms: translate, scale, rotate, repeat
    - Boundary and interior point sampling with SDF computation
    - Parameterization for variable geometry
    """

    def __init__(self, curves: Curve, sdf_func, dims: int, bounds: Bounds,
                 parameterization: Parameterization = None):
        self._curves = curves
        self._sdf_func = sdf_func  # callable(dict) -> (N,1) array
        self.dims = dims
        self.bounds = bounds
        self.parameterization = parameterization or Parameterization()

    # ── SDF evaluation ──────────────────────────────────────────────

    def sdf(self, points: dict, params: dict = None) -> dict:
        """Evaluate signed distance field at given coordinate points.

        Args:
            points: {'x': (N,1), 'y': (N,1), ...}
            params: optional parameter values dict

        Returns:
            dict with at least {'sdf': (N,1) array}
        """
        return {"sdf": np.asarray(self._sdf_func(points, params), dtype=np.float64)}

    # ── Boundary sampling ───────────────────────────────────────────

    def sample_boundary(self, nr_points: int,
                        parameterization: Parameterization = None) -> dict:
        """Sample points on the geometry boundary.

        Args:
            nr_points: number of points to sample
            parameterization: optional override parameterization

        Returns:
            dict with 'x','y','z' coords, 'normal_x','normal_y','normal_z', and 'area'
        """
        s = self._curves.sample(nr_points)
        # Ensure all spatial dims are present
        dim_names = ["x", "y", "z"][:self.dims]
        for d in dim_names:
            if d not in s:
                s[d] = np.zeros((nr_points, 1))
            if f"normal_{d}" not in s:
                s[f"normal_{d}"] = np.zeros((nr_points, 1))
        return s

    # ── Interior sampling ───────────────────────────────────────────

    def sample_interior(self, nr_points: int,
                        compute_sdf_derivatives: bool = False,
                        parameterization: Parameterization = None) -> dict:
        """Sample points inside the geometry volume.

        Uses bounding-box rejection sampling with the SDF.

        Args:
            nr_points: target number of interior points
            compute_sdf_derivatives: if True, also compute SDF spatial gradients
            parameterization: optional override

        Returns:
            dict with 'x','y','z' coords, optional 'sdf' and 'sdf__x', etc.
        """
        dim_names = ["x", "y", "z"][:self.dims]
        bounds_arr = self.bounds.as_array(dim_names)

        # Rejection sampling — sample in bounding box, keep SDF <= 0
        factor = 3  # oversample factor for rejection
        collected = {d: [] for d in dim_names}
        needed = nr_points

        while needed > 0:
            batch = max(needed * factor, 1000)
            pts = {}
            for i, d in enumerate(dim_names):
                lo, hi = bounds_arr[0, i], bounds_arr[1, i]
                pts[d] = np.random.uniform(lo, hi, (batch, 1))

            sdf_vals = np.asarray(self._sdf_func(pts), dtype=np.float64).ravel()
            inside = sdf_vals <= 0.0

            if np.any(inside):
                take = min(needed, inside.sum())
                idx = np.where(inside)[0][:take]
                for d in dim_names:
                    collected[d].append(pts[d][idx])
                needed -= take

        result = {d: np.concatenate(collected[d], axis=0) for d in dim_names}

        if compute_sdf_derivatives:
            derivs = _sdf_derivatives(self._sdf_func, result, self.dims)
            result.update(derivs)

        return result

    # ── Boolean CSG operations ──────────────────────────────────────

    def __add__(self, other: Geometry) -> Geometry:
        """CSG union: min(sdf_self, sdf_other)."""
        return self._csg_op(other, "union")

    def __and__(self, other: Geometry) -> Geometry:
        """CSG intersection: max(sdf_self, sdf_other)."""
        return self._csg_op(other, "intersection")

    def __sub__(self, other: Geometry) -> Geometry:
        """CSG subtraction: max(sdf_self, -sdf_other)."""
        return self._csg_op(other, "subtraction")

    def _csg_op(self, other: Geometry, op: str) -> Geometry:
        """Create a new Geometry from a CSG boolean operation."""
        dims = max(self.dims, other.dims)
        sdf_a = self._sdf_func
        sdf_b = other._sdf_func

        if op == "union":
            def combined_sdf(pts, params=None):
                return np.minimum(
                    np.asarray(sdf_a(pts), dtype=np.float64),
                    np.asarray(sdf_b(pts), dtype=np.float64),
                )
        elif op == "intersection":
            def combined_sdf(pts, params=None):
                return np.maximum(
                    np.asarray(sdf_a(pts), dtype=np.float64),
                    np.asarray(sdf_b(pts), dtype=np.float64),
                )
        elif op == "subtraction":
            def combined_sdf(pts, params=None):
                return np.maximum(
                    np.asarray(sdf_a(pts), dtype=np.float64),
                    -np.asarray(sdf_b(pts), dtype=np.float64),
                )
        else:
            raise ValueError(f"Unknown CSG op: {op}")

        # Combine curves (union of both boundary sets)
        curves_a = list(self._curves.curves) if self._curves else []
        curves_b = list(other._curves.curves) if other._curves else []
        combined_curves = Curve(curves_a + curves_b, dims)

        # Combine bounds (take the union of bounding boxes)
        dim_names = ["x", "y", "z"][:dims]
        ba = self.bounds.as_array(dim_names)
        bb = other.bounds.as_array(dim_names)
        combined_bounds_dict = {}
        for i, d in enumerate(dim_names):
            lo = min(ba[0, i], bb[0, i])
            hi = max(ba[1, i], bb[1, i])
            combined_bounds_dict[d] = (lo, hi)
        combined_bounds = Bounds(combined_bounds_dict)

        return Geometry(combined_curves, combined_sdf, dims, combined_bounds)

    # ── Transforms ──────────────────────────────────────────────────

    def translate(self, translation: tuple) -> Geometry:
        """Translate geometry by (dx, dy) or (dx, dy, dz)."""
        t = np.asarray(translation, dtype=np.float64)
        orig_sdf = self._sdf_func
        dim_names = ["x", "y", "z"][:self.dims]

        def translated_sdf(pts, params=None):
            shifted = {}
            for i, d in enumerate(dim_names):
                shifted[d] = pts[d] - (t[i] if i < len(t) else 0.0)
            return orig_sdf(shifted)

        new_bounds_dict = {}
        ba = self.bounds.as_array(dim_names)
        for i, d in enumerate(dim_names):
            offset = t[i] if i < len(t) else 0.0
            new_bounds_dict[d] = (ba[0, i] + offset, ba[1, i] + offset)

        return Geometry(self._curves, translated_sdf, self.dims,
                        Bounds(new_bounds_dict), self.parameterization)

    def scale(self, scale_factor) -> Geometry:
        """Scale geometry uniformly by scale_factor (scalar or per-axis tuple)."""
        s = np.asarray(scale_factor, dtype=np.float64)
        if s.ndim == 0:
            s = np.full(self.dims, s)
        orig_sdf = self._sdf_func
        dim_names = ["x", "y", "z"][:self.dims]

        def scaled_sdf(pts, params=None):
            scaled_pts = {}
            for i, d in enumerate(dim_names):
                si = s[i] if i < len(s) else 1.0
                scaled_pts[d] = pts[d] / max(si, 1e-12)
            return np.asarray(orig_sdf(scaled_pts), dtype=np.float64) * np.mean(s)

        new_bounds_dict = {}
        ba = self.bounds.as_array(dim_names)
        for i, d in enumerate(dim_names):
            si = s[i] if i < len(s) else 1.0
            new_bounds_dict[d] = (ba[0, i] * si, ba[1, i] * si)

        return Geometry(self._curves, scaled_sdf, self.dims,
                        Bounds(new_bounds_dict), self.parameterization)

    def rotate(self, angle: float, axis: str = "z") -> Geometry:
        """Rotate geometry by `angle` (radians) around given axis ('x','y','z')."""
        c = np.cos(angle)
        s = np.sin(angle)
        orig_sdf = self._sdf_func
        dim_names = ["x", "y", "z"][:self.dims]

        axis_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]

        def rotated_sdf(pts, params=None):
            # Build rotation matrix and apply inverse to points
            p = np.column_stack([pts[d].ravel() for d in dim_names])  # (N, D)
            R = np.eye(self.dims)
            if axis_idx == 0:  # rotate around x → yz plane
                if self.dims >= 3:
                    R[1, 1] = c; R[1, 2] = -s
                    R[2, 1] = s; R[2, 2] = c
                elif self.dims == 2:
                    R[0, 0] = c; R[0, 1] = -s
                    R[1, 0] = s; R[1, 1] = c
            elif axis_idx == 1:  # rotate around y → xz plane
                if self.dims >= 3:
                    R[0, 0] = c; R[0, 2] = s
                    R[2, 0] = -s; R[2, 2] = c
            elif axis_idx == 2:  # rotate around z → xy plane
                R[0, 0] = c; R[0, 1] = -s
                R[1, 0] = s; R[1, 1] = c

            p_rot = (R.T @ p.T).T  # inverse rotation
            rotated_pts = {d: p_rot[:, i].reshape(-1, 1) for i, d in enumerate(dim_names)}
            return orig_sdf(rotated_pts)

        return Geometry(self._curves, rotated_sdf, self.dims,
                        self.bounds, self.parameterization)

    def repeat(self, spacing: float, repeat_lower: tuple, repeat_higher: tuple) -> Geometry:
        """Periodically tile the geometry along each axis (infinite repetition)."""
        sp = np.asarray(spacing, dtype=np.float64)
        if sp.ndim == 0:
            sp = np.full(self.dims, sp)
        rl = np.asarray(repeat_lower, dtype=np.float64)
        rh = np.asarray(repeat_higher, dtype=np.float64)
        orig_sdf = self._sdf_func
        dim_names = ["x", "y", "z"][:self.dims]

        def repeated_sdf(pts, params=None):
            result = np.full((list(pts.values())[0].shape[0], 1), np.inf)
            for ix in range(int(rl[0]), int(rh[0]) + 1):
                for iy in range(int(rl[1]) if len(rl) > 1 else 0,
                                int(rh[1]) + 1 if len(rh) > 1 else 1):
                    for iz in range(int(rl[2]) if len(rl) > 2 else 0,
                                    int(rh[2]) + 1 if len(rh) > 2 else 1):
                        offsets = [ix * sp[0], iy * sp[1] if len(sp) > 1 else 0,
                                   iz * sp[2] if len(sp) > 2 else 0]
                        shifted = {}
                        for j, d in enumerate(dim_names):
                            shifted[d] = pts[d] - offsets[j]
                        sdf = np.asarray(orig_sdf(shifted), dtype=np.float64)
                        result = np.minimum(result, sdf)
            return result

        # Expand bounds
        new_bounds_dict = {}
        ba = self.bounds.as_array(dim_names)
        for i, d in enumerate(dim_names):
            ri_low = int(rl[i]) if i < len(rl) else 0
            ri_high = int(rh[i]) if i < len(rh) else 0
            sp_i = sp[i] if i < len(sp) else 1.0
            new_lo = ba[0, i] + ri_low * sp_i
            new_hi = ba[1, i] + ri_high * sp_i
            new_bounds_dict[d] = (min(new_lo, ba[0, i]), max(new_hi, ba[1, i]))

        return Geometry(self._curves, repeated_sdf, self.dims,
                        Bounds(new_bounds_dict), self.parameterization)
