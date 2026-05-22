"""2D geometry primitives: Rectangle, Circle, Line, Polygon.

Mirrors physicsnemo.sym.geometry.primitives_2d API.
"""

import numpy as np

from .geometry import Geometry
from .curve import Curve, SympyCurve
from .parameterization import Parameter, Parameterization, Bounds


def _sdf_rectangle(pts, x1, y1, x2, y2):
    """SDF for axis-aligned rectangle [x1,x2]×[y1,y2]."""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    hx = (x2 - x1) / 2.0
    hy = (y2 - y1) / 2.0
    qx = np.abs(pts["x"] - cx) - hx
    qy = np.abs(pts["y"] - cy) - hy
    outside = np.maximum(qx, qy)
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside


class Rectangle(Geometry):
    """2D rectangle defined by two opposite corner points."""

    def __init__(self, point_1: tuple, point_2: tuple,
                 parameterization: Parameterization = None):
        x1, y1 = float(point_1[0]), float(point_1[1])
        x2, y2 = float(point_2[0]), float(point_2[1])
        self.point_1 = (x1, y1)
        self.point_2 = (x2, y2)

        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)

        # Four edges as SympyCurves
        curves = _rectangle_curves(xmin, xmax, ymin, ymax)
        c = Curve(curves, dims=2)

        def sdf_func(pts, params=None):
            return _sdf_rectangle(pts, xmin, ymin, xmax, ymax)

        bounds = Bounds({"x": (xmin - 0.1, xmax + 0.1),
                         "y": (ymin - 0.1, ymax + 0.1)})

        super().__init__(c, sdf_func, dims=2, bounds=bounds,
                         parameterization=parameterization)


def _rectangle_curves(xmin, xmax, ymin, ymax):
    w = xmax - xmin
    h = ymax - ymin
    curves = []

    # Bottom edge: y = ymin, normal = (0, -1)
    curves.append(SympyCurve(
        parametric_pos={
            "x": lambda p: xmin + p["h"] * w,
            "y": lambda p: np.full_like(p["h"], ymin),
        },
        normal={
            "x": lambda p: np.zeros_like(p["h"]),
            "y": lambda p: -np.ones_like(p["h"]),
        },
        area=w,
        bounds={"h": (0.0, 1.0)},
        dims=2,
    ))
    # Top edge: y = ymax, normal = (0, 1)
    curves.append(SympyCurve(
        parametric_pos={
            "x": lambda p: xmin + p["h"] * w,
            "y": lambda p: np.full_like(p["h"], ymax),
        },
        normal={
            "x": lambda p: np.zeros_like(p["h"]),
            "y": lambda p: np.ones_like(p["h"]),
        },
        area=w,
        bounds={"h": (0.0, 1.0)},
        dims=2,
    ))
    # Left edge: x = xmin, normal = (-1, 0)
    curves.append(SympyCurve(
        parametric_pos={
            "x": lambda p: np.full_like(p["h"], xmin),
            "y": lambda p: ymin + p["h"] * h,
        },
        normal={
            "x": lambda p: -np.ones_like(p["h"]),
            "y": lambda p: np.zeros_like(p["h"]),
        },
        area=h,
        bounds={"h": (0.0, 1.0)},
        dims=2,
    ))
    # Right edge: x = xmax, normal = (1, 0)
    curves.append(SympyCurve(
        parametric_pos={
            "x": lambda p: np.full_like(p["h"], xmax),
            "y": lambda p: ymin + p["h"] * h,
        },
        normal={
            "x": lambda p: np.ones_like(p["h"]),
            "y": lambda p: np.zeros_like(p["h"]),
        },
        area=h,
        bounds={"h": (0.0, 1.0)},
        dims=2,
    ))
    return curves


def _resolve_param(value, params):
    """Resolve a Parameter or numeric value, optionally from a Parameterization or dict."""
    if isinstance(value, Parameter):
        if params is not None:
            if isinstance(params, Parameterization):
                rng = params.param_ranges.get(value)
                if rng is not None:
                    if isinstance(rng, (tuple, list)):
                        return (rng[0] + rng[1]) / 2.0
                    return float(rng)
            elif isinstance(params, dict) and value.name in params:
                v = params[value.name]
                return float(np.asarray(v).ravel()[0])
            elif hasattr(params, 'get') and params.get(value.name) is not None:
                v = params[value.name]
                return float(np.asarray(v).ravel()[0])
        return 0.0
    return float(value) if value is not None else 0.0


class Circle(Geometry):
    """2D circle defined by center and radius.

    Center coordinates can be Parameter objects for parametric variation.
    """

    def __init__(self, center: tuple, radius: float,
                 parameterization: Parameterization = None):
        self.radius = float(radius)

        # Resolve center coordinates — they may be Parameters
        cx_raw, cy_raw = center[0], center[1]
        if isinstance(cx_raw, str):
            cx_raw = Parameter(cx_raw)
        if isinstance(cy_raw, str):
            cy_raw = Parameter(cy_raw)

        self._cx_param = cx_raw if isinstance(cx_raw, Parameter) else None
        self._cy_param = cy_raw if isinstance(cy_raw, Parameter) else None

        # Build parameterization — start with any passed-in, ensure center params are included
        if parameterization is None:
            parameterization = Parameterization()
        if self._cx_param and self._cx_param not in parameterization.param_ranges:
            parameterization.param_ranges[self._cx_param] = (-1.0, 1.0)
        if self._cy_param and self._cy_param not in parameterization.param_ranges:
            parameterization.param_ranges[self._cy_param] = (-1.0, 1.0)

        # Default numeric values for bounds/curves
        cx_nominal = _resolve_param(cx_raw, parameterization)
        cy_nominal = _resolve_param(cy_raw, parameterization)
        self.center = (cx_nominal, cy_nominal)

        curves = _circle_curve(cx_nominal, cy_nominal, radius)
        c = Curve(curves, dims=2)

        def sdf_func(pts, params=None):
            _cx = _resolve_param(cx_raw, params) if params and (
                self._cx_param or self._cy_param) else cx_nominal
            _cy = _resolve_param(cy_raw, params) if params and (
                self._cx_param or self._cy_param) else cy_nominal
            dx = pts["x"] - _cx
            dy = pts["y"] - _cy
            return np.sqrt(dx ** 2 + dy ** 2) - radius

        rp = radius + 0.5
        bounds = Bounds({
            "x": (cx_nominal - rp, cx_nominal + rp),
            "y": (cy_nominal - rp, cy_nominal + rp),
        })

        super().__init__(c, sdf_func, dims=2, bounds=bounds,
                         parameterization=parameterization)


def _circle_curve(cx, cy, r, n_segments=64):
    """Approximate circle boundary with a single parametric curve."""
    circumference = 2.0 * np.pi * r

    def pos_fn(p):
        theta = p["h"] * 2.0 * np.pi
        return cx + r * np.cos(theta)

    def pos_y(p):
        theta = p["h"] * 2.0 * np.pi
        return cy + r * np.sin(theta)

    def normal_x(p):
        theta = p["h"] * 2.0 * np.pi
        return np.cos(theta)

    def normal_y(p):
        theta = p["h"] * 2.0 * np.pi
        return np.sin(theta)

    return [SympyCurve(
        parametric_pos={"x": pos_fn, "y": pos_y},
        normal={"x": normal_x, "y": normal_y},
        area=circumference,
        bounds={"h": (0.0, 1.0)},
        dims=2,
    )]


class Line(Geometry):
    """2D line segment."""

    def __init__(self, point_1: tuple, point_2: tuple,
                 parameterization: Parameterization = None):
        x1, y1 = float(point_1[0]), float(point_1[1])
        x2, y2 = float(point_2[0]), float(point_2[1])
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        self.point_1 = (x1, y1)
        self.point_2 = (x2, y2)

        nx = -dy / length if length > 0 else 0.0
        ny = dx / length if length > 0 else 1.0

        curve = SympyCurve(
            parametric_pos={
                "x": lambda p: x1 + p["h"] * dx,
                "y": lambda p: y1 + p["h"] * dy,
            },
            normal={"x": lambda p: np.full_like(p["h"], nx),
                    "y": lambda p: np.full_like(p["h"], ny)},
            area=length,
            bounds={"h": (0.0, 1.0)},
            dims=2,
        )
        c = Curve([curve], dims=2)

        def sdf_func(pts, params=None):
            px = pts["x"] - x1
            py = pts["y"] - y1
            t = np.clip((px * dx + py * dy) / (length ** 2 + 1e-12), 0.0, 1.0)
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            return np.sqrt((pts["x"] - closest_x) ** 2 +
                           (pts["y"] - closest_y) ** 2)

        margin = 0.5
        bounds = Bounds({
            "x": (min(x1, x2) - margin, max(x1, x2) + margin),
            "y": (min(y1, y2) - margin, max(y1, y2) + margin),
        })

        super().__init__(c, sdf_func, dims=2, bounds=bounds,
                         parameterization=parameterization)


class Polygon(Geometry):
    """2D polygon defined by a list of vertices."""

    def __init__(self, vertices: list, parameterization: Parameterization = None):
        verts = np.asarray(vertices, dtype=np.float64)
        self.vertices = verts

        curves = _polygon_curves(verts)
        c = Curve(curves, dims=2)

        def sdf_func(pts, params=None):
            return _polygon_sdf(pts, verts)

        xs = verts[:, 0]
        ys = verts[:, 1]
        margin = 0.5
        bounds = Bounds({"x": (xs.min() - margin, xs.max() + margin),
                         "y": (ys.min() - margin, ys.max() + margin)})

        super().__init__(c, sdf_func, dims=2, bounds=bounds,
                         parameterization=parameterization)


def _polygon_curves(verts):
    n = len(verts)
    curves = []
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        nx = -dy / length if length > 0 else 0.0
        ny = dx / length if length > 0 else 1.0

        curves.append(SympyCurve(
            parametric_pos={
                "x": lambda p, _x1=x1, _dx=dx: _x1 + p["h"] * _dx,
                "y": lambda p, _y1=y1, _dy=dy: _y1 + p["h"] * _dy,
            },
            normal={
                "x": lambda p, _nx=nx: np.full_like(p["h"], _nx),
                "y": lambda p, _ny=ny: np.full_like(p["h"], _ny),
            },
            area=length,
            bounds={"h": (0.0, 1.0)},
            dims=2,
        ))
    return curves


def _polygon_sdf(pts, verts):
    """Compute 2D polygon SDF using winding number and closest edge distance."""
    px = pts["x"].ravel()
    py = pts["y"].ravel()
    n_pts = len(px)
    n_verts = len(verts)

    # Compute winding number to determine inside/outside
    sdf = np.full(n_pts, np.inf)
    for i in range(n_verts):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n_verts]
        dx = x2 - x1
        dy = y2 - y1
        length2 = dx**2 + dy**2

        t = np.clip(((px - x1) * dx + (py - y1) * dy) / (length2 + 1e-12), 0.0, 1.0)
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        dist = np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
        sdf = np.minimum(sdf, dist)

    # Determine inside/outside with winding number
    angle_sum = np.zeros(n_pts)
    for i in range(n_verts):
        x1, y1 = verts[i] - np.column_stack([px, py])
        x2, y2 = verts[(i + 1) % n_verts] - np.column_stack([px, py])
        a1 = np.arctan2(y1, x1)
        a2 = np.arctan2(y2, x2)
        da = a2 - a1
        da = np.where(da > np.pi, da - 2 * np.pi, da)
        da = np.where(da < -np.pi, da + 2 * np.pi, da)
        angle_sum += da

    inside = np.abs(angle_sum) > np.pi
    sdf = np.where(inside, -sdf, sdf)
    return sdf.reshape(-1, 1)
