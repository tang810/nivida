"""3D geometry primitives: Box, Sphere, Cylinder, Plane.

Mirrors physicsnemo.sym.geometry.primitives_3d API.
"""

import numpy as np

from .geometry import Geometry
from .curve import Curve, SympyCurve
from .parameterization import Parameterization, Bounds


# ── Box ─────────────────────────────────────────────────────────────

class Box(Geometry):
    """3D axis-aligned box defined by two opposite corner points."""

    def __init__(self, point_1: tuple, point_2: tuple,
                 parameterization: Parameterization = None):
        x1, y1, z1 = float(point_1[0]), float(point_1[1]), float(point_1[2])
        x2, y2, z2 = float(point_2[0]), float(point_2[1]), float(point_2[2])
        self.point_1 = (x1, y1, z1)
        self.point_2 = (x2, y2, z2)

        self.xmin, self.xmax = min(x1, x2), max(x1, x2)
        self.ymin, self.ymax = min(y1, y2), max(y1, y2)
        self.zmin, self.zmax = min(z1, z2), max(z1, z2)

        curves = _box_curves(self.xmin, self.xmax, self.ymin, self.ymax,
                             self.zmin, self.zmax)
        c = Curve(curves, dims=3)

        def sdf_func(pts, params=None):
            return _sdf_box(
                pts, self.xmin, self.xmax, self.ymin, self.ymax,
                self.zmin, self.zmax)

        margin = 0.5
        bounds = Bounds({
            "x": (self.xmin - margin, self.xmax + margin),
            "y": (self.ymin - margin, self.ymax + margin),
            "z": (self.zmin - margin, self.zmax + margin),
        })

        super().__init__(c, sdf_func, dims=3, bounds=bounds,
                         parameterization=parameterization)


def _sdf_box(pts, xmin, xmax, ymin, ymax, zmin, zmax):
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cz = (zmin + zmax) / 2.0
    hx = (xmax - xmin) / 2.0
    hy = (ymax - ymin) / 2.0
    hz = (zmax - zmin) / 2.0
    qx = np.abs(pts["x"] - cx) - hx
    qy = np.abs(pts["y"] - cy) - hy
    qz = np.abs(pts["z"] - cz) - hz
    outside = np.sqrt(np.maximum(qx, 0) ** 2 +
                      np.maximum(qy, 0) ** 2 +
                      np.maximum(qz, 0) ** 2)
    inside = np.minimum(np.maximum(np.maximum(qx, qy), qz), 0.0)
    return outside + inside


def _box_curves(xmin, xmax, ymin, ymax, zmin, zmax):
    w = xmax - xmin
    h = ymax - ymin
    d = zmax - zmin

    def face(pos_fn, normal_val, area):
        return SympyCurve(
            parametric_pos=pos_fn,
            normal={
                "x": lambda p, v=normal_val[0]: np.full_like(p["h"], v),
                "y": lambda p, v=normal_val[1]: np.full_like(p["h"], v),
                "z": lambda p, v=normal_val[2]: np.full_like(p["h"], v),
            },
            area=area,
            bounds={"h": (0.0, 1.0), "r": (0.0, 1.0)},
            dims=3,
        )

    # -z face, normal (0, 0, -1)
    c1 = face(
        pos_fn={
            "x": lambda p: xmin + p["h"] * w,
            "y": lambda p: ymin + p["r"] * h,
            "z": lambda p: np.full_like(p["h"], zmin),
        },
        normal_val=(0, 0, -1),
        area=w * h,
    )
    # +z face, normal (0, 0, 1)
    c2 = face(
        pos_fn={
            "x": lambda p: xmin + p["h"] * w,
            "y": lambda p: ymin + p["r"] * h,
            "z": lambda p: np.full_like(p["h"], zmax),
        },
        normal_val=(0, 0, 1),
        area=w * h,
    )
    # -y face, normal (0, -1, 0)
    c3 = face(
        pos_fn={
            "x": lambda p: xmin + p["h"] * w,
            "y": lambda p: np.full_like(p["h"], ymin),
            "z": lambda p: zmin + p["r"] * d,
        },
        normal_val=(0, -1, 0),
        area=w * d,
    )
    # +y face, normal (0, 1, 0)
    c4 = face(
        pos_fn={
            "x": lambda p: xmin + p["h"] * w,
            "y": lambda p: np.full_like(p["h"], ymax),
            "z": lambda p: zmin + p["r"] * d,
        },
        normal_val=(0, 1, 0),
        area=w * d,
    )
    # -x face, normal (-1, 0, 0)
    c5 = face(
        pos_fn={
            "x": lambda p: np.full_like(p["h"], xmin),
            "y": lambda p: ymin + p["h"] * h,
            "z": lambda p: zmin + p["r"] * d,
        },
        normal_val=(-1, 0, 0),
        area=h * d,
    )
    # +x face, normal (1, 0, 0)
    c6 = face(
        pos_fn={
            "x": lambda p: np.full_like(p["h"], xmax),
            "y": lambda p: ymin + p["h"] * h,
            "z": lambda p: zmin + p["r"] * d,
        },
        normal_val=(1, 0, 0),
        area=h * d,
    )
    return [c1, c2, c3, c4, c5, c6]


# ── Sphere ───────────────────────────────────────────────────────────

class Sphere(Geometry):
    """3D sphere defined by center point and radius."""

    def __init__(self, center: tuple, radius: float,
                 parameterization: Parameterization = None):
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        self.center = (cx, cy, cz)
        self.radius = float(radius)

        curves = _sphere_curves(cx, cy, cz, radius)
        c = Curve(curves, dims=3)

        def sdf_func(pts, params=None):
            dx = pts["x"] - cx
            dy = pts["y"] - cy
            dz = pts["z"] - cz
            return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) - radius

        rp = radius + 0.5
        bounds = Bounds({
            "x": (cx - rp, cx + rp),
            "y": (cy - rp, cy + rp),
            "z": (cz - rp, cz + rp),
        })

        super().__init__(c, sdf_func, dims=3, bounds=bounds,
                         parameterization=parameterization)


def _sphere_curves(cx, cy, cz, r):
    """Sphere surface as a single parametric curve (u=theta, v=phi)."""
    area = 4.0 * np.pi * r * r

    def pos_x(p):
        theta = p["h"] * 2.0 * np.pi  # azimuth [0, 2π]
        phi = p["r"] * np.pi          # polar [0, π]
        return cx + r * np.sin(phi) * np.cos(theta)

    def pos_y(p):
        theta = p["h"] * 2.0 * np.pi
        phi = p["r"] * np.pi
        return cy + r * np.sin(phi) * np.sin(theta)

    def pos_z(p):
        phi = p["r"] * np.pi
        return cz + r * np.cos(phi)

    def n_x(p):
        theta = p["h"] * 2.0 * np.pi
        phi = p["r"] * np.pi
        return np.sin(phi) * np.cos(theta)

    def n_y(p):
        theta = p["h"] * 2.0 * np.pi
        phi = p["r"] * np.pi
        return np.sin(phi) * np.sin(theta)

    def n_z(p):
        phi = p["r"] * np.pi
        return np.cos(phi)

    return [SympyCurve(
        parametric_pos={"x": pos_x, "y": pos_y, "z": pos_z},
        normal={"x": n_x, "y": n_y, "z": n_z},
        area=area,
        bounds={"h": (0.0, 1.0), "r": (0.0, 1.0)},
        dims=3,
    )]


# ── Cylinder ─────────────────────────────────────────────────────────

class Cylinder(Geometry):
    """3D cylinder defined by center, radius, and height (along z-axis)."""

    def __init__(self, center: tuple, radius: float, height: float,
                 parameterization: Parameterization = None):
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        self.center = (cx, cy, cz)
        self.radius = float(radius)
        self.height = float(height)

        hh = height / 2.0
        curves = _cylinder_curves(cx, cy, cz - hh, cz + hh, radius)
        c = Curve(curves, dims=3)

        def sdf_func(pts, params=None):
            return _sdf_cylinder(pts, cx, cy, cz, radius, height)

        rp = radius + 0.5
        hp = hh + 0.5
        bounds = Bounds({
            "x": (cx - rp, cx + rp),
            "y": (cy - rp, cy + rp),
            "z": (cz - hp, cz + hp),
        })

        super().__init__(c, sdf_func, dims=3, bounds=bounds,
                         parameterization=parameterization)


def _sdf_cylinder(pts, cx, cy, cz, radius, height):
    hh = height / 2.0
    dx = pts["x"] - cx
    dy = pts["y"] - cy
    dz = pts["z"] - cz
    r_dist = np.sqrt(dx ** 2 + dy ** 2) - radius
    z_dist = np.abs(dz) - hh

    # Interior when both are negative
    outside_r = np.maximum(r_dist, 0.0)
    outside_z = np.maximum(z_dist, 0.0)
    inside_r = np.minimum(np.maximum(r_dist, z_dist), 0.0)

    # For points outside radially but within z range: use r_dist
    # For points outside z range but within radius: use z_dist
    # For points outside both: use sqrt(r_dist² + z_dist²)
    e = np.sqrt(outside_r ** 2 + outside_z ** 2)
    return np.where(np.maximum(r_dist, z_dist) > 0.0,
                    np.where(np.minimum(r_dist, z_dist) > 0.0, e,
                             np.where(r_dist > z_dist, r_dist, z_dist)),
                    inside_r)


def _cylinder_curves(cx, cy, zmin, zmax, r):
    area_side = 2.0 * np.pi * r * (zmax - zmin)
    area_cap = np.pi * r * r

    curves = []

    # Side surface
    def side_x(p):
        theta = p["h"] * 2.0 * np.pi
        return cx + r * np.cos(theta)

    def side_y(p):
        theta = p["h"] * 2.0 * np.pi
        return cy + r * np.sin(theta)

    def side_z(p):
        return zmin + p["r"] * (zmax - zmin)

    curves.append(SympyCurve(
        parametric_pos={"x": side_x, "y": side_y, "z": side_z},
        normal={
            "x": lambda p: np.cos(p["h"] * 2.0 * np.pi),
            "y": lambda p: np.sin(p["h"] * 2.0 * np.pi),
            "z": lambda p: np.zeros_like(p["h"]),
        },
        area=area_side,
        bounds={"h": (0.0, 1.0), "r": (0.0, 1.0)},
        dims=3,
    ))

    # Bottom cap (z = zmin), normal (0, 0, -1)
    def cap_bottom_x(p):
        rad = np.sqrt(p["h"]) * r
        theta = p["r"] * 2.0 * np.pi
        return cx + rad * np.cos(theta)

    def cap_bottom_y(p):
        rad = np.sqrt(p["h"]) * r
        theta = p["r"] * 2.0 * np.pi
        return cy + rad * np.sin(theta)

    curves.append(SympyCurve(
        parametric_pos={
            "x": cap_bottom_x,
            "y": cap_bottom_y,
            "z": lambda p: np.full_like(p["h"], zmin),
        },
        normal={
            "x": lambda p: np.zeros_like(p["h"]),
            "y": lambda p: np.zeros_like(p["h"]),
            "z": lambda p: -np.ones_like(p["h"]),
        },
        area=area_cap,
        bounds={"h": (0.0, 1.0), "r": (0.0, 1.0)},
        dims=3,
    ))

    # Top cap (z = zmax), normal (0, 0, 1)
    curves.append(SympyCurve(
        parametric_pos={
            "x": cap_bottom_x,
            "y": cap_bottom_y,
            "z": lambda p: np.full_like(p["h"], zmax),
        },
        normal={
            "x": lambda p: np.zeros_like(p["h"]),
            "y": lambda p: np.zeros_like(p["h"]),
            "z": lambda p: np.ones_like(p["h"]),
        },
        area=area_cap,
        bounds={"h": (0.0, 1.0), "r": (0.0, 1.0)},
        dims=3,
    ))

    return curves


# ── Plane ────────────────────────────────────────────────────────────

class Plane(Geometry):
    """Infinite 3D plane defined by two corner points (extruded to a finite slab)."""

    def __init__(self, point_1: tuple, point_2: tuple,
                 parameterization: Parameterization = None):
        x1, y1, z1 = float(point_1[0]), float(point_1[1]), float(point_1[2])
        x2, y2, z2 = float(point_2[0]), float(point_2[1]), float(point_2[2])

        # Plane normal from cross product with z-up when xy-plane, or use points
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1

        # Define normal as perpendicular to the vector between points
        length = np.sqrt(dx**2 + dy**2 + dz**2)
        if length > 0:
            ux, uy, uz = dx / length, dy / length, dz / length
        else:
            ux, uy, uz = 1.0, 0.0, 0.0

        # Find normal by crossing with (0,0,1) or (0,1,0)
        nx = -uy
        ny = ux
        nz = 0.0
        n_len = np.sqrt(nx**2 + ny**2 + nz**2)
        if n_len < 0.01:
            nx = -uz
            ny = 0.0
            nz = ux
            n_len = np.sqrt(nx**2 + ny**2 + nz**2)
        nx /= n_len
        ny /= n_len
        nz /= n_len

        self.normal = (nx, ny, nz)
        self.point_1 = (x1, y1, z1)
        self.point_2 = (x2, y2, z2)

        # Plane as finite rectangle
        curves = _plane_curves(x1, y1, z1, x2, y2, z2, nx, ny, nz)
        c = Curve(curves, dims=3)

        def sdf_func(pts, params=None):
            return (pts["x"] - x1) * nx + (pts["y"] - y1) * ny + (pts["z"] - z1) * nz

        margin = 5.0
        bounds = Bounds({
            "x": (min(x1, x2) - margin, max(x1, x2) + margin),
            "y": (min(y1, y2) - margin, max(y1, y2) + margin),
            "z": (min(z1, z2) - margin, max(z1, z2) + margin),
        })

        super().__init__(c, sdf_func, dims=3, bounds=bounds,
                         parameterization=parameterization)


def _plane_curves(x1, y1, z1, x2, y2, z2, nx, ny, nz):
    """Simple rectangular plane face."""
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    length = np.sqrt(dx**2 + dy**2 + dz**2)
    area = length * 2.0  # approximate width

    return [SympyCurve(
        parametric_pos={
            "x": lambda p: x1 + p["h"] * dx,
            "y": lambda p: y1 + p["h"] * dy,
            "z": lambda p: z1 + p["h"] * dz,
        },
        normal={
            "x": lambda p: np.full_like(p["h"], nx),
            "y": lambda p: np.full_like(p["h"], ny),
            "z": lambda p: np.full_like(p["h"], nz),
        },
        area=area,
        bounds={"h": (0.0, 1.0), "r": (0.0, 1.0)},
        dims=3,
    )]
