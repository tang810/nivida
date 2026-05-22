"""CSG and Tessellated Geometry Module.

NVIDIA PhysicsNeMo-compatible constructive solid geometry with SDF computation,
point cloud sampling, boolean operations, and STL tessellation support.

Main components:
- Geometry: base class with CSG ops (+, &, -), transforms, SDF, sampling
- Primitives 2D: Rectangle, Circle, Line, Polygon
- Primitives 3D: Box, Sphere, Cylinder, Plane
- Tessellation: STL import with ray-tracing SDF
- I/O: VTK export for ParaView, SDF-to-STL marching cubes

Usage:
    from src.geometry import Box, Sphere, Cylinder
    from src.geometry.io import var_to_polyvtk

    box = Box(point_1=(-1, -1, -1), point_2=(1, 1, 1))
    sphere = Sphere(center=(0, 0, 0), radius=1.2)
    geo = box & sphere  # intersection

    s = geo.sample_boundary(nr_points=100000)
    var_to_polyvtk(s, "output")
    s = geo.sample_interior(nr_points=50000, compute_sdf_derivatives=True)
"""

from .geometry import Geometry
from .curve import Curve, SympyCurve
from .helper import csg_curve_naming, _sympy_sdf_to_sdf
from .parameterization import Parameter, Parameterization, Bounds
from .primitives_2d import Rectangle, Circle, Line, Polygon
from .primitives_3d import Box, Sphere, Cylinder, Plane
from .tessellation import Tessellation

__all__ = [
    # Base
    "Geometry",
    # Curves & helpers
    "Curve",
    "SympyCurve",
    "csg_curve_naming",
    "_sympy_sdf_to_sdf",
    # Parameterization
    "Parameter",
    "Parameterization",
    "Bounds",
    # 2D primitives
    "Rectangle",
    "Circle",
    "Line",
    "Polygon",
    # 3D primitives
    "Box",
    "Sphere",
    "Cylinder",
    "Plane",
    # Tessellation
    "Tessellation",
]
