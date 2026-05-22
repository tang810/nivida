"""Demonstration of the CSG and tessellated geometry module.

Showcases all major features matching the NVIDIA PhysicsNeMo API:
- 2D primitives with boolean operations
- 3D CSG: Box ∩ Sphere - cylinders (the "Swiss cheese cube")
- Transforms: translate, scale, rotate, repeat
- Parameterization
- Boundary and interior sampling
- SDF computation
- VTK export for ParaView visualization
- STL mesh reconstruction via marching cubes
"""

import os
import numpy as np

from src.geometry.primitives_3d import Box, Sphere, Cylinder
from src.geometry.primitives_2d import Rectangle, Circle
from src.geometry.parameterization import Parameterization, Parameter
from src.geometry.io.vtk import var_to_polyvtk, sdf_to_stl


def demo_2d_csg():
    """2D CSG: rectangle with circular hole."""
    print("=" * 50)
    print("2D CSG Demo: Rectangle with circular hole")
    print("=" * 50)

    plate = Rectangle(point_1=(-2, -2), point_2=(2, 2))
    hole = Circle(center=(0, 0), radius=1.0)
    geo = plate - hole

    s = geo.sample_boundary(nr_points=5000)
    print(f"  Boundary points: {len(s['x'])}")
    var_to_polyvtk(s, "output/demo_2d")


def demo_3d_csg():
    """3D CSG: Box intersected with sphere, minus three cylinders.

    This is the classic example from PhysicsNeMo docs.
    """
    print("\n" + "=" * 50)
    print("3D CSG Demo: Box & Sphere - Cylinders")
    print("=" * 50)

    nr_points = 50000

    box = Box(point_1=(-1, -1, -1), point_2=(1, 1, 1))
    sphere = Sphere(center=(0, 0, 0), radius=1.2)
    cylinder_1 = Cylinder(center=(0, 0, 0), radius=0.5, height=2)
    cylinder_2 = cylinder_1.rotate(angle=float(np.pi / 2.0), axis="x")
    cylinder_3 = cylinder_1.rotate(angle=float(np.pi / 2.0), axis="y")

    all_cylinders = cylinder_1 + cylinder_2 + cylinder_3
    box_minus_sphere = box & sphere
    geo = box_minus_sphere - all_cylinders

    s = geo.sample_boundary(nr_points=nr_points)
    print(f"  Boundary points: {len(s['x'])}")
    var_to_polyvtk(s, "output/demo_3d_boundary")

    s = geo.sample_interior(nr_points=nr_points, compute_sdf_derivatives=True)
    print(f"  Interior points: {len(s['x'])}")
    print(f"  SDF keys: {[k for k in s.keys() if k.startswith('sdf')]}")
    var_to_polyvtk(s, "output/demo_3d_interior")


def demo_transforms():
    """Demonstrate geometry transforms: scale, rotate, repeat."""
    print("\n" + "=" * 50)
    print("Transform Demo: Scaled, rotated, repeated box")
    print("=" * 50)

    box = Box(point_1=(-0.5, -0.5, -0.5), point_2=(0.5, 0.5, 0.5))
    geo = box.scale(2.0)
    geo = geo.rotate(angle=np.pi / 4, axis="z")
    geo = geo.translate((1.0, 0.0, 0.0))

    # Repeat in a 3x1x1 grid
    geo = geo.repeat(spacing=4.0, repeat_lower=(-1, 0, 0), repeat_higher=(1, 0, 0))

    s = geo.sample_boundary(nr_points=30000)
    print(f"  Boundary points: {len(s['x'])}")
    var_to_polyvtk(s, "output/demo_transform")


def demo_parameterization():
    """Demonstrate parameterized geometry.

    A plate with a hole whose position is a parameter.
    """
    print("\n" + "=" * 50)
    print("Parameterization Demo: Plate with parameterized hole")
    print("=" * 50)

    plate = Rectangle(point_1=(-1, -1), point_2=(1, 1))
    y_pos = Parameter("y_pos")
    param = Parameterization({y_pos: (-0.5, 0.5)})
    hole = Circle(center=(0, y_pos), radius=0.3, parameterization=param)
    geo = plate - hole

    s = geo.sample_boundary(nr_points=10000)
    print(f"  Boundary points (full param range): {len(s['x'])}")
    var_to_polyvtk(s, "output/demo_param")


def demo_sdf_marching_cubes():
    """Reconstruct STL mesh from SDF using marching cubes."""
    print("\n" + "=" * 50)
    print("Marching Cubes Demo: SDF → STL reconstruction")
    print("=" * 50)

    box = Box(point_1=(-1, -1, -1), point_2=(1, 1, 1))
    sphere = Sphere(center=(0, 0, 0), radius=1.2)

    # Evaluate SDF on a grid
    nx = ny = nz = 80
    x = np.linspace(-1.5, 1.5, nx)
    y = np.linspace(-1.5, 1.5, ny)
    z = np.linspace(-1.5, 1.5, nz)
    XX, YY, ZZ = np.meshgrid(x, y, z, indexing="ij")

    sdf_vals = box.sdf({
        "x": XX.reshape(-1, 1),
        "y": YY.reshape(-1, 1),
        "z": ZZ.reshape(-1, 1),
    })["sdf"].reshape(nx, ny, nz)

    sdf_to_stl(sdf_vals, threshold=0.0, backend="skimage",
               filename="output/demo_box.stl")
    print("  Saved: output/demo_box.stl")


def main():
    os.makedirs("output", exist_ok=True)

    demo_2d_csg()
    demo_3d_csg()
    demo_transforms()
    demo_parameterization()
    demo_sdf_marching_cubes()

    print("\n" + "=" * 50)
    print("All demos complete. VTK files in output/ can be viewed in ParaView.")
    print("STL files in output/ can be viewed in any 3D viewer.")
    print("=" * 50)


if __name__ == "__main__":
    main()
