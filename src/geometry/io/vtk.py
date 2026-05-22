"""VTK and STL I/O for geometry samples.

Mirrors physicsnemo.sym.utils.io.vtk.var_to_polyvtk and
physicsnemo.utils.mesh.sdf_to_stl.
"""

import os
import numpy as np


def var_to_polyvtk(sample_dict: dict, filename_prefix: str):
    """Write sampled point data to VTK PolyData XML (.vtp) files.

    Creates two files:
    - {filename_prefix}_boundary.vtp — surface points (if normals present)
    - {filename_prefix}_interior.vtp — interior/volume points

    Args:
        sample_dict: dict from sample_boundary() or sample_interior() with
                     'x','y','z' coords and optional 'normal_x', etc.
        filename_prefix: output path prefix
    """
    dim_names = ["x", "y", "z"]
    present_dims = [d for d in dim_names if d in sample_dict]
    if not present_dims:
        raise ValueError("sample_dict must contain at least 'x','y','z' coordinates")

    n_points = len(sample_dict[present_dims[0]])
    if n_points == 0:
        return

    # Determine if this is boundary data (has normals)
    has_normals = all(f"normal_{d}" in sample_dict for d in present_dims)
    suffix = "_boundary" if has_normals else "_interior"

    filepath = f"{filename_prefix}{suffix}.vtp"

    with open(filepath, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">\n')
        f.write('  <PolyData>\n')
        f.write(f'    <Piece NumberOfPoints="{n_points}" NumberOfVerts="0" '
                f'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n')

        # Points
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float64" NumberOfComponents="3" '
                'format="ascii">\n')
        pts = np.column_stack([sample_dict[d].ravel() for d in present_dims])
        for row in pts:
            f.write(f"          {row[0]:.12f} {row[1]:.12f}")
            if len(row) > 2:
                f.write(f" {row[2]:.12f}")
            f.write("\n")
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')

        # Point data arrays
        f.write('      <PointData>\n')

        if has_normals:
            f.write('        <DataArray type="Float64" Name="Normals" '
                    'NumberOfComponents="3" format="ascii">\n')
            normal_pts = np.column_stack(
                [sample_dict[f"normal_{d}"].ravel() for d in present_dims]
            )
            for row in normal_pts:
                f.write(f"          {row[0]:.12f} {row[1]:.12f}")
                if len(row) > 2:
                    f.write(f" {row[2]:.12f}")
                f.write("\n")
            f.write('        </DataArray>\n')

        if "area" in sample_dict:
            f.write('        <DataArray type="Float64" Name="Area" '
                    'NumberOfComponents="1" format="ascii">\n')
            for val in sample_dict["area"].ravel():
                f.write(f"          {val:.12f}\n")
            f.write('        </DataArray>\n')

        if "sdf" in sample_dict:
            f.write('        <DataArray type="Float64" Name="SDF" '
                    'NumberOfComponents="1" format="ascii">\n')
            for val in sample_dict["sdf"].ravel():
                f.write(f"          {val:.12f}\n")
            f.write('        </DataArray>\n')

        # SDF derivatives
        for d in present_dims:
            key = f"sdf__{d}"
            if key in sample_dict:
                f.write(f'        <DataArray type="Float64" Name="SDF_grad_{d}" '
                        'NumberOfComponents="1" format="ascii">\n')
                for val in sample_dict[key].ravel():
                    f.write(f"          {val:.12f}\n")
                f.write('        </DataArray>\n')

        f.write('      </PointData>\n')
        f.write('    </Piece>\n')
        f.write('  </PolyData>\n')
        f.write('</VTKFile>\n')


def sdf_to_stl(sdf_field: np.ndarray, threshold: float = 0.0,
               backend: str = "skimage", filename: str = "output.stl",
               voxel_size: tuple = (1.0, 1.0, 1.0)):
    """Reconstruct an STL mesh from an SDF volume using marching cubes.

    Args:
        sdf_field: 3D numpy array [nx, ny, nz] of SDF values
        threshold: isosurface level (default 0.0 for the boundary)
        backend: 'skimage' (better for CSG) or 'warp' (faster for STL)
        filename: output .stl file path
        voxel_size: (dx, dy, dz) voxel dimensions
    """
    nx, ny, nz = sdf_field.shape

    if backend == "warp":
        try:
            import warp as wp
            # Warp-based marching cubes would go here
            # Fall back to skimage for now
            _sdf_to_stl_skimage(sdf_field, threshold, filename, voxel_size)
        except ImportError:
            _sdf_to_stl_skimage(sdf_field, threshold, filename, voxel_size)
    else:
        _sdf_to_stl_skimage(sdf_field, threshold, filename, voxel_size)


def _sdf_to_stl_skimage(sdf_field, threshold, filename, voxel_size):
    """Use skimage marching cubes to extract isosurface and save as STL."""
    try:
        from skimage import measure
    except ImportError:
        raise ImportError(
            "scikit-image is required for sdf_to_stl. "
            "Install with: pip install scikit-image"
        )

    verts, faces, normals, values = measure.marching_cubes(
        sdf_field, level=threshold, spacing=voxel_size
    )

    # Write binary STL
    n_faces = len(faces)
    with open(filename, "wb") as f:
        # 80-byte header
        f.write(b"Binary STL from sdf_to_stl" + b"\x00" * (80 - 30))
        # Number of triangles (4-byte uint32)
        f.write(np.uint32(n_faces).tobytes())

        for i, face in enumerate(faces):
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            # Normal
            n = normals[i] if i < len(normals) else np.cross(v1 - v0, v2 - v0)
            n_len = np.linalg.norm(n)
            if n_len > 0:
                n = n / n_len
            f.write(n.astype(np.float32).tobytes())
            # Vertices
            for v in [v0, v1, v2]:
                f.write(v.astype(np.float32).tobytes())
            # Attribute byte count (2 bytes, zero)
            f.write(np.uint16(0).tobytes())
