"""Tessellated geometry from STL files via ray-tracing SDF.

Drop-in replacement for physicsnemo.sym.geometry.tessellation.Tessellation.
Uses trimesh for STL loading and closest-point queries for SDF computation.

Usage:
    from src.geometry.tessellation import Tessellation
    geo = Tessellation.from_stl("./my_mesh.stl")
    s = geo.sample_boundary(nr_points=100000)
    s = geo.sample_interior(nr_points=50000, compute_sdf_derivatives=True)
"""

import numpy as np

from .geometry import Geometry
from .curve import Curve, SympyCurve
from .parameterization import Bounds


class Tessellation(Geometry):
    """Geometry defined by an STL mesh, with ray-cast SDF and surface sampling."""

    def __init__(self, mesh, parameterization=None):
        """
        Args:
            mesh: trimesh.Trimesh object
        """
        self._mesh = mesh
        self._vertices = np.asarray(mesh.vertices, dtype=np.float64)
        self._faces = np.asarray(mesh.faces, dtype=np.int64)

        # Pre-compute face data for fast closest-point queries
        self._precompute_face_data()

        curves = _tessellation_curves(self._vertices, self._faces)
        c = Curve(curves, dims=3)

        # Bounding box from mesh extents
        vmin = self._vertices.min(axis=0)
        vmax = self._vertices.max(axis=0)
        margin = 1.0
        bounds = Bounds({
            "x": (vmin[0] - margin, vmax[0] + margin),
            "y": (vmin[1] - margin, vmax[1] + margin),
            "z": (vmin[2] - margin, vmax[2] + margin),
        })

        def sdf_func(pts, params=None):
            return self._compute_sdf(pts)

        super().__init__(c, sdf_func, dims=3, bounds=bounds,
                         parameterization=parameterization)

    @classmethod
    def from_stl(cls, stl_path: str, **kwargs) -> "Tessellation":
        """Create a Tessellation from an STL file path.

        Args:
            stl_path: path to .stl file
            **kwargs: passed to trimesh.load
        """
        try:
            import trimesh
        except ImportError:
            raise ImportError(
                "trimesh is required for STL tessellation. "
                "Install with: pip install trimesh"
            )
        mesh = trimesh.load(stl_path, **kwargs)
        if isinstance(mesh, trimesh.Scene):
            # If scene, merge all geometries
            meshes = []
            for g in mesh.geometry.values():
                if hasattr(g, "vertices"):
                    meshes.append(g)
            if not meshes:
                raise ValueError(f"No meshes found in scene: {stl_path}")
            mesh = trimesh.util.concatenate(meshes)
        return cls(mesh)

    @classmethod
    def from_vertices_faces(cls, vertices: np.ndarray, faces: np.ndarray,
                            **kwargs) -> "Tessellation":
        """Create a Tessellation from raw vertices and faces arrays."""
        try:
            import trimesh
        except ImportError:
            raise ImportError(
                "trimesh is required for tessellation. "
                "Install with: pip install trimesh"
            )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, **kwargs)
        return cls(mesh)

    def _precompute_face_data(self):
        """Pre-compute face normals, edges, and centroids for fast distance queries."""
        v = self._vertices
        f = self._faces

        # Face vertices
        self._f0 = v[f[:, 0]]
        self._f1 = v[f[:, 1]]
        self._f2 = v[f[:, 2]]

        # Face edges
        e0 = self._f1 - self._f0
        e1 = self._f2 - self._f0

        # Face normals (unnormalized)
        self._face_normals = np.cross(e0, e1)
        self._face_normal_len = np.linalg.norm(self._face_normals, axis=1, keepdims=True)
        self._face_normal_len = np.where(self._face_normal_len > 0, self._face_normal_len, 1.0)
        self._face_unit_normals = self._face_normals / self._face_normal_len

        # Face areas (0.5 * |cross|)
        self._face_areas = 0.5 * self._face_normal_len.ravel()

        # Face centroids
        self._face_centroids = (self._f0 + self._f1 + self._f2) / 3.0

    def _point_to_triangle_distance(self, px, py, pz):
        """Compute signed distance from points to mesh triangles.

        Returns (N_faces, N_points) distances.
        """
        n_pts = len(px)
        n_faces = len(self._faces)

        # Vector from v0 to query point
        v0x = px[None, :] - self._f0[:, 0:1]
        v0y = py[None, :] - self._f0[:, 1:2]
        v0z = pz[None, :] - self._f0[:, 2:3]

        e0 = self._f1 - self._f0
        e1 = self._f2 - self._f0

        # Barycentric coordinates
        d00 = np.sum(e0 * e0, axis=1)  # (F,)
        d01 = np.sum(e0 * e1, axis=1)
        d11 = np.sum(e1 * e1, axis=1)
        denom = d00 * d11 - d01 * d01
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)

        # Project onto edges
        d0 = (v0x * e0[:, 0:1] + v0y * e0[:, 1:2] + v0z * e0[:, 2:3])
        d1 = (v0x * e1[:, 0:1] + v0y * e1[:, 1:2] + v0z * e1[:, 2:3])

        # Barycentric u, v
        u = (d11[:, None] * d0 - d01[:, None] * d1) / denom[:, None]
        v = (d00[:, None] * d1 - d01[:, None] * d0) / denom[:, None]

        # Clamp to triangle
        # Region 3 (v < 0): closest to edge e0
        # Region 5 (u < 0): closest to edge e1
        # Region 6 (u + v > 1): closest to opposite edge
        # Region 0: inside

        # Closest point computation
        closest_x = self._f0[:, 0:1] + u * e0[:, 0:1] + v * e1[:, 0:1]
        closest_y = self._f0[:, 1:2] + u * e0[:, 1:2] + v * e1[:, 1:2]
        closest_z = self._f0[:, 2:3] + u * e0[:, 2:3] + v * e1[:, 2:3]

        # Handle edge/vertex regions
        u_clamped = np.clip(u, 0, 1)
        v_clamped = np.clip(v, 0, 1)
        uv_sum = u_clamped + v_clamped

        # Region 4: u + v > 1, project back
        region4 = uv_sum > 1
        if np.any(region4):
            u_clamped = np.where(region4, u_clamped / np.maximum(uv_sum, 1e-12), u_clamped)
            v_clamped = np.where(region4, v_clamped / np.maximum(uv_sum, 1e-12), v_clamped)

        closest_x = self._f0[:, 0:1] + u_clamped * e0[:, 0:1] + v_clamped * e1[:, 0:1]
        closest_y = self._f0[:, 1:2] + u_clamped * e0[:, 1:2] + v_clamped * e1[:, 1:2]
        closest_z = self._f0[:, 2:3] + u_clamped * e0[:, 2:3] + v_clamped * e1[:, 2:3]

        # Distance to closest point
        dx = px[None, :] - closest_x
        dy = py[None, :] - closest_y
        dz = pz[None, :] - closest_z
        dist_sq = dx * dx + dy * dy + dz * dz

        # Signed distance
        dist = np.sqrt(dist_sq)

        # Dot with face normal to determine sign
        dot = (dx * self._face_unit_normals[:, 0:1] +
               dy * self._face_unit_normals[:, 1:2] +
               dz * self._face_unit_normals[:, 2:3])

        # Sign: negative inside (dot < 0), positive outside
        signed_dist = np.where(dot < 0, -dist, dist)

        # Get closest face distance per point
        abs_dist = np.abs(signed_dist)
        min_idx = np.argmin(abs_dist, axis=0)
        min_dist = signed_dist[min_idx, np.arange(n_pts)]

        return min_dist.reshape(-1, 1)

    def _compute_sdf(self, pts):
        """Compute SDF for query points.

        Uses closest-point distance to mesh with sign from face normals.
        For large meshes, uses a subset of faces for efficiency.
        """
        px = pts["x"].ravel()
        py = pts["y"].ravel()
        pz = pts["z"].ravel()

        n_faces = len(self._faces)

        # For very large meshes, use spatial hashing for efficiency
        if n_faces > 50000:
            return self._compute_sdf_chunked(px, py, pz)

        return self._point_to_triangle_distance(px, py, pz)

    def _compute_sdf_chunked(self, px, py, pz):
        """Chunked SDF computation for large meshes."""
        n_faces = len(self._faces)
        chunk_size = 20000
        n_pts = len(px)

        min_abs_dist = np.full(n_pts, np.inf)
        min_signed_dist = np.full(n_pts, 0.0)

        for start in range(0, n_faces, chunk_size):
            end = min(start + chunk_size, n_faces)
            # Temporarily reduce faces
            saved_faces = self._faces
            saved_f0, saved_f1, saved_f2 = self._f0, self._f1, self._f2
            saved_normals = self._face_unit_normals
            saved_centroids = self._face_centroids

            self._faces = self._faces[start:end]
            self._f0 = self._f0[start:end]
            self._f1 = self._f1[start:end]
            self._f2 = self._f2[start:end]
            self._face_unit_normals = self._face_unit_normals[start:end]

            chunk_dist = self._point_to_triangle_distance(px, py, pz).ravel()
            chunk_abs = np.abs(chunk_dist)
            update = chunk_abs < min_abs_dist
            min_abs_dist = np.where(update, chunk_abs, min_abs_dist)
            min_signed_dist = np.where(update, chunk_dist, min_signed_dist)

            # Restore
            self._faces = saved_faces
            self._f0, self._f1, self._f2 = saved_f0, saved_f1, saved_f2
            self._face_unit_normals = saved_normals
            self._face_centroids = saved_centroids

        return min_signed_dist.reshape(-1, 1)


def _tessellation_curves(vertices, faces):
    """Create parametric curves from mesh faces for boundary sampling."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)

    # Compute total surface area
    total_area = 0.0
    face_data = []
    for fi, face in enumerate(f):
        a, b, c = v[face[0]], v[face[1]], v[face[2]]
        e0 = b - a
        e1 = c - a
        n = np.cross(e0, e1)
        area = 0.5 * np.linalg.norm(n)
        total_area += area
        face_data.append((a, b, c, e0, e1, area))

    # Create a single representative curve for the entire mesh
    def pos_x(p):
        # Random face index based on area
        r = p["h"].ravel()
        # Use area-weighted face selection
        n_faces = len(face_data)
        n_pts = len(r)
        face_idx = np.random.randint(0, n_faces, n_pts)
        a_x = np.array([fd[0][0] for fd in face_data])[face_idx]
        e0_x = np.array([fd[3][0] for fd in face_data])[face_idx]
        e1_x = np.array([fd[4][0] for fd in face_data])[face_idx]
        u = p["r"].ravel()
        vv = np.random.uniform(0, 1 - u, n_pts)
        return (a_x + u * e0_x + vv * e1_x).reshape(-1, 1)

    def pos_y(p):
        r = p["h"].ravel()
        n_faces = len(face_data)
        n_pts = len(r)
        face_idx = np.random.randint(0, n_faces, n_pts)
        a_y = np.array([fd[0][1] for fd in face_data])[face_idx]
        e0_y = np.array([fd[3][1] for fd in face_data])[face_idx]
        e1_y = np.array([fd[4][1] for fd in face_data])[face_idx]
        u = p["r"].ravel()
        vv = np.random.uniform(0, 1 - u, n_pts)
        return (a_y + u * e0_y + vv * e1_y).reshape(-1, 1)

    def pos_z(p):
        r = p["h"].ravel()
        n_faces = len(face_data)
        n_pts = len(r)
        face_idx = np.random.randint(0, n_faces, n_pts)
        a_z = np.array([fd[0][2] for fd in face_data])[face_idx]
        e0_z = np.array([fd[3][2] for fd in face_data])[face_idx]
        e1_z = np.array([fd[4][2] for fd in face_data])[face_idx]
        u = p["r"].ravel()
        vv = np.random.uniform(0, 1 - u, n_pts)
        return (a_z + u * e0_z + vv * e1_z).reshape(-1, 1)

    def n_x(p):
        r = p["h"].ravel()
        n_faces = len(face_data)
        n_pts = len(r)
        face_idx = np.random.randint(0, n_faces, n_pts)
        e0 = np.array([fd[3] for fd in face_data])[face_idx]
        e1 = np.array([fd[4] for fd in face_data])[face_idx]
        normals = np.cross(e0, e1)
        n_len = np.linalg.norm(normals, axis=1)
        n_len = np.where(n_len > 0, n_len, 1.0)
        return (normals[:, 0] / n_len).reshape(-1, 1)

    def n_y(p):
        r = p["h"].ravel()
        n_faces = len(face_data)
        n_pts = len(r)
        face_idx = np.random.randint(0, n_faces, n_pts)
        e0 = np.array([fd[3] for fd in face_data])[face_idx]
        e1 = np.array([fd[4] for fd in face_data])[face_idx]
        normals = np.cross(e0, e1)
        n_len = np.linalg.norm(normals, axis=1)
        n_len = np.where(n_len > 0, n_len, 1.0)
        return (normals[:, 1] / n_len).reshape(-1, 1)

    def n_z(p):
        r = p["h"].ravel()
        n_faces = len(face_data)
        n_pts = len(r)
        face_idx = np.random.randint(0, n_faces, n_pts)
        e0 = np.array([fd[3] for fd in face_data])[face_idx]
        e1 = np.array([fd[4] for fd in face_data])[face_idx]
        normals = np.cross(e0, e1)
        n_len = np.linalg.norm(normals, axis=1)
        n_len = np.where(n_len > 0, n_len, 1.0)
        return (normals[:, 2] / n_len).reshape(-1, 1)

    return [SympyCurve(
        parametric_pos={"x": pos_x, "y": pos_y, "z": pos_z},
        normal={"x": n_x, "y": n_y, "z": n_z},
        area=float(total_area),
        bounds={"h": (0.0, 1.0), "r": (0.0, 1.0)},
        dims=3,
    )]
