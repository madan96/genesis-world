"""Continuous skin geometry and reference-frame particle bindings."""

from pathlib import Path

import igl
import numpy as np
import trimesh
from skimage.measure import marching_cubes

import genesis as gs
import genesis.utils.geom as gu


def _load_mapping(source, what: str) -> dict:
    """Load a dict-like payload from an ``.npz`` path or pass through an in-memory mapping."""
    if source is None:
        return {}
    if isinstance(source, (str, Path)):
        with np.load(source, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    if isinstance(source, dict):
        return dict(source)
    gs.raise_exception(f"Hybrid skin {what} must be an .npz path or a dict, got {type(source)}.")


def _as_str_list(value) -> list[str]:
    return [str(v) for v in np.asarray(value).reshape(-1)]


def load_surface(surface, scale: float = 1.0) -> trimesh.Trimesh:
    """Load and sanity-check the continuous reference surface (entity frame)."""
    if isinstance(surface, (str, Path)):
        loaded = trimesh.load(surface, force="mesh")
        if not isinstance(loaded, trimesh.Trimesh):
            gs.raise_exception(f"Hybrid skin surface file '{surface}' did not load as a single mesh.")
        surface = loaded
    if not isinstance(surface, trimesh.Trimesh):
        gs.raise_exception(f"Hybrid skin surface must be a mesh path or trimesh.Trimesh, got {type(surface)}.")
    surface = surface.copy()
    surface.apply_scale(scale)

    verts = np.asarray(surface.vertices)
    faces = np.asarray(surface.faces)
    if verts.size == 0 or faces.size == 0:
        gs.raise_exception("Hybrid skin surface has no vertices or faces.")
    if not np.isfinite(verts).all():
        bad = np.flatnonzero(~np.isfinite(verts).all(axis=1))
        gs.raise_exception(f"Hybrid skin surface has non-finite vertices at indices {bad[:8].tolist()}.")
    if faces.min() < 0 or faces.max() >= len(verts):
        gs.raise_exception("Hybrid skin surface has out-of-range face indices.")
    return surface


def validate_solid_surface(surface: trimesh.Trimesh):
    """Require a finite, watertight, outward-oriented triangle surface with positive volume."""
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces)
    if not np.isfinite(vertices).all() or vertices.ndim != 2 or vertices.shape[1] != 3:
        gs.raise_exception("Hybrid skin vertices must be a finite [V, 3] array.")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0:
        gs.raise_exception("Hybrid skin faces must be a nonempty [F, 3] triangle array.")
    if faces.min() < 0 or faces.max() >= len(vertices):
        gs.raise_exception("Hybrid skin face indices are outside the vertex array.")
    triangles = vertices[faces]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    scale = np.linalg.norm(np.ptp(vertices, axis=0))
    if np.any(np.linalg.norm(np.cross(edge_a, edge_b), axis=1) <= np.finfo(np.float64).eps * scale**2):
        gs.raise_exception("Hybrid solid skin contains degenerate triangles.")
    if not surface.is_watertight or not surface.is_winding_consistent:
        gs.raise_exception("Hybrid solid skin must be watertight with consistent triangle winding.")
    if surface.volume <= np.finfo(np.float64).eps * scale**3:
        gs.raise_exception("Hybrid solid skin must have positive volume and outward triangle winding.")


def load_skinning(skinning, surface: trimesh.Trimesh, link_name_to_idx: dict[str, int]):
    """
    Load per-vertex link influences and map link names to global link indices.

    Expected payload: ``link_names`` [L], ``indices`` int [V, K] (into link_names, -1 unused),
    ``weights`` float [V, K].
    """
    payload = _load_mapping(skinning, "skinning")
    if not payload:
        return None, None
    for key in ("link_names", "indices", "weights"):
        if key not in payload:
            gs.raise_exception(f"Hybrid skinning payload is missing '{key}' (has {sorted(payload)}).")

    link_names = _as_str_list(payload["link_names"])
    indices = np.asarray(payload["indices"])
    if indices.dtype.kind not in "iu":
        gs.raise_exception("Hybrid skinning indices must be integers.")
    indices = indices.astype(np.int64, copy=False)
    weights = np.asarray(payload["weights"], dtype=np.float64)

    n_verts = len(surface.vertices)
    if indices.ndim != 2 or weights.shape != indices.shape:
        gs.raise_exception(
            f"Hybrid skinning 'indices'/'weights' must both be [V, K]; got {indices.shape} and {weights.shape}."
        )
    if indices.shape[0] != n_verts:
        gs.raise_exception(f"Hybrid skinning arrays are for {indices.shape[0]} vertices but the surface has {n_verts}.")

    if not link_names or len(set(link_names)) != len(link_names):
        gs.raise_exception("Hybrid skinning link_names must be nonempty and unique.")
    if indices.shape[1] == 0 or np.any(indices < -1) or np.any(indices >= len(link_names)):
        gs.raise_exception("Hybrid skinning indices must be -1 or an index into link_names.")
    missing = sorted(set(link_names) - set(link_name_to_idx))
    if missing:
        gs.raise_exception(
            f"Hybrid skinning references links that do not exist in the rigid articulation: {missing}. "
            f"Available links: {sorted(link_name_to_idx)}."
        )
    if not np.isfinite(weights).all():
        gs.raise_exception("Hybrid skinning weights contain non-finite values.")
    if (weights < 0.0).any():
        gs.raise_exception("Hybrid skinning weights must be nonnegative.")
    used = indices >= 0
    if np.any(weights[~used] > 0.0):
        gs.raise_exception("Unused Hybrid skinning slots must have zero weight.")
    row_max = weights.max(axis=1)
    dead = np.flatnonzero(row_max <= 0.0)
    if dead.size:
        gs.raise_exception(
            f"{dead.size} surface vertices have no positive skinning weight (first: {dead[:8].tolist()}); "
            "every vertex must be normalizable."
        )

    # Scale each row before summation to preserve normalization for tiny and near-overflow weights
    weights = weights / row_max[:, None]
    name_idx_to_link = np.array([link_name_to_idx[name] for name in link_names], dtype=np.int64)
    vert_link_idx = np.where(used, name_idx_to_link[np.clip(indices, 0, None)], -1)
    vert_weights = weights / weights.sum(axis=1, keepdims=True)
    return vert_link_idx, vert_weights


def load_regions(regions, surface: trimesh.Trimesh):
    """Load semantic region IDs per surface vertex."""
    payload = _load_mapping(regions, "regions")
    if not payload:
        return [], None
    for key in ("region_names", "vertex_region"):
        if key not in payload:
            gs.raise_exception(f"Hybrid regions payload is missing '{key}' (has {sorted(payload)}).")
    region_names = _as_str_list(payload["region_names"])
    vertex_region = np.asarray(payload["vertex_region"])
    if vertex_region.dtype.kind not in "iu" or vertex_region.ndim != 1:
        gs.raise_exception("Hybrid vertex_region must be a one-dimensional integer array.")
    if not region_names or len(set(region_names)) != len(region_names):
        gs.raise_exception("Hybrid region_names must be nonempty and unique.")
    vertex_region = vertex_region.astype(np.int64, copy=False)
    if len(vertex_region) != len(surface.vertices):
        gs.raise_exception(
            f"Hybrid regions cover {len(vertex_region)} vertices but the surface has {len(surface.vertices)}."
        )
    if vertex_region.min() < 0 or vertex_region.max() >= len(region_names):
        gs.raise_exception("Hybrid 'vertex_region' contains indices outside region_names.")
    return region_names, vertex_region


def load_vertex_attributes(surface, material_id, surface_contact, n_materials):
    """Validate one material index and contact flag per surface vertex, using homogeneous defaults when omitted."""
    n_vertices = len(surface.vertices)
    if material_id is None:
        material_id = np.zeros(n_vertices, dtype=gs.np_int)
    else:
        material_id = np.asarray(material_id)
        if material_id.shape != (n_vertices,) or material_id.dtype.kind not in "iu":
            gs.raise_exception("Hybrid vertex_material_id must be an integer array with one entry per surface vertex.")
        if np.any(material_id < 0) or np.any(material_id >= n_materials):
            gs.raise_exception("Hybrid vertex_material_id contains an index outside the soft material palette.")
    if surface_contact is None:
        surface_contact = np.ones(n_vertices, dtype=bool)
    else:
        surface_contact = np.asarray(surface_contact)
        if surface_contact.shape != (n_vertices,) or surface_contact.dtype.kind != "b":
            gs.raise_exception(
                "Hybrid vertex_surface_contact must be a boolean array with one entry per surface vertex."
            )
    return material_id, surface_contact


def link_collision_mesh_world(link, link_poses) -> trimesh.Trimesh | None:
    """Union (concatenation) of ALL collision geometries of a link, at its init world pose."""
    meshes = []
    for geom in link.geoms:
        trans, quat = gu.transform_pos_quat_by_trans_quat(geom.init_pos, geom.init_quat, *link_poses[link.idx])
        mesh = trimesh.Trimesh(vertices=np.asarray(geom.init_verts), faces=np.asarray(geom.init_faces), process=False)
        mesh.apply_transform(gu.trans_quat_to_T(np.asarray(trans), np.asarray(quat)))
        meshes.append(mesh)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def build_urdf_union_surface(part_rigid, link_poses, pitch: float, inflate: float) -> trimesh.Trimesh:
    """
    Continuous closed surface from the union of ALL collision geometries of every rigid link
    (spec 4.1 URDF fallback: consumes every geometry, not ``link.geoms[0]``).

    The union is expressed as a signed-distance grid (minimum over closed component distances)
    and extracted with marching cubes at iso-level ``inflate``, which both
    unions overlapping shapes and inflates the result outward by ``inflate`` meters in one step.
    """
    link_meshes = [
        m for m in (link_collision_mesh_world(link, link_poses) for link in part_rigid.links) if m is not None
    ]
    if not link_meshes:
        gs.raise_exception("Hybrid URDF fallback found no collision geometry on any rigid link.")
    merged = trimesh.util.concatenate(link_meshes)

    lo = merged.bounds[0] - (inflate + 4 * pitch)
    hi = merged.bounds[1] + (inflate + 4 * pitch)
    dims = np.maximum(np.ceil((hi - lo) / pitch).astype(int) + 1, 2)
    if int(np.prod(dims)) > 4e7:
        gs.raise_exception(
            f"Hybrid URDF fallback voxel grid of {dims.tolist()} cells is too large; "
            "increase `voxel_pitch` or tighten the collision geometry."
        )
    xs = [lo[i] + np.arange(dims[i]) * pitch for i in range(3)]
    grid = np.stack(np.meshgrid(*xs, indexing="ij"), axis=-1).reshape(-1, 3)

    # Minimum signed distance implements a union, including overlaps within a single link.
    distances = np.full(len(grid), np.inf)
    for mesh in link_meshes:
        for component in mesh.split(only_watertight=False):
            validate_solid_surface(component)
            distance = igl.signed_distance(
                grid,
                np.asarray(component.vertices, dtype=np.float64),
                np.asarray(component.faces, dtype=np.int64),
            )[0]
            np.minimum(distances, distance, out=distances)
    sdf = distances
    field = sdf.reshape(dims)
    # Iso-surface at +inflate = the union surface pushed outward by `inflate` meters.
    level = float(inflate)
    if not (field.min() < level < field.max()):
        gs.raise_exception(
            "Hybrid URDF fallback marching-cubes level is outside the sampled SDF range; "
            "check `urdf_inflate` and `voxel_pitch`."
        )
    verts, faces, _, _ = marching_cubes(field, level=level, allow_degenerate=False)
    verts = verts * pitch + lo
    surface = trimesh.Trimesh(vertices=verts, faces=faces, process=True, validate=True)
    if surface.volume < 0:
        surface.invert()
    validate_solid_surface(surface)
    return surface


def _closest_point_barycentric(particles: np.ndarray, surface: trimesh.Trimesh):
    """Closest surface triangle and barycentric coordinates for each query point."""
    verts = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    _, face_idx, closest = igl.point_mesh_squared_distance(np.asarray(particles, dtype=np.float64), verts, faces)
    tri = verts[faces[face_idx]]  # [N, 3, 3]
    v0 = tri[:, 1] - tri[:, 0]
    v1 = tri[:, 2] - tri[:, 0]
    v2 = closest - tri[:, 0]
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01
    v = np.clip((d11 * d20 - d01 * d21) / denom, 0.0, 1.0)
    w = np.clip((d00 * d21 - d01 * d20) / denom, 0.0, 1.0)
    u = np.clip(1.0 - v - w, 0.0, 1.0)
    bary = np.stack([u, v, w], axis=1)
    bary /= bary.sum(axis=1, keepdims=True)
    return face_idx, bary


def transfer_weights_to_particles(
    particles: np.ndarray,
    surface: trimesh.Trimesh,
    vert_link_idx: np.ndarray,
    vert_weights: np.ndarray,
    max_influences: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transfer per-surface-vertex influences to volume particles (spec 4.2): closest-triangle
    barycentric interpolation of the three corner vertices' influence lists, truncated to the
    ``max_influences`` largest and renormalized.
    """
    face_idx, bary = _closest_point_barycentric(particles, surface)
    faces = np.asarray(surface.faces, dtype=np.int64)
    corner_links = vert_link_idx[faces[face_idx]]  # [N, 3, K]
    corner_weights = vert_weights[faces[face_idx]] * bary[:, :, None]  # [N, 3, K]

    n = len(particles)
    cand_links = corner_links.reshape(n, -1)  # [N, 3K]
    cand_weights = corner_weights.reshape(n, -1)
    cand_weights = np.where(cand_links >= 0, cand_weights, 0.0)

    # Merge duplicate link ids per particle by sorting on (link, -weight) and summing runs.
    order = np.argsort(cand_links, axis=1, kind="stable")
    links_sorted = np.take_along_axis(cand_links, order, axis=1)
    weights_sorted = np.take_along_axis(cand_weights, order, axis=1)
    merged_w = np.zeros_like(weights_sorted)
    merged_l = np.full_like(links_sorted, -1)
    # Run-length merge along axis 1 (K is tiny, a python loop over columns is fine).
    for col in range(links_sorted.shape[1]):
        li = links_sorted[:, col]
        wi = weights_sorted[:, col]
        if col == 0:
            merged_l[:, 0] = li
            merged_w[:, 0] = wi
            write_col = np.zeros(n, dtype=np.int64)
            continue
        same = li == merged_l[np.arange(n), write_col]
        merged_w[np.arange(n), write_col] += np.where(same, wi, 0.0)
        advance = ~same
        write_col = write_col + advance
        rows = np.flatnonzero(advance)
        merged_l[rows, write_col[rows]] = li[rows]
        merged_w[rows, write_col[rows]] = wi[rows]

    merged_w = np.where(merged_l >= 0, merged_w, 0.0)
    top = np.argsort(-merged_w, axis=1, kind="stable")[:, :max_influences]
    out_links = np.take_along_axis(merged_l, top, axis=1)
    out_weights = np.take_along_axis(merged_w, top, axis=1)
    out_links = np.where(out_weights > 1e-12, out_links, -1)
    out_weights = np.where(out_links >= 0, out_weights, 0.0)

    row_sum = out_weights.sum(axis=1)
    dead = np.flatnonzero(row_sum <= 1e-12)
    if dead.size:
        gs.raise_exception(
            f"{dead.size} particles received no valid link influence during weight transfer "
            f"(first particle indices: {dead[:8].tolist()}). Check the skinning payload coverage."
        )
    out_weights = out_weights / row_sum[:, None]
    padding = max_influences - out_links.shape[1]
    if padding > 0:
        out_links = np.pad(out_links, ((0, 0), (0, padding)), constant_values=-1)
        out_weights = np.pad(out_weights, ((0, 0), (0, padding)))
    active_count = (out_links >= 0).sum(axis=1).astype(np.int32)
    return out_links.astype(np.int32), out_weights.astype(np.float64), active_count


def nearest_link_fallback_weights(
    particles: np.ndarray,
    links,
    link_poses,
    blend_radius: float,
    max_influences: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth nearest-link weights when no skinning payload is supplied (spec 4.2 fallback): distance
    from each particle to every link's collision surface, converted to weights
    ``w ~ exp(-(d/blend_radius)^2)`` over the closest ``max_influences`` links. With
    ``max_influences=1`` this reproduces legacy hard nearest-link association.
    """
    dists = []
    link_ids = []
    for link in links:
        mesh = link_collision_mesh_world(link, link_poses)
        if mesh is None:
            continue
        sq_dist, _, _ = igl.point_mesh_squared_distance(
            np.asarray(particles, dtype=np.float64),
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
        )
        dists.append(np.sqrt(np.clip(sq_dist, 0.0, None)))
        link_ids.append(link.idx)
    if not dists:
        gs.raise_exception("Nearest-link fallback found no collision geometry on any rigid link.")
    dists = np.stack(dists, axis=1)  # [N, L]
    link_ids = np.asarray(link_ids, dtype=np.int64)

    k = min(max_influences, dists.shape[1])
    nearest = np.argsort(dists, axis=1, kind="stable")[:, :k]
    d_sel = np.take_along_axis(dists, nearest, axis=1)
    # Subtracting the minimum squared distance stabilizes the normalized Gaussian weights.
    w = np.exp(-(d_sel**2 - d_sel[:, :1] ** 2) / blend_radius**2)
    w = w / w.sum(axis=1, keepdims=True)
    out_links = link_ids[nearest]
    pad = max_influences - k
    if pad > 0:
        out_links = np.pad(out_links, ((0, 0), (0, pad)), constant_values=-1)
        w = np.pad(w, ((0, 0), (0, pad)), constant_values=0.0)
    active_count = (out_links >= 0).sum(axis=1).astype(np.int32)
    return out_links.astype(np.int32), w.astype(np.float64), active_count


def transfer_regions_to_particles(particles: np.ndarray, surface: trimesh.Trimesh, vertex_region: np.ndarray):
    """Region ID per particle: region of the barycentric-dominant vertex of the closest triangle."""
    face_idx, bary = _closest_point_barycentric(particles, surface)
    faces = np.asarray(surface.faces, dtype=np.int64)
    corner_regions = vertex_region[faces[face_idx]]  # [N, 3]
    dominant = np.argmax(bary, axis=1)
    return corner_regions[np.arange(len(particles)), dominant].astype(np.int32)


def compute_rest_local_positions(particles: np.ndarray, link_idx: np.ndarray, link_poses) -> np.ndarray:
    """
    Rest-pose particle position expressed in each influencing link's frame:
    ``rest_local[p, i] = T_link_i,rest^-1 * x_rest[p]`` (spec 4.2). Unused slots are zero.
    """
    n, k = link_idx.shape
    rest_local = np.zeros((n, k, 3), dtype=np.float64)
    for slot in range(k):
        for l_idx in np.unique(link_idx[:, slot]):
            if l_idx < 0:
                continue
            rows = np.flatnonzero(link_idx[:, slot] == l_idx)
            rest_local[rows, slot] = gu.inv_transform_by_trans_quat(particles[rows], *link_poses[int(l_idx)])
    return rest_local


def reference_link_poses(links, overrides=None):
    """Initial simulation link frames in world coordinates, keyed by global link index."""
    poses = {}
    for link in links:
        desc = link.desc
        if link.parent_idx < 0:
            pos, quat = np.asarray(desc.pos).copy(), np.asarray(desc.quat).copy()
        else:
            pos, quat = gu.transform_pos_quat_by_trans_quat(desc.pos, desc.quat, *poses[link.parent_idx])
        for joint in link.joints:
            if joint.type == gs.JOINT_TYPE.FREE:
                pos, quat = joint.init_qpos[:3].copy(), joint.init_qpos[3:7].copy()
            elif joint.type == gs.JOINT_TYPE.SPHERICAL:
                anchor = gu.transform_by_trans_quat(joint.pos, pos, quat)
                quat = gu.transform_quat_by_quat(joint.init_qpos, quat)
                pos = anchor - gu.transform_by_quat(joint.pos, quat)
        poses[link.idx] = (pos, quat)
    if overrides:
        links_by_name = {link.name: link.idx for link in links}
        unknown = set(overrides) - set(links_by_name)
        if unknown:
            gs.raise_exception(f"Hybrid reference poses name unknown links: {sorted(unknown)}.")
        for name, value in overrides.items():
            value = np.asarray(value, dtype=np.float64)
            if value.shape != (7,) or not np.isfinite(value).all() or not np.isclose(np.linalg.norm(value[3:]), 1.0):
                gs.raise_exception(f"Hybrid reference pose for {name} must be finite xyz,wxyz with a unit quaternion.")
            poses[links_by_name[name]] = (value[:3].copy(), value[3:].copy())
    return poses


def subtract_particle_inertial(mass, com, inertia, positions, masses):
    """Subtract point masses from a rigid inertial, with all coordinates in the link frame."""
    remaining = mass - masses.sum()
    if remaining <= 0.0:
        raise ValueError("The assigned particle mass exhausts the link mass.")
    first_moment = mass * com - np.einsum("n,ni->i", masses, positions)
    center = first_moment / remaining
    origin_inertia = inertia + mass * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
    particle_inertia = np.einsum("n,n->", masses, np.einsum("ni,ni->n", positions, positions)) * np.eye(3)
    particle_inertia -= np.einsum("n,ni,nj->ij", masses, positions, positions)
    remaining_inertia = origin_inertia - particle_inertia
    remaining_inertia -= remaining * (np.dot(center, center) * np.eye(3) - np.outer(center, center))
    eigenvalues = np.linalg.eigvalsh(remaining_inertia)
    if eigenvalues[0] <= 0.0 or eigenvalues[2] > eigenvalues[:2].sum() * (1.0 + 1e-9):
        raise ValueError("The assigned particles leave a nonphysical link inertia.")
    return remaining, center, remaining_inertia
