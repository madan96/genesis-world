import numpy as np
import pytest
import trimesh

import genesis as gs
from genesis.utils import geom as gu
from genesis.utils import hybrid_skin

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.parametrize("is_transformed", [False, True])
def test_collision_union_volume(skin_collision_union, show_viewer, is_transformed):
    PITCH = 0.002
    position = np.array([0.3, -0.2, 0.5]) if is_transformed else np.zeros(3)
    quaternion = gu.xyz_to_quat(np.array([0.0, 0.0, np.pi / 2 if is_transformed else 0.0]))
    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    entity = scene.add_entity(
        morph=gs.morphs.URDF(
            file=skin_collision_union,
            pos=position,
            quat=quaternion,
            fixed=True,
        ),
    )
    poses = hybrid_skin.reference_link_poses(entity.links)
    surface = hybrid_skin.build_urdf_union_surface(entity, poses, pitch=PITCH, inflate=0.0)
    hybrid_skin.validate_solid_surface(surface)
    # The union is a 0.12 x 0.08 x 0.08 cuboid plus a separate 0.08 m cube. Edge cells bound voxelization error
    assert_allclose(surface.volume, (0.12 + 0.08) * 0.08**2, atol=2e-5)
    vertices_local = gu.inv_transform_by_trans_quat(surface.vertices, position, quaternion)
    assert_allclose(vertices_local.min(axis=0), (-0.06, -0.04, -0.04), atol=PITCH)
    assert_allclose(vertices_local.max(axis=0), (0.24, 0.04, 0.04), atol=PITCH)
    assert_allclose(
        poses[entity.links[1].idx][0],
        gu.transform_by_trans_quat(np.array([0.2, 0, 0]), position, quaternion),
        atol=1e-14,
    )


@pytest.mark.required
@pytest.mark.parametrize("defect", ["open", "inward", "degenerate", "nonfinite", "index"])
def test_solid_surface_validation(defect):
    surface = trimesh.creation.box(extents=(0.2, 0.3, 0.4))
    copied = hybrid_skin.load_surface(surface, scale=2)
    hybrid_skin.validate_solid_surface(copied)
    assert_allclose(copied.volume, 8 * surface.volume, atol=1e-15)
    assert_allclose(surface.extents, (0.2, 0.3, 0.4), atol=1e-15)
    vertices = np.asarray(surface.vertices).copy()
    faces = np.asarray(surface.faces).copy()
    if defect == "open":
        faces = faces[1:]
    elif defect == "inward":
        faces = faces[:, ::-1]
    elif defect == "degenerate":
        faces[0, 1] = faces[0, 0]
    elif defect == "nonfinite":
        vertices[0, 0] = np.nan
    else:
        faces[0, 0] = len(vertices)
    malformed = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    with pytest.raises(gs.GenesisException):
        hybrid_skin.validate_solid_surface(malformed)


@pytest.mark.required
@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "float_index", "range", "unused", "negative", "nonfinite", "zero"]
)
def test_skinning_validation(defect):
    surface = trimesh.creation.box()
    payload = {
        "link_names": ["base", "tip"],
        "indices": np.tile([0, 1, -1], (len(surface.vertices), 1)),
        "weights": np.tile([2.0, 3.0, 0.0], (len(surface.vertices), 1)),
    }
    indices, weights = hybrid_skin.load_skinning(payload, surface, {"base": 4, "tip": 9})
    assert_equal(indices, (4, 9, -1))
    assert_allclose(weights, (0.4, 0.6, 0), atol=1e-15)
    if defect == "missing":
        payload["link_names"] = ["base", "unknown"]
    elif defect == "duplicate":
        payload["link_names"] = ["base", "base"]
    elif defect == "float_index":
        payload["indices"] = payload["indices"].astype(float)
    elif defect == "range":
        payload["indices"][0, 0] = 2
    elif defect == "unused":
        payload["weights"][0, 2] = 1
    elif defect == "negative":
        payload["weights"][0, 0] = -1
    elif defect == "nonfinite":
        payload["weights"][0, 0] = np.nan
    else:
        payload["weights"][0] = 0
    with pytest.raises(gs.GenesisException):
        hybrid_skin.load_skinning(payload, surface, {"base": 4, "tip": 9})


@pytest.mark.required
@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e308])
def test_skinning_weight_scale(scale):
    surface = trimesh.creation.box()
    indices, weights = hybrid_skin.load_skinning(
        {
            "link_names": ["base", "tip"],
            "indices": np.tile([0, 1], (len(surface.vertices), 1)),
            "weights": np.tile([scale, scale], (len(surface.vertices), 1)),
        },
        surface,
        {"base": 4, "tip": 9},
    )
    assert_equal(indices, (4, 9))
    assert_allclose(weights, (0.5, 0.5), atol=1e-15)


@pytest.mark.required
def test_surface_attribute_transfer():
    surface = trimesh.Trimesh(vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0)), faces=((0, 1, 2),), process=False)
    particles = np.array([[0.3, 0.5, 0.2]])
    indices, weights = hybrid_skin.load_skinning(
        {
            "link_names": ["base", "middle", "tip"],
            "indices": np.array([[0, 1], [1, 2], [0, 2]]),
            "weights": np.array([[1, 0], [0.5, 0.5], [0.2, 0.8]]),
        },
        surface,
        {"base": 4, "middle": 9, "tip": 12},
    )
    links, transferred, counts = hybrid_skin.transfer_weights_to_particles(particles, surface, indices, weights, 4)
    assert_equal(links, (12, 4, 9, -1))
    assert_equal(counts, 3)
    assert_allclose(transferred, (0.55, 0.3, 0.15, 0), atol=1e-15)
    links, transferred, counts = hybrid_skin.transfer_weights_to_particles(particles, surface, indices, weights, 2)
    assert_equal(links, (12, 4))
    assert_equal(counts, 2)
    assert_allclose(transferred, np.array([0.55, 0.3]) / 0.85, atol=1e-15)
    names, regions = hybrid_skin.load_regions(
        {"region_names": ["root", "shaft", "tip"], "vertex_region": np.array([0, 1, 2])}, surface
    )
    assert names == ["root", "shaft", "tip"]
    assert_equal(hybrid_skin.transfer_regions_to_particles(particles, surface, regions), 2)
    material, contact = hybrid_skin.load_vertex_attributes(
        surface, np.array([0, 1, 0]), np.array([True, False, True]), 2
    )
    assert_equal(material, (0, 1, 0))
    assert_equal(contact, (True, False, True))
    for material, contact in (
        (np.array([0, 2, 0]), None),
        (np.array([0.0, 1.0, 0.0]), None),
        (np.array([0, 1]), None),
        (None, np.array([1, 0, 1])),
        (None, np.array([True, False])),
    ):
        with pytest.raises(gs.GenesisException):
            hybrid_skin.load_vertex_attributes(surface, material, contact, 2)
    for names, regions in ((["a", "a"], [0, 0, 0]), (["a"], [0, 1, 0]), (["a"], [0.0, 0.0, 0.0])):
        with pytest.raises(gs.GenesisException):
            hybrid_skin.load_regions({"region_names": names, "vertex_region": np.array(regions)}, surface)


@pytest.mark.required
@pytest.mark.parametrize("transformed", [False, True])
def test_rigid_total_inertial_subtraction(transformed):
    points = np.array(
        [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]],
        dtype=float,
    )
    points *= (0.1, 0.2, 0.3)
    masses = np.arange(1, 9, dtype=float) / 10
    if transformed:
        rotation = np.array([[0, -1, 0], [0.6, 0, -0.8], [0.8, 0, 0.6]])
        points = points @ rotation.T + (1.2, -0.7, 0.3)
    inertials = []
    for selected in (slice(None), slice(3, None)):
        mass = masses[selected].sum()
        center = np.average(points[selected], axis=0, weights=masses[selected])
        x, y, z = (points[selected] - center).T
        weights = masses[selected]
        inertia = np.array(
            [
                [weights @ (y * y + z * z), -weights @ (x * y), -weights @ (x * z)],
                [-weights @ (x * y), weights @ (x * x + z * z), -weights @ (y * z)],
                [-weights @ (x * z), -weights @ (y * z), weights @ (x * x + y * y)],
            ]
        )
        inertials.append((mass, center, inertia))
    remaining = hybrid_skin.subtract_particle_inertial(*inertials[0], points[:3], masses[:3])
    for actual, expected in zip(remaining, inertials[1]):
        assert_allclose(actual, expected, atol=2e-14)
    with pytest.raises(ValueError, match="exhausts"):
        hybrid_skin.subtract_particle_inertial(*inertials[0], points, 2 * masses)
    with pytest.raises(ValueError, match="nonphysical"):
        hybrid_skin.subtract_particle_inertial(
            1, np.zeros(3), np.eye(3) * 0.001, np.array([[1, 0, 0]]), np.array([0.1])
        )
