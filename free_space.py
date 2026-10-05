import os
import struct
import argparse

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
from plyfile import PlyData
from tqdm.auto import tqdm


# ============================================================
# 1. ARGUMENTS / PATHS / SETTINGS
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source_path", required=True)
args = parser.parse_args()

# Input is the scene directory itself:
# /.../scene-001
SCENE_DIR = os.path.normpath(args.source_path)

SPARSE_DIR = os.path.join(SCENE_DIR, "dense", "sparse", "0")

POINTS_PATH = os.path.join(SPARSE_DIR, "points3D.bin")
IMAGES_PATH = os.path.join(SPARSE_DIR, "images.bin")
COLMAP_PLY = os.path.join(SPARSE_DIR, "points3D.ply")

OUTPUT_PATH = os.path.join(SCENE_DIR, "free_space_field.npz")

BASE_RESOLUTION = 128
EXPANSION_FACTOR = 2.0

FREE_SPACE_THRESHOLD = 1
CONFIDENCE_SATURATION = 105
RAY_STEP_FACTOR = 0.75

# Adaptive refinement:
# 0 = only voxels containing COLMAP points
# 1 = those voxels + their 26 neighbors
NEIGHBOR_RADIUS = 0
REFINE_FACTOR = 2


# ============================================================
# 2. COLMAP POINT LOADING
# ============================================================

def read_points3d_with_tracks(path):
    points = {}

    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_points):
            point_id = struct.unpack("<Q", f.read(8))[0]

            xyz = np.array(
                struct.unpack("<ddd", f.read(24)),
                dtype=np.float64
            )

            f.read(3)  # RGB
            f.read(8)  # reprojection error

            track_length = struct.unpack("<Q", f.read(8))[0]

            track = []
            for _ in range(track_length):
                image_id, point2d_idx = struct.unpack("<ii", f.read(8))
                track.append(image_id)

            points[point_id] = {
                "xyz": xyz,
                "image_ids": track
            }

    return points


# ============================================================
# 3. COLMAP CAMERA LOADING
# ============================================================

def read_images_binary(path):
    images = {}

    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_images):
            image_id = struct.unpack("<i", f.read(4))[0]

            qvec = np.array(struct.unpack("<dddd", f.read(32)))
            tvec = np.array(struct.unpack("<ddd", f.read(24)))

            camera_id = struct.unpack("<i", f.read(4))[0]

            name = b""
            while True:
                char = f.read(1)

                if char == b"\x00":
                    break

                if not char:
                    raise EOFError(
                        "Unexpected EOF while reading image name"
                    )

                name += char

            name = name.decode("utf-8")

            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.seek(num_points2D * 24, 1)

            R = Rotation.from_quat(
                [qvec[1], qvec[2], qvec[3], qvec[0]]
            ).as_matrix()

            C = -R.T @ tvec

            images[image_id] = {
                "name": name,
                "camera_center": C
            }

    return images


# ============================================================
# 4. BOUNDS AND VOXEL GRID
# ============================================================

def compute_bounds(points_xyz):
    if len(points_xyz) == 0:
        raise ValueError("No COLMAP points found.")

    percentile_min = np.percentile(points_xyz, 1, axis=0)
    percentile_max = np.percentile(points_xyz, 99, axis=0)

    original_center = (
        percentile_min + percentile_max
    ) / 2.0

    original_extent = percentile_max - percentile_min

    if np.any(original_extent <= 0):
        raise ValueError("Invalid scene extent.")

    original_voxel_size = (
        original_extent / BASE_RESOLUTION
    )

    new_extent = original_extent * EXPANSION_FACTOR

    bound_min = original_center - new_extent / 2.0
    bound_max = original_center + new_extent / 2.0

    grid_resolution = np.ceil(
        new_extent / original_voxel_size
    ).astype(int)

    voxel_size = (
        bound_max - bound_min
    ) / grid_resolution

    print("\nOriginal bounds:")
    print("Min:", percentile_min)
    print("Max:", percentile_max)

    print("\nExpanded bounds:")
    print("Min:", bound_min)
    print("Max:", bound_max)

    print("\nOriginal extent:", original_extent)
    print("Expanded extent:", new_extent)

    print("\nOriginal voxel size:", original_voxel_size)
    print("New voxel size:", voxel_size)

    print("\nGrid resolution:", grid_resolution)
    print("Total voxels:", np.prod(grid_resolution))

    return (
        bound_min,
        bound_max,
        grid_resolution,
        voxel_size
    )


# ============================================================
# 5. FREE-SPACE RAY GENERATION
# ============================================================

def generate_ray_count(
    points,
    images,
    bound_min,
    bound_max,
    grid_resolution,
    voxel_size
):
    NX, NY, NZ = grid_resolution

    ray_count = np.zeros(
        (NX, NY, NZ),
        dtype=np.uint16
    )

    step_size = voxel_size.min() * RAY_STEP_FACTOR

    print("\nGrid:", ray_count.shape)
    print("Voxel size:", voxel_size)
    print("Step size:", step_size)

    total_rays = 0
    valid_rays = 0

    for point_id, point_data in tqdm(
        points.items(),
        desc="Generating rays"
    ):
        P = point_data["xyz"]

        for image_id in point_data["image_ids"]:
            if image_id not in images:
                continue

            C = images[image_id]["camera_center"]

            direction = P - C
            ray_length = np.linalg.norm(direction)

            if ray_length < 1e-8:
                continue

            total_rays += 1

            num_steps = int(np.ceil(ray_length / step_size))

            if num_steps <= 1:
                continue

            t_values = np.arange(num_steps) / num_steps

            samples = (
                C[None, :]
                + t_values[:, None] * direction[None, :]
            )

            ijk = np.floor(
                (samples - bound_min) / voxel_size
            ).astype(np.int32)

            inside = (
                (ijk[:, 0] >= 0) & (ijk[:, 0] < NX) &
                (ijk[:, 1] >= 0) & (ijk[:, 1] < NY) &
                (ijk[:, 2] >= 0) & (ijk[:, 2] < NZ)
            )

            ijk = ijk[inside]

            if len(ijk) == 0:
                continue

            valid_rays += 1

            # One vote per ray per voxel.
            ijk = np.unique(ijk, axis=0)

            np.add.at(
                ray_count,
                (
                    ijk[:, 0],
                    ijk[:, 1],
                    ijk[:, 2]
                ),
                1
            )

    print("\nRay generation complete.")
    print("Total rays:", total_rays)
    print("Valid rays:", valid_rays)
    print("Nonzero voxels:", np.count_nonzero(ray_count))
    print("Maximum ray count:", ray_count.max())

    return ray_count


# ============================================================
# 6. SURFACE / FREE / UNKNOWN CLASSIFICATION
# ============================================================

def build_masks(
    ray_count,
    colmap_xyz,
    bound_min,
    bound_max
):
    grid_shape = np.array(ray_count.shape)

    voxel_size = (
        bound_max - bound_min
    ) / grid_shape

    voxel_indices = np.floor(
        (colmap_xyz - bound_min) / voxel_size
    ).astype(np.int64)

    valid = np.all(
        (voxel_indices >= 0) &
        (voxel_indices < grid_shape),
        axis=1
    )

    voxel_indices = voxel_indices[valid]

    surface_mask = np.zeros(
        grid_shape,
        dtype=bool
    )

    surface_mask[
        voxel_indices[:, 0],
        voxel_indices[:, 1],
        voxel_indices[:, 2]
    ] = True

    free_mask = ray_count >= FREE_SPACE_THRESHOLD
    free_mask &= ~surface_mask

    unknown_mask = ~(free_mask | surface_mask)

    return (
        free_mask,
        surface_mask,
        unknown_mask,
        voxel_indices,
        voxel_size
    )


# ============================================================
# 7. DISTANCE FIELD AND CONFIDENCE
# ============================================================

def compute_distance_and_confidence(
    ray_count,
    free_mask,
    voxel_size
):
    distance_world = distance_transform_edt(
        free_mask,
        sampling=voxel_size
    ).astype(np.float32)

    confidence = np.clip(
        (
            ray_count.astype(np.float32)
            - FREE_SPACE_THRESHOLD
        )
        / (
            CONFIDENCE_SATURATION
            - FREE_SPACE_THRESHOLD
        ),
        0.0,
        1.0
    )

    confidence *= free_mask

    return distance_world, confidence


# ============================================================
# 8. ADAPTIVE REFINEMENT
# ============================================================

def expand_voxel_neighborhood(
    voxel_indices,
    radius,
    grid_shape
):
    if radius == 0:
        return np.unique(voxel_indices, axis=0)

    offsets = np.arange(-radius, radius + 1)

    dx, dy, dz = np.meshgrid(
        offsets,
        offsets,
        offsets,
        indexing="ij"
    )

    offsets = np.stack(
        [dx.ravel(), dy.ravel(), dz.ravel()],
        axis=1
    )

    expanded = (
        voxel_indices[:, None, :]
        + offsets[None, :, :]
    ).reshape(-1, 3)

    valid = (
        (expanded[:, 0] >= 0) &
        (expanded[:, 0] < grid_shape[0]) &
        (expanded[:, 1] >= 0) &
        (expanded[:, 1] < grid_shape[1]) &
        (expanded[:, 2] >= 0) &
        (expanded[:, 2] < grid_shape[2])
    )

    return np.unique(expanded[valid], axis=0)


def generate_adaptive_children(
    points,
    images,
    colmap_xyz,
    bound_min,
    grid_resolution,
    voxel_size,
    threshold
):
    NX, NY, NZ = grid_resolution

    # --------------------------------------------------------
    # Parent voxels containing COLMAP points
    # --------------------------------------------------------

    point_ijk = np.floor(
        (colmap_xyz - bound_min) / voxel_size
    ).astype(np.int32)

    point_ijk[:, 0] = np.clip(
        point_ijk[:, 0], 0, NX - 1
    )
    point_ijk[:, 1] = np.clip(
        point_ijk[:, 1], 0, NY - 1
    )
    point_ijk[:, 2] = np.clip(
        point_ijk[:, 2], 0, NZ - 1
    )

    surface_parent_voxels = np.unique(
        point_ijk,
        axis=0
    )

    # --------------------------------------------------------
    # Expand by neighbour radius
    # --------------------------------------------------------

    refine_parent_voxels = expand_voxel_neighborhood(
        surface_parent_voxels,
        NEIGHBOR_RADIUS,
        (NX, NY, NZ)
    )

    refine_parent_set = {
        tuple(v) for v in refine_parent_voxels
    }

    # --------------------------------------------------------
    # Fine grid
    # --------------------------------------------------------

    fine_voxel_size = voxel_size / REFINE_FACTOR
    step_size_fine = (
        fine_voxel_size.min() * RAY_STEP_FACTOR
    )

    child_ray_count = {}

    print("\nAdaptive refinement")
    print("-------------------")
    print(
        "Surface parent voxels:",
        len(surface_parent_voxels)
    )
    print(
        "Refined parent voxels:",
        len(refine_parent_voxels)
    )
    print(
        "Neighbor radius:",
        NEIGHBOR_RADIUS
    )
    print(
        "Children per parent:",
        REFINE_FACTOR ** 3
    )
    print(
        "Fine voxel size:",
        fine_voxel_size
    )
    print(
        "Fine ray step:",
        step_size_fine
    )

    # --------------------------------------------------------
    # Rerasterize camera -> COLMAP point rays
    # --------------------------------------------------------

    for point_id, point_data in tqdm(
        points.items(),
        desc="Adaptive ray rasterization"
    ):
        P = point_data["xyz"]

        for image_id in set(point_data["image_ids"]):
            if image_id not in images:
                continue

            C = images[image_id]["camera_center"]

            ray = P - C
            ray_length = np.linalg.norm(ray)

            if ray_length <= 1e-8:
                continue

            num_steps = int(
                np.ceil(ray_length / step_size_fine)
            )

            if num_steps <= 1:
                continue

            t_values = (
                np.arange(num_steps, dtype=np.float32)
                / num_steps
            )

            samples = (
                C[None, :]
                + t_values[:, None] * ray[None, :]
            )

            # Parent voxel coordinates.
            parent_ijk = np.floor(
                (samples - bound_min) / voxel_size
            ).astype(np.int32)

            inside = (
                (parent_ijk[:, 0] >= 0) &
                (parent_ijk[:, 0] < NX) &
                (parent_ijk[:, 1] >= 0) &
                (parent_ijk[:, 1] < NY) &
                (parent_ijk[:, 2] >= 0) &
                (parent_ijk[:, 2] < NZ)
            )

            parent_ijk = parent_ijk[inside]
            samples = samples[inside]

            if len(parent_ijk) == 0:
                continue

            # Keep only samples inside refined parents.
            keep = np.array([
                tuple(v) in refine_parent_set
                for v in parent_ijk
            ])

            if not np.any(keep):
                continue

            samples = samples[keep]

            # Fine child coordinates.
            child_ijk = np.floor(
                (samples - bound_min) / fine_voxel_size
            ).astype(np.int32)

            fine_shape = (
                np.asarray(grid_resolution, dtype=np.int32)
                * REFINE_FACTOR
            )

            valid_child = (
                (child_ijk[:, 0] >= 0) &
                (child_ijk[:, 0] < fine_shape[0]) &
                (child_ijk[:, 1] >= 0) &
                (child_ijk[:, 1] < fine_shape[1]) &
                (child_ijk[:, 2] >= 0) &
                (child_ijk[:, 2] < fine_shape[2])
            )

            child_ijk = child_ijk[valid_child]

            if len(child_ijk) == 0:
                continue

            # One vote per ray per child.
            child_ijk = np.unique(
                child_ijk,
                axis=0
            )

            for idx in child_ijk:
                key = tuple(idx)
                child_ray_count[key] = (
                    child_ray_count.get(key, 0) + 1
                )

    # --------------------------------------------------------
    # Apply child ray-evidence threshold
    # --------------------------------------------------------

    child_indices = []
    child_counts = []

    for idx, count in child_ray_count.items():
        if count >= threshold:
            child_indices.append(idx)
            child_counts.append(count)

    if child_indices:
        child_indices = np.asarray(
            child_indices,
            dtype=np.int32
        )
        child_counts = np.asarray(
            child_counts,
            dtype=np.uint16
        )
    else:
        child_indices = np.empty(
            (0, 3),
            dtype=np.int32
        )
        child_counts = np.empty(
            (0,),
            dtype=np.uint16
        )

    print("\nChild voxel statistics")
    print("----------------------")
    print(
        "Children with any ray evidence:",
        len(child_ray_count)
    )
    print(
        "Children passing threshold:",
        len(child_indices)
    )

    if len(child_counts) > 0:
        print(
            "Maximum child ray count:",
            child_counts.max()
        )
        print(
            "Mean surviving child ray count:",
            child_counts.mean()
        )
        print(
            "Median surviving child ray count:",
            np.median(child_counts)
        )

    return (
        refine_parent_voxels,
        child_indices,
        child_counts,
        fine_voxel_size
    )


# ============================================================
# 9. FIELD GENERATION PIPELINE
# ============================================================

def build_free_space_field():
    print("Loading COLMAP points...")

    points = read_points3d_with_tracks(
        POINTS_PATH
    )

    points_xyz = np.array(
        [p["xyz"] for p in points.values()],
        dtype=np.float64
    )

    print(
        "COLMAP points:",
        len(points_xyz)
    )
    print(
        "Point array shape:",
        points_xyz.shape
    )

    print("\nLoading registered cameras...")

    images = read_images_binary(
        IMAGES_PATH
    )

    print(
        "Registered images:",
        len(images)
    )

    camera_centers = np.array([
        img["camera_center"]
        for img in images.values()
    ])

    print(
        "Camera centers shape:",
        camera_centers.shape
    )

    print("\nComputing bounds...")

    (
        bound_min,
        bound_max,
        grid_resolution,
        voxel_size
    ) = compute_bounds(points_xyz)

    print("\nGenerating free-space rays...")

    ray_count = generate_ray_count(
        points,
        images,
        bound_min,
        bound_max,
        grid_resolution,
        voxel_size
    )

    print("\nLoading COLMAP surface points...")

    ply = PlyData.read(COLMAP_PLY)
    vertices = ply["vertex"]

    colmap_xyz = np.column_stack([
        vertices["x"],
        vertices["y"],
        vertices["z"]
    ])

    print(
        "PLY points:",
        len(colmap_xyz)
    )

    print("\nBuilding masks...")

    (
        free_mask,
        surface_mask,
        unknown_mask,
        voxel_indices,
        voxel_size
    ) = build_masks(
        ray_count,
        colmap_xyz,
        bound_min,
        bound_max
    )

    print("\nComputing distance field...")

    distance_world, confidence = (
        compute_distance_and_confidence(
            ray_count,
            free_mask,
            voxel_size
        )
    )

    print("\nGenerating adaptive refinement...")

    (
        refine_parent_voxels,
        child_indices,
        child_ray_count,
        fine_voxel_size
    ) = generate_adaptive_children(
        points,
        images,
        colmap_xyz,
        bound_min,
        grid_resolution,
        voxel_size,
        FREE_SPACE_THRESHOLD
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    np.savez_compressed(
        OUTPUT_PATH,

        # Coarse field
        ray_count=ray_count,
        free_mask=free_mask,
        surface_mask=surface_mask,
        unknown_mask=unknown_mask,

        distance_world=distance_world,
        confidence=confidence,

        bound_min=bound_min,
        bound_max=bound_max,

        voxel_size=voxel_size,
        grid_resolution=grid_resolution,

        threshold=FREE_SPACE_THRESHOLD,

        # Adaptive field
        refine_parent_voxels=refine_parent_voxels,
        child_indices=child_indices,
        child_ray_count=child_ray_count,
        fine_voxel_size=fine_voxel_size,
        neighbor_radius=NEIGHBOR_RADIUS,
        refine_factor=REFINE_FACTOR
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_voxels = free_mask.size

    print("\nSaved:", OUTPUT_PATH)

    print(
        "\n========== FIELD STATISTICS =========="
    )
    print(
        "Grid shape:",
        ray_count.shape
    )
    print(
        "Total voxels:",
        total_voxels
    )

    print(
        f"Free voxels: {free_mask.sum():,} "
        f"({free_mask.mean() * 100:.6f}%)"
    )

    print(
        f"COLMAP surface voxels: "
        f"{surface_mask.sum():,} "
        f"({surface_mask.mean() * 100:.6f}%)"
    )

    print(
        f"Unknown voxels: {unknown_mask.sum():,} "
        f"({unknown_mask.mean() * 100:.6f}%)"
    )

    print(
        "COLMAP points:",
        len(colmap_xyz)
    )

    print(
        "COLMAP points inside grid:",
        len(voxel_indices)
    )

    print(
        "COLMAP points in free voxels:",
        int(
            free_mask[
                voxel_indices[:, 0],
                voxel_indices[:, 1],
                voxel_indices[:, 2]
            ].sum()
        )
    )

    print(
        "Maximum distance:",
        distance_world.max()
    )

    nonzero = ray_count > 0

    print(
        "Mean nonzero ray count:",
        ray_count[nonzero].mean()
        if np.any(nonzero)
        else 0
    )

    print(
        "\n========== ADAPTIVE FIELD =========="
    )
    print(
        "Neighbor radius:",
        NEIGHBOR_RADIUS
    )
    print(
        "Refine factor:",
        REFINE_FACTOR
    )
    print(
        "Refined parent voxels:",
        len(refine_parent_voxels)
    )
    print(
        "Surviving child voxels:",
        len(child_indices)
    )


# ============================================================
# 10. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    build_free_space_field()
