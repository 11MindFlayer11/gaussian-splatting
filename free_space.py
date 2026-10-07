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
parser.add_argument("-o", "--output_path", default=None)
parser.add_argument("--base_resolution", type=int, default=128)
parser.add_argument("--expansion_factor", type=float, default=2.0)
parser.add_argument("--free_space_threshold", type=int, default=1)
parser.add_argument("--confidence_saturation", type=int, default=105)
parser.add_argument("--ray_step_factor", type=float, default=0.75)
parser.add_argument("--neighbor_radius", type=int, default=None)
parser.add_argument("--refine_factor", type=int, default=2)
parser.add_argument("--local_radius", type=int, default=1)
args = parser.parse_args()

SCENE_DIR = os.path.normpath(args.source_path)
SPARSE_DIR = os.path.join(SCENE_DIR, "dense", "sparse", "0")
POINTS_PATH = os.path.join(SPARSE_DIR, "points3D.bin")
IMAGES_PATH = os.path.join(SPARSE_DIR, "images.bin")
COLMAP_PLY = os.path.join(SPARSE_DIR, "points3D.ply")
OUTPUT_PATH = args.output_path if args.output_path is not None else os.path.join(SCENE_DIR, "free_space_field.npz")

BASE_RESOLUTION = args.base_resolution
EXPANSION_FACTOR = args.expansion_factor
FREE_SPACE_THRESHOLD = args.free_space_threshold
CONFIDENCE_SATURATION = args.confidence_saturation
RAY_STEP_FACTOR = args.ray_step_factor
NEIGHBOR_RADIUS = args.neighbor_radius
REFINE_FACTOR = args.refine_factor
LOCAL_RADIUS = args.local_radius

# ============================================================
# 2. COLMAP POINT / CAMERA LOADING
# ============================================================
def read_points3d_with_tracks(path):
    points = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point_id = struct.unpack("<Q", f.read(8))[0]
            xyz = np.array(struct.unpack("<ddd", f.read(24)), dtype=np.float64)
            f.read(3)
            f.read(8)
            track_length = struct.unpack("<Q", f.read(8))[0]
            track = []
            for _ in range(track_length):
                image_id, _ = struct.unpack("<ii", f.read(8))
                track.append(image_id)
            points[point_id] = {"xyz": xyz, "image_ids": track}
    return points

def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<i", f.read(4))[0]
            qvec = np.array(struct.unpack("<dddd", f.read(32)))
            tvec = np.array(struct.unpack("<ddd", f.read(24)))
            _ = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                char = f.read(1)
                if char == b"\x00":
                    break
                if not char:
                    raise EOFError("Unexpected EOF while reading image name")
                name += char
            name = name.decode("utf-8")
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.seek(num_points2D * 24, 1)
            R = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
            C = -R.T @ tvec
            images[image_id] = {"name": name, "camera_center": C}
    return images

# ============================================================
# 3. BOUNDS / VOXEL GRID
# ============================================================
def compute_bounds(points_xyz):
    if len(points_xyz) == 0:
        raise ValueError("No COLMAP points found.")
    percentile_min = np.percentile(points_xyz, 1, axis=0)
    percentile_max = np.percentile(points_xyz, 99, axis=0)
    original_center = (percentile_min + percentile_max) / 2.0
    original_extent = percentile_max - percentile_min
    if np.any(original_extent <= 0):
        raise ValueError("Invalid scene extent.")
    original_voxel_size = original_extent / BASE_RESOLUTION
    new_extent = original_extent * EXPANSION_FACTOR
    bound_min = original_center - new_extent / 2.0
    bound_max = original_center + new_extent / 2.0
    grid_resolution = np.ceil(new_extent / original_voxel_size).astype(int)
    voxel_size = (bound_max - bound_min) / grid_resolution
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
    return bound_min, bound_max, grid_resolution, voxel_size

# ============================================================
# 4. COARSE FREE-SPACE RAY GENERATION
# ============================================================
def generate_ray_count(points, images, bound_min, bound_max, grid_resolution, voxel_size):
    NX, NY, NZ = grid_resolution
    ray_count = np.zeros((NX, NY, NZ), dtype=np.uint16)
    step_size = voxel_size.min() * RAY_STEP_FACTOR
    print("\nGrid:", ray_count.shape)
    print("Voxel size:", voxel_size)
    print("Step size:", step_size)
    total_rays = 0
    valid_rays = 0

    for _, point_data in tqdm(points.items(), desc="Generating rays"):
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
            samples = C[None, :] + t_values[:, None] * direction[None, :]
            ijk = np.floor((samples - bound_min) / voxel_size).astype(np.int32)
            inside = ((ijk[:, 0] >= 0) & (ijk[:, 0] < NX) &
                      (ijk[:, 1] >= 0) & (ijk[:, 1] < NY) &
                      (ijk[:, 2] >= 0) & (ijk[:, 2] < NZ))
            ijk = ijk[inside]
            if len(ijk) == 0:
                continue
            valid_rays += 1
            ijk = np.unique(ijk, axis=0)
            np.add.at(ray_count, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1)

    print("\nRay generation complete.")
    print("Total rays:", total_rays)
    print("Valid rays:", valid_rays)
    print("Nonzero voxels:", np.count_nonzero(ray_count))
    print("Maximum ray count:", ray_count.max())
    return ray_count

# ============================================================
# 5. SURFACE / FREE / UNKNOWN CLASSIFICATION
# ============================================================
def build_masks(ray_count, colmap_xyz, bound_min, bound_max):
    grid_shape = np.array(ray_count.shape)
    voxel_size = (bound_max - bound_min) / grid_shape
    voxel_indices = np.floor((colmap_xyz - bound_min) / voxel_size).astype(np.int64)
    valid = np.all((voxel_indices >= 0) & (voxel_indices < grid_shape), axis=1)
    voxel_indices = voxel_indices[valid]
    surface_mask = np.zeros(grid_shape, dtype=bool)
    surface_mask[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True
    free_mask = ray_count >= FREE_SPACE_THRESHOLD
    free_mask &= ~surface_mask
    unknown_mask = ~(free_mask | surface_mask)
    return free_mask, surface_mask, unknown_mask, voxel_indices, voxel_size

# ============================================================
# 6. COARSE DISTANCE FIELD / CONFIDENCE
# ============================================================
def compute_distance_and_confidence(ray_count, free_mask, voxel_size):
    distance_world = distance_transform_edt(free_mask, sampling=voxel_size).astype(np.float32)
    denominator = CONFIDENCE_SATURATION - FREE_SPACE_THRESHOLD
    confidence = np.clip((ray_count.astype(np.float32) - FREE_SPACE_THRESHOLD) / denominator, 0.0, 1.0)
    confidence *= free_mask
    return distance_world, confidence

# ============================================================
# 7. ADAPTIVE PARENT SELECTION
# ============================================================
def expand_voxel_neighborhood(voxel_indices, radius, grid_shape):
    if radius == 0:
        return np.unique(voxel_indices, axis=0)
    offsets = np.arange(-radius, radius + 1)
    dx, dy, dz = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    offsets = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
    expanded = (voxel_indices[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    valid = ((expanded[:, 0] >= 0) & (expanded[:, 0] < grid_shape[0]) &
             (expanded[:, 1] >= 0) & (expanded[:, 1] < grid_shape[1]) &
             (expanded[:, 2] >= 0) & (expanded[:, 2] < grid_shape[2]))
    return np.unique(expanded[valid], axis=0)

# ============================================================
# 8. ADAPTIVE CHILD RAY EVIDENCE
# ============================================================
def generate_adaptive_children(points, images, colmap_xyz, bound_min, grid_resolution, voxel_size, threshold):
    NX, NY, NZ = grid_resolution
    point_ijk = np.floor((colmap_xyz - bound_min) / voxel_size).astype(np.int32)
    point_ijk[:, 0] = np.clip(point_ijk[:, 0], 0, NX - 1)
    point_ijk[:, 1] = np.clip(point_ijk[:, 1], 0, NY - 1)
    point_ijk[:, 2] = np.clip(point_ijk[:, 2], 0, NZ - 1)
    surface_parent_voxels = np.unique(point_ijk, axis=0)
    refine_parent_voxels = expand_voxel_neighborhood(surface_parent_voxels, NEIGHBOR_RADIUS, (NX, NY, NZ))
    refine_parent_set = {tuple(v) for v in refine_parent_voxels}
    fine_voxel_size = voxel_size / REFINE_FACTOR
    step_size_fine = fine_voxel_size.min() * RAY_STEP_FACTOR
    fine_shape = np.asarray(grid_resolution, dtype=np.int32) * REFINE_FACTOR
    child_ray_count = {}

    print("\nAdaptive refinement")
    print("-------------------")
    print("Surface parent voxels:", len(surface_parent_voxels))
    print("Refined parent voxels:", len(refine_parent_voxels))
    print("Neighbor radius:", NEIGHBOR_RADIUS)
    print("Children per parent:", REFINE_FACTOR ** 3)
    print("Fine voxel size:", fine_voxel_size)
    print("Fine ray step:", step_size_fine)

    for _, point_data in tqdm(points.items(), desc="Adaptive ray rasterization"):
        P = point_data["xyz"]
        for image_id in set(point_data["image_ids"]):
            if image_id not in images:
                continue
            C = images[image_id]["camera_center"]
            ray = P - C
            ray_length = np.linalg.norm(ray)
            if ray_length <= 1e-8:
                continue
            num_steps = int(np.ceil(ray_length / step_size_fine))
            if num_steps <= 1:
                continue
            t_values = np.arange(num_steps, dtype=np.float32) / num_steps
            samples = C[None, :] + t_values[:, None] * ray[None, :]
            parent_ijk = np.floor((samples - bound_min) / voxel_size).astype(np.int32)
            inside = ((parent_ijk[:, 0] >= 0) & (parent_ijk[:, 0] < NX) &
                      (parent_ijk[:, 1] >= 0) & (parent_ijk[:, 1] < NY) &
                      (parent_ijk[:, 2] >= 0) & (parent_ijk[:, 2] < NZ))
            parent_ijk = parent_ijk[inside]
            samples = samples[inside]
            if len(parent_ijk) == 0:
                continue
            keep = np.array([tuple(v) in refine_parent_set for v in parent_ijk])
            if not np.any(keep):
                continue
            samples = samples[keep]
            child_ijk = np.floor((samples - bound_min) / fine_voxel_size).astype(np.int32)
            valid_child = ((child_ijk[:, 0] >= 0) & (child_ijk[:, 0] < fine_shape[0]) &
                           (child_ijk[:, 1] >= 0) & (child_ijk[:, 1] < fine_shape[1]) &
                           (child_ijk[:, 2] >= 0) & (child_ijk[:, 2] < fine_shape[2]))
            child_ijk = child_ijk[valid_child]
            if len(child_ijk) == 0:
                continue
            child_ijk = np.unique(child_ijk, axis=0)
            for idx in child_ijk:
                key = tuple(idx)
                child_ray_count[key] = child_ray_count.get(key, 0) + 1

    child_indices = []
    child_counts = []
    for idx, count in child_ray_count.items():
        if count >= threshold:
            child_indices.append(idx)
            child_counts.append(count)

    if child_indices:
        child_indices = np.asarray(child_indices, dtype=np.int32)
        child_counts = np.asarray(child_counts, dtype=np.uint16)
    else:
        child_indices = np.empty((0, 3), dtype=np.int32)
        child_counts = np.empty((0,), dtype=np.uint16)

    print("\nChild voxel statistics")
    print("----------------------")
    print("Children with any ray evidence:", len(child_ray_count))
    print("Children passing threshold:", len(child_indices))
    if len(child_counts) > 0:
        print("Maximum child ray count:", child_counts.max())
        print("Mean surviving child ray count:", child_counts.mean())
        print("Median surviving child ray count:", np.median(child_counts))

    return refine_parent_voxels, child_indices, child_counts, fine_voxel_size

# ============================================================
# 9. FINE CHILD DISTANCE FIELD
# ============================================================
def compute_child_distances(child_indices, refine_parent_voxels, colmap_xyz, bound_min, fine_voxel_size, refine_factor, grid_shape):
    if len(child_indices) == 0:
        return np.zeros(0, dtype=np.float32)

    child_distance = np.zeros(len(child_indices), dtype=np.float32)
    child_lookup = {tuple(idx): i for i, idx in enumerate(child_indices)}

    # Group surviving free-space children by their coarse parent.
    child_parent_map = {}
    for idx in child_indices:
        parent_idx = tuple((idx // refine_factor).astype(np.int32))
        child_parent_map.setdefault(parent_idx, []).append(idx)

    # Group fine-resolution COLMAP surface voxels by coarse parent.
    surface_child = np.floor((colmap_xyz - bound_min) / fine_voxel_size).astype(np.int32)
    fine_shape = np.asarray(grid_shape, dtype=np.int32) * refine_factor
    valid = np.all((surface_child >= 0) & (surface_child < fine_shape), axis=1)
    surface_child = surface_child[valid]
    surface_child = np.unique(surface_child, axis=0)
    surface_parent_map = {}
    for idx in surface_child:
        parent_idx = tuple((idx // refine_factor).astype(np.int32))
        surface_parent_map.setdefault(parent_idx, []).append(idx)

    grid_shape = np.asarray(grid_shape, dtype=np.int32)

    for parent in tqdm(refine_parent_voxels, desc="Computing fine child distances"):
        pmin = np.maximum(np.asarray(parent) - LOCAL_RADIUS, 0)
        pmax = np.minimum(np.asarray(parent) + LOCAL_RADIUS + 1, grid_shape)
        local_parent_shape = pmax - pmin
        local_fine_shape = local_parent_shape * refine_factor

        # Unknown child cells are NOT treated as surfaces. Only actual
        # fine-resolution COLMAP surface cells form the EDT boundary.
        local_surface = np.zeros(tuple(local_fine_shape), dtype=bool)

        for px in range(pmin[0], pmax[0]):
            for py in range(pmin[1], pmax[1]):
                for pz in range(pmin[2], pmax[2]):
                    for idx in surface_parent_map.get((px, py, pz), []):
                        lx = idx[0] - pmin[0] * refine_factor
                        ly = idx[1] - pmin[1] * refine_factor
                        lz = idx[2] - pmin[2] * refine_factor
                        if (0 <= lx < local_fine_shape[0] and 0 <= ly < local_fine_shape[1] and 0 <= lz < local_fine_shape[2]):
                            local_surface[lx, ly, lz] = True

        # Distance to the nearest surface voxel in world units.
        local_distance = distance_transform_edt(~local_surface, sampling=fine_voxel_size).astype(np.float32)

        for idx in child_parent_map.get(tuple(parent), []):
            lx = idx[0] - pmin[0] * refine_factor
            ly = idx[1] - pmin[1] * refine_factor
            lz = idx[2] - pmin[2] * refine_factor
            if (0 <= lx < local_fine_shape[0] and 0 <= ly < local_fine_shape[1] and 0 <= lz < local_fine_shape[2]):
                child_id = child_lookup[tuple(idx)]
                child_distance[child_id] = local_distance[lx, ly, lz]

    return child_distance

# ============================================================
# 10. FIELD GENERATION PIPELINE
# ============================================================
def build_free_space_field():
    print("Loading COLMAP points...")
    points = read_points3d_with_tracks(POINTS_PATH)
    points_xyz = np.array([p["xyz"] for p in points.values()], dtype=np.float64)
    print("COLMAP points:", len(points_xyz))
    print("Point array shape:", points_xyz.shape)

    print("\nLoading registered cameras...")
    images = read_images_binary(IMAGES_PATH)
    print("Registered images:", len(images))

    print("\nComputing bounds...")
    bound_min, bound_max, grid_resolution, voxel_size = compute_bounds(points_xyz)

    print("\nGenerating free-space rays...")
    ray_count = generate_ray_count(points, images, bound_min, bound_max, grid_resolution, voxel_size)

    print("\nLoading COLMAP surface points...")
    ply = PlyData.read(COLMAP_PLY)
    vertices = ply["vertex"]
    colmap_xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]])
    print("PLY points:", len(colmap_xyz))

    print("\nBuilding masks...")
    free_mask, surface_mask, unknown_mask, voxel_indices, voxel_size = build_masks(ray_count, colmap_xyz, bound_min, bound_max)

    print("\nComputing distance field...")
    distance_world, confidence = compute_distance_and_confidence(ray_count, free_mask, voxel_size)

    if NEIGHBOR_RADIUS is not None:
        print("\nGenerating adaptive refinement...")
        refine_parent_voxels, child_indices, child_ray_count, fine_voxel_size = generate_adaptive_children(
            points, images, colmap_xyz, bound_min, grid_resolution, voxel_size, FREE_SPACE_THRESHOLD
        )

        print("\nComputing fine child distances...")
        child_distance = compute_child_distances(
            child_indices, refine_parent_voxels, colmap_xyz, bound_min,s
            fine_voxel_size, REFINE_FACTOR, grid_resolution
        )
    else:
        refine_parent_voxels = np.empty((0, 3), dtype=np.int32)
        child_indices = np.empty((0, 3), dtype=np.int32)
        child_ray_count = np.empty((0,), dtype=np.uint16)
        child_distance = np.empty((0,), dtype=np.float32)
        fine_voxel_size = np.zeros(3, dtype=np.float64)
    print("Maximum child distance:", child_distance.max() if len(child_distance) > 0 else 0.0)
    print("Mean child distance:", child_distance.mean() if len(child_distance) > 0 else 0.0)

    np.savez_compressed(
        OUTPUT_PATH,
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
        refine_parent_voxels=refine_parent_voxels,
        child_indices=child_indices,
        child_ray_count=child_ray_count,
        child_distance=child_distance,
        fine_voxel_size=fine_voxel_size,
        neighbor_radius=NEIGHBOR_RADIUS,
        refine_factor=REFINE_FACTOR
    )

    total_voxels = free_mask.size
    print("\nSaved:", OUTPUT_PATH)
    print("\n========== FIELD STATISTICS ==========")
    print("Grid shape:", ray_count.shape)
    print("Total voxels:", total_voxels)
    print(f"Free voxels: {free_mask.sum():,} ({free_mask.mean() * 100:.6f}%)")
    print(f"COLMAP surface voxels: {surface_mask.sum():,} ({surface_mask.mean() * 100:.6f}%)")
    print(f"Unknown voxels: {unknown_mask.sum():,} ({unknown_mask.mean() * 100:.6f}%)")
    print("COLMAP points:", len(colmap_xyz))
    print("COLMAP points inside grid:", len(voxel_indices))
    print("COLMAP points in free voxels:", int(free_mask[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]].sum()))
    print("Maximum distance:", distance_world.max())

    nonzero = ray_count > 0
    print("Mean nonzero ray count:", ray_count[nonzero].mean() if np.any(nonzero) else 0)

    print("\n========== ADAPTIVE FIELD ==========")
    print("Neighbor radius:", NEIGHBOR_RADIUS)
    print("Refine factor:", REFINE_FACTOR)
    print("Refined parent voxels:", len(refine_parent_voxels))
    print("Surviving child voxels:", len(child_indices))
    print("Child distance shape:", child_distance.shape)
    print("Maximum child distance:", child_distance.max() if len(child_distance) > 0 else 0.0)
    print("Mean child distance:", child_distance.mean() if len(child_distance) > 0 else 0.0)

if __name__ == "__main__":
    build_free_space_field()
