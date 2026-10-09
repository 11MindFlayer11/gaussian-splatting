import os
import struct
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from plyfile import PlyData
from scipy.ndimage import binary_dilation

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
parser.add_argument("--local_radius", type=int, default=1)   # unused now (kept for CLI compatibility)
parser.add_argument("--surface_margin", type=int, default=0)
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
SURFACE_MARGIN = args.surface_margin

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
# 4. VECTORIZED RAY HELPERS
# ============================================================
def build_rays(points, images, dedupe=False):
    """Flatten every (3D point, observing camera) pair into arrays: camera centers, directions, lengths."""
    P, C = [], []
    for p in points.values():
        ids = set(p["image_ids"]) if dedupe else p["image_ids"]
        for i in ids:
            if i in images:
                P.append(p["xyz"])
                C.append(images[i]["camera_center"])
    if len(P) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0,))
    P, C = np.asarray(P), np.asarray(C)
    d = P - C
    return C, d, np.linalg.norm(d, axis=1)

def iter_ray_samples(C, d, L, step, chunk=2000):
    """
    Yield (ray_id, sample_xyz) for chunks of rays. Samples are t = k/n for k in [0, n),
    n = ceil(L/step), identical to the original per-ray np.arange(n)/n. ray_id is local to the chunk.
    """
    n = np.ceil(L / step).astype(np.int64)
    ok = np.where((L > 1e-8) & (n > 1))[0]
    for s in range(0, len(ok), chunk):
        r = ok[s:s + chunk]
        nr = n[r]
        rid = np.repeat(np.arange(len(r)), nr)
        k = np.arange(nr.sum()) - np.repeat(np.cumsum(nr) - nr, nr)
        t = k / nr[rid]
        yield rid, C[r][rid] + t[:, None] * d[r][rid]

# ============================================================
# 5. COARSE FREE-SPACE RAY GENERATION
# ============================================================
def generate_ray_count(points, images, bound_min, bound_max, grid_resolution, voxel_size):
    grid = np.asarray(grid_resolution, dtype=np.int64)
    V = int(grid.prod())
    C, d, L = build_rays(points, images)
    step = voxel_size.min() * RAY_STEP_FACTOR
    ray_count = np.zeros(V, dtype=np.int64)
    valid_rays = 0

    print("\nGrid:", tuple(grid))
    print("Voxel size:", voxel_size)
    print("Step size:", step)
    print("Rays:", len(L))

    for rid, samples in iter_ray_samples(C, d, L, step):
        ijk = np.floor((samples - bound_min) / voxel_size).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < grid), axis=1)
        ijk, rid = ijk[inside], rid[inside]
        if len(ijk) == 0:
            continue
        lin = (ijk[:, 0] * grid[1] + ijk[:, 1]) * grid[2] + ijk[:, 2]
        key = np.unique(rid * V + lin)              # one hit per (ray, voxel)
        valid_rays += len(np.unique(key // V))
        u, c = np.unique(key % V, return_counts=True)
        ray_count[u] += c

    ray_count = ray_count.reshape(tuple(grid)).astype(np.uint16)

    print("\nRay generation complete.")
    print("Total rays:", int((L >= 1e-8).sum()))
    print("Valid rays:", valid_rays)
    print("Nonzero voxels:", np.count_nonzero(ray_count))
    print("Maximum ray count:", ray_count.max())
    return ray_count

# ============================================================
# 6. SURFACE / FREE / UNKNOWN CLASSIFICATION
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
# 7. COARSE DISTANCE FIELD / CONFIDENCE
# ============================================================
def compute_distance_and_confidence(ray_count, free_mask, voxel_size):
    distance_world = distance_transform_edt(free_mask, sampling=voxel_size).astype(np.float32)
    denominator = CONFIDENCE_SATURATION - FREE_SPACE_THRESHOLD
    confidence = np.clip((ray_count.astype(np.float32) - FREE_SPACE_THRESHOLD) / denominator, 0.0, 1.0)
    confidence *= free_mask
    return distance_world, confidence

# ============================================================
# 8. ADAPTIVE PARENT SELECTION
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
# 9. ADAPTIVE CHILD RAY EVIDENCE
# ============================================================
def generate_adaptive_children(points, images, colmap_xyz, bound_min, grid_resolution, voxel_size, threshold):
    grid = np.asarray(grid_resolution, dtype=np.int64)
    point_ijk = np.floor((colmap_xyz - bound_min) / voxel_size).astype(np.int64)
    point_ijk = np.clip(point_ijk, 0, grid - 1)
    surface_parent_voxels = np.unique(point_ijk, axis=0)
    refine_parent_voxels = expand_voxel_neighborhood(surface_parent_voxels, NEIGHBOR_RADIUS, tuple(grid))

    # boolean mask replaces the python set of tuples
    refine_mask = np.zeros(tuple(grid), dtype=bool)
    refine_mask[tuple(refine_parent_voxels.T)] = True

    fine_voxel_size = voxel_size / REFINE_FACTOR
    step = fine_voxel_size.min() * RAY_STEP_FACTOR
    fine = grid * REFINE_FACTOR
    FV = int(fine.prod())
    C, d, L = build_rays(points, images, dedupe=True)

    print("\nAdaptive refinement")
    print("-------------------")
    print("Surface parent voxels:", len(surface_parent_voxels))
    print("Refined parent voxels:", len(refine_parent_voxels))
    print("Neighbor radius:", NEIGHBOR_RADIUS)
    print("Children per parent:", REFINE_FACTOR ** 3)
    print("Fine voxel size:", fine_voxel_size)
    print("Fine ray step:", step)
    print("Rays:", len(L))

    all_lin, all_cnt = [], []
    for rid, samples in iter_ray_samples(C, d, L, step):
        parent = np.floor((samples - bound_min) / voxel_size).astype(np.int64)
        inside = np.all((parent >= 0) & (parent < grid), axis=1)
        parent, samples, rid = parent[inside], samples[inside], rid[inside]
        if len(parent) == 0:
            continue
        keep = refine_mask[parent[:, 0], parent[:, 1], parent[:, 2]]
        samples, rid = samples[keep], rid[keep]
        if len(samples) == 0:
            continue

        child = np.floor((samples - bound_min) / fine_voxel_size).astype(np.int64)
        ok = np.all((child >= 0) & (child < fine), axis=1)
        child, rid = child[ok], rid[ok]
        if len(child) == 0:
            continue
        lin = (child[:, 0] * fine[1] + child[:, 1]) * fine[2] + child[:, 2]
        key = np.unique(rid * FV + lin)              # one hit per (ray, child voxel)
        u, c = np.unique(key % FV, return_counts=True)
        all_lin.append(u)
        all_cnt.append(c)

    if all_lin:
        all_lin = np.concatenate(all_lin)
        all_cnt = np.concatenate(all_cnt)
        u, inv = np.unique(all_lin, return_inverse=True)
        counts = np.bincount(inv.ravel(), weights=all_cnt).astype(np.int64)
        n_any = len(u)
        m = counts >= threshold
        u, counts = u[m], counts[m]
        child_indices = np.stack(np.unravel_index(u, tuple(int(x) for x in fine)), axis=1).astype(np.int32)
        child_counts = counts.astype(np.uint16)
    else:
        n_any = 0
        child_indices = np.empty((0, 3), dtype=np.int32)
        child_counts = np.empty((0,), dtype=np.uint16)

    print("\nChild voxel statistics")
    print("----------------------")
    print("Children with any ray evidence:", n_any)
    print("Children passing threshold:", len(child_indices))
    if len(child_counts) > 0:
        print("Maximum child ray count:", child_counts.max())
        print("Mean surviving child ray count:", child_counts.mean())
        print("Median surviving child ray count:", np.median(child_counts))

    return refine_parent_voxels, child_indices, child_counts, fine_voxel_size

# ============================================================
# 10. FINE CHILD DISTANCE FIELD
# ============================================================
def compute_child_distances(child_indices, refine_parent_voxels, colmap_xyz, bound_min,
                            fine_voxel_size, refine_factor, grid_shape):
    """
    Distance from each surviving child voxel to the nearest fine-resolution COLMAP surface voxel,
    in world units. One KD-tree query replaces the per-parent EDT loop. Unlike the windowed EDT,
    this is not limited to a (2*LOCAL_RADIUS+1)^3 parent window.
    """
    if len(child_indices) == 0:
        return np.zeros(0, dtype=np.float32)

    fine = np.asarray(grid_shape, dtype=np.int64) * refine_factor
    s = np.floor((colmap_xyz - bound_min) / fine_voxel_size).astype(np.int64)
    s = np.unique(s[np.all((s >= 0) & (s < fine), axis=1)], axis=0)
    if len(s) == 0:
        return np.zeros(len(child_indices), dtype=np.float32)

    tree = cKDTree(s * fine_voxel_size)
    dist, _ = tree.query(child_indices * fine_voxel_size, workers=-1)
    return dist.astype(np.float32)

# ============================================================
# 11. FIELD GENERATION PIPELINE
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
    free_mask, surface_mask, unknown_mask, voxel_indices, voxel_size = build_masks(
        ray_count, colmap_xyz, bound_min, bound_max
    )
    
    margin_mask = np.zeros_like(surface_mask)
    if SURFACE_MARGIN > 0:
        margin_mask = binary_dilation(surface_mask, structure=np.ones((3, 3, 3), dtype=bool),
                                      iterations=SURFACE_MARGIN)
        free_mask &= ~margin_mask
        unknown_mask = ~(free_mask | surface_mask)
        print("Surface margin:", SURFACE_MARGIN, "| voxels removed from free space:", int(margin_mask.sum() - surface_mask.sum()))

    print("\nComputing distance field...")
    distance_world, confidence = compute_distance_and_confidence(ray_count, free_mask, voxel_size)

    if NEIGHBOR_RADIUS is not None:
        print("\nGenerating adaptive refinement...")
        refine_parent_voxels, child_indices, child_ray_count, fine_voxel_size = generate_adaptive_children(
            points, images, colmap_xyz, bound_min, grid_resolution, voxel_size, FREE_SPACE_THRESHOLD
        )

        print("\nComputing fine child distances...")
        child_distance = compute_child_distances(
            child_indices, refine_parent_voxels, colmap_xyz, bound_min,
            fine_voxel_size, REFINE_FACTOR, grid_resolution
        )
    else:
        refine_parent_voxels = np.empty((0, 3), dtype=np.int32)
        child_indices = np.empty((0, 3), dtype=np.int32)
        child_ray_count = np.empty((0,), dtype=np.uint16)
        child_distance = np.empty((0,), dtype=np.float32)
        fine_voxel_size = np.zeros(3, dtype=np.float64)

    if SURFACE_MARGIN > 0 and len(child_indices) > 0:
        parent = child_indices // REFINE_FACTOR
        child_distance[margin_mask[parent[:, 0], parent[:, 1], parent[:, 2]]] = 0.0
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