import os

# Must be set before importing pyrender — selects the EGL backend so PyRender
# uses the GPU on headless Linux machines instead of falling back to OSMesa.
os.environ["PYOPENGL_PLATFORM"] = "egl"
import csv
import math
import random
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyrender
import trimesh

csv.field_size_limit(sys.maxsize)


@dataclass
class Config:
    # --- PATHS ---
    gso_root = Path("./datasets/gso")
    replica_root = Path("./datasets/Replica")
    out_root = Path("./datasets/synth_train")

    debug_visualize = False  # True opens an interactive cv2.imshow preview per patch
                             # (blocking, needs a display) and does NOT write data.
    export_3d_scenes = False

    width = 512
    height = 512

    # --- Distinct Patch Sizes ---
    patch_template = 128
    patch_search = 192

    yfov_deg_min = 45.0
    yfov_deg_max = 75.0
    znear = 0.1
    zfar = 25.0

    # --- V7: AGGRESSIVE ZOOM AUGMENTATION ---
    zoom_prob = 0.7  # Increased to 70% to force scale invariance
    zoom_scale_min = 0.4  # View can be 2.5x farther away
    zoom_scale_max = 3.0  # View can be 3.0x closer

    # --- DATASET TARGETS ---
    target_train = 350_000
    target_val = 30_000
    seed = 42

    # --- SCENE PARAMS ---
    gso_per_scene_min = 40
    gso_per_scene_max = 60
    scale_min = 0.85
    scale_max = 1.20
    max_point_attempts = 2000
    patches_per_camera = 3

    # --- FILTERS ---
    min_texture_variance = 50.0
    min_local_variance = 30.0
    local_window_size = 24

    template_suffix = "_template.jpg"
    search_suffix = "_search.jpg"
    jpeg_quality = 95
    depth_eps = 0.02

    # --- REPLICA SPECIFIC PARAMS ---
    replica_max_vert_candidates = 100_000
    replica_cam_height_min = 1.0
    replica_cam_height_max = 1.8
    replica_fov_coverage_min = 0.4
    max_cam_fails = 200


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


def list_model_objs(split_root):
    if not split_root.exists(): return []
    out = sorted(list(split_root.rglob("model.obj")))
    return out


def load_obj_trimesh(path):
    try:
        m = trimesh.load(str(path), force='mesh', process=True)
        return m
    except Exception:
        return None


def unit_scale_trimesh(mesh):
    v = mesh.vertices.astype(np.float32)
    mn = v.min(axis=0)
    mx = v.max(axis=0)
    c = (mn + mx) * 0.5
    v = v - c
    s = float(np.max(mx - mn))
    if s < 1e-9: s = 1.0
    mesh2 = mesh.copy()
    mesh2.vertices = (v / s).astype(np.float32)
    return mesh2


def rot_from_euler(roll, pitch, yaw):
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def make_pose(R, t):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return T


def camera_pose_look_at(yaw_deg, pitch_deg, dist, target):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    eye = np.array([
        dist * math.cos(pitch) * math.sin(yaw),
        dist * math.sin(pitch),
        dist * math.cos(pitch) * math.cos(yaw),
    ], dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    f = target - eye
    norm_f = np.linalg.norm(f)
    f = f / (norm_f + 1e-9)
    r = np.cross(f, up)
    norm_r = np.linalg.norm(r)
    if norm_r < 1e-9:
        r = np.array([1, 0, 0], dtype=np.float32)
    else:
        r = r / norm_r
    u = np.cross(r, f)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = r
    pose[:3, 1] = u
    pose[:3, 2] = -f
    pose[:3, 3] = eye
    return pose.astype(np.float32)


def camera_pose_at_position(pos, yaw_deg, pitch_deg, roll_deg=0.0):
    """V7: Added Roll support for extreme in-plane skew."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    forward = np.array([
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
        math.cos(pitch) * math.cos(yaw),
    ], dtype=np.float32)
    eye = np.array(pos, dtype=np.float32)
    target = eye + forward
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-9)
    r = np.cross(f, up)
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-9:
        r = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        r = r / r_norm
    u = np.cross(r, f)

    # Apply camera roll
    new_u = u * math.cos(roll) + r * math.sin(roll)
    new_r = r * math.cos(roll) - u * math.sin(roll)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = new_r
    pose[:3, 1] = new_u
    pose[:3, 2] = -f
    pose[:3, 3] = eye
    return pose


def project_point_px(world_pt, view, proj, width, height):
    p4 = np.ones((4,), dtype=np.float32)
    p4[:3] = world_pt
    clip = proj @ (view @ p4)
    w = float(clip[3])
    if abs(w) < 1e-9: return None
    ndc = clip[:3] / w
    if ndc[2] < -1.0 or ndc[2] > 1.0: return None
    x = (ndc[0] * 0.5 + 0.5) * width
    y = (1.0 - (ndc[1] * 0.5 + 0.5)) * height
    return float(x), float(y)


def crop_patch_rgb(img_rgb, center_xy, patch):
    cx, cy = center_xy
    x0 = int(round(cx - patch / 2))
    y0 = int(round(cy - patch / 2))
    x1 = x0 + patch
    y1 = y0 + patch
    h, w = img_rgb.shape[:2]
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    if pad_l or pad_t or pad_r or pad_b:
        img_rgb = cv2.copyMakeBorder(img_rgb, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
        x0 += pad_l
        y0 += pad_t
        x1 = x0 + patch
        y1 = y0 + patch
    return img_rgb[y0:y1, x0:x1].copy()


def sample_offset(rng):
    """V7: Expanded to handle 32px massive displacements."""
    r = rng.random()
    if r < 0.30:
        return rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
    elif r < 0.60:
        lo, hi = 1.0, 5.0
    elif r < 0.85:
        lo, hi = 5.0, 15.0
    elif r < 0.95:
        lo, hi = 15.0, 25.0
    else:
        lo, hi = 25.0, 32.0

    dx = rng.uniform(-hi, hi)
    dy = rng.uniform(-hi, hi)

    if lo > 0:
        for _ in range(10):
            if abs(dx) >= lo or abs(dy) >= lo: break
            dx = rng.uniform(-hi, hi)
            dy = rng.uniform(-hi, hi)

    return float(dx), float(dy)


def write_jpeg(path, rgb, quality):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])


def random_light_color(rng: random.Random) -> np.ndarray:
    r = rng.random()
    if r < 0.4:
        return np.array([1.0, rng.uniform(0.8, 0.95), rng.uniform(0.6, 0.85)])
    elif r < 0.8:
        return np.array([rng.uniform(0.75, 0.9), rng.uniform(0.85, 0.95), 1.0])
    else:
        return np.ones(3)


def setup_lighting(scene, rng, bounds, harsh=False):
    """V7: Dynamic scene relighting to simulate flash/shadows."""
    for node in list(scene.light_nodes):
        scene.remove_node(node)

    x_min, x_max, floor_y, ceiling_y, z_min, z_max = bounds

    if harsh:
        dir_light = pyrender.DirectionalLight(color=random_light_color(rng), intensity=rng.uniform(5.0, 10.0))
        dl_pose = camera_pose_look_at(rng.uniform(-180, 180), rng.uniform(-80, -10), 2.0, [0, 0, 0])
        scene.add(dir_light, pose=dl_pose)
    else:
        for _ in range(rng.randint(2, 4)):
            lpos = np.array([
                rng.uniform(x_min, x_max),
                floor_y + rng.uniform(2.0, ceiling_y),
                rng.uniform(z_min, z_max),
            ], dtype=np.float32)
            light = pyrender.PointLight(color=random_light_color(rng), intensity=rng.uniform(8.0, 25.0))
            lpose = np.eye(4, dtype=np.float32)
            lpose[:3, 3] = lpos
            scene.add(light, pose=lpose)

        dir_light = pyrender.DirectionalLight(color=random_light_color(rng), intensity=rng.uniform(1.0, 3.0))
        dl_pose = camera_pose_look_at(rng.uniform(-180, 180), rng.uniform(-60, -20), 2.0, [0, 0, 0])
        scene.add(dir_light, pose=dl_pose)


def apply_photometric_augmentation(patch_rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    img = patch_rgb.copy()
    if rng.random() < 0.5:
        alpha = rng.uniform(0.7, 1.3)
        beta = rng.uniform(-20, 20)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        sigma = rng.uniform(3.0, 15.0)
        noise = np.random.randn(*img.shape).astype(np.float32) * sigma
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        ksize = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    if rng.random() < 0.5:
        quality = rng.randint(70, 95)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img


def is_occluded(world_pt, view, depth_map, u, v, eps):
    h, w = depth_map.shape[:2]
    ix = int(round(u))
    iy = int(round(v))
    if ix < 0 or iy < 0 or ix >= w or iy >= h: return 1
    pc = view @ np.array([world_pt[0], world_pt[1], world_pt[2], 1.0], dtype=np.float32)
    d_true = float(-pc[2])
    if d_true <= 0: return 1
    patch = depth_map[max(0, iy - 1):min(h, iy + 2), max(0, ix - 1):min(w, ix + 2)]
    valid = patch[(np.isfinite(patch)) & (patch > 0)]
    if valid.size == 0: return 1
    return 1 if (d_true - valid.min()) > float(eps) else 0


def split_models_disjoint(cfg):
    train_dir = cfg.gso_root / "train"
    test_dir = cfg.gso_root / "val"
    if train_dir.exists() and test_dir.exists():
        return list_model_objs(train_dir), list_model_objs(test_dir)

    all_models = list_model_objs(cfg.gso_root)
    unique_models = sorted(list(set(all_models)))

    rng = random.Random(cfg.seed)
    rng.shuffle(unique_models)
    n_train = max(1, int(0.9 * len(unique_models)))
    return unique_models[:n_train], unique_models[n_train:]


def list_replica_scenes(replica_root):
    scenes = []
    for d in sorted(replica_root.iterdir()):
        if d.is_dir() and (d / "mesh.ply").exists():
            scenes.append(d)
    return scenes


def gen_split_replica(cfg, split_name, target_total, scene_dirs, model_paths):
    out_split = cfg.out_root / split_name
    ensure_dir(out_split)
    labels_path = out_split / "labels.csv"

    scenes_dir = cfg.out_root / "3dscenes"
    if cfg.export_3d_scenes: ensure_dir(scenes_dir)

    written = 0
    file_mode = "w"
    if labels_path.exists():
        with open(labels_path, "r") as f:
            row_count = sum(1 for _ in f)
        if row_count > 1:
            written = row_count - 1
            file_mode = "a"

    initial_written = int(written)

    if written >= target_total:
        print(f"[{split_name.upper()}] Already at {written} >= {target_total}, skipping.")
        return

    renderer = pyrender.OffscreenRenderer(cfg.width, cfg.height)
    rng = random.Random(cfg.seed + (500 if split_name == "train" else 999500))
    np.random.seed(cfg.seed + (500 if split_name == "train" else 999500))

    margin_t = cfg.patch_template // 2
    margin_s = cfg.patch_search // 2
    start_time = time.time()
    last_print = start_time

    with open(labels_path, file_mode, newline="", encoding="utf-8") as fcsv:
        csv_w = csv.writer(fcsv)
        if file_mode == "w":
            csv_w.writerow(["filename_template", "filename_search",
                            "dx", "dy", "dyaw", "dpitch", "droll", "occluded",
                            "zoom_target", "zoom_fov_deg"])

        for scene_i, scene_dir in enumerate(scene_dirs):
            if written >= target_total: break

            remaining_scenes = len(scene_dirs) - scene_i
            remaining_pairs = target_total - written
            pairs_this_scene = math.ceil(remaining_pairs / remaining_scenes)
            scene_target = min(written + pairs_this_scene, target_total)

            print(f"\n[{split_name.upper()}] Scene {scene_i + 1}/{len(scene_dirs)}: "
                  f"{scene_dir.name} (target: {pairs_this_scene} pairs)")

            try:
                tm = trimesh.load(str(scene_dir / "mesh.ply"), process=False)
                if isinstance(tm, trimesh.Scene): tm = tm.dump(concatenate=True)
            except Exception as e:
                print(f"  Failed to load: {e}")
                continue

            verts = tm.vertices.astype(np.float32)
            bbox_min, bbox_max = verts.min(0), verts.max(0)
            floor_y = float(bbox_min[1])
            ceiling_y = float(bbox_max[1])
            room_height = ceiling_y - floor_y

            pad_x = min(0.3, (bbox_max[0] - bbox_min[0]) / 6.0)
            pad_z = min(0.3, (bbox_max[2] - bbox_min[2]) / 6.0)
            x_min = float(bbox_min[0] + pad_x)
            x_max = float(bbox_max[0] - pad_x)
            z_min = float(bbox_min[2] + pad_z)
            z_max = float(bbox_max[2] - pad_z)

            if len(verts) > cfg.replica_max_vert_candidates:
                vert_idx = rng.sample(range(len(verts)), cfg.replica_max_vert_candidates)
                candidate_verts = verts[vert_idx]
            else:
                candidate_verts = verts

            export_scene = trimesh.Scene()
            export_scene.add_geometry(tm, transform=np.eye(4), node_name="room")
            pm = pyrender.Mesh.from_trimesh(tm, smooth=False)
            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0], ambient_light=[0.15, 0.15, 0.15, 1.0])

            gso_verts_world = []
            num_gso = rng.randint(cfg.gso_per_scene_min, cfg.gso_per_scene_max)
            chosen_gso = [model_paths[rng.randrange(len(model_paths))] for _ in range(num_gso)]

            for idx, obj_path in enumerate(chosen_gso):
                m = load_obj_trimesh(obj_path)
                if m is None: continue
                gso_tm = unit_scale_trimesh(m)
                gso_pm = pyrender.Mesh.from_trimesh(gso_tm, smooth=True)

                sc = rng.uniform(cfg.scale_min, cfg.scale_max) * 0.4
                R = rot_from_euler(
                    rng.uniform(-3.14, 3.14),
                    rng.uniform(-3.14, 3.14),
                    rng.uniform(-3.14, 3.14)
                ) * sc

                max_obj_y = max(0.2, room_height - 0.1)
                t = np.array([
                    rng.uniform(x_min, x_max),
                    floor_y + rng.uniform(0.05, max_obj_y),
                    rng.uniform(z_min, z_max)
                ], dtype=np.float32)

                pose = make_pose(R, t)
                scene.add(gso_pm, pose=pose)
                export_scene.add_geometry(gso_tm, transform=pose, node_name=f"gso_{idx}")

                v = gso_tm.vertices.astype(np.float32)
                v_world = (v @ pose[:3, :3].T) + pose[:3, 3][None, :]
                gso_verts_world.append(v_world)

            if cfg.export_3d_scenes:
                scene_path = scenes_dir / f"{split_name}_scene_{scene_i:05d}.glb"
                if not scene_path.exists():
                    export_scene.export(str(scene_path))
                    print(f"  Exported 3D Scene to: {scene_path}")

            all_candidates = [candidate_verts] + gso_verts_world
            candidate_verts = np.concatenate(all_candidates, axis=0)
            scene.add(pm)

            scene_fov_deg = rng.uniform(cfg.yfov_deg_min, cfg.yfov_deg_max)
            camera = pyrender.PerspectiveCamera(
                yfov=np.deg2rad(scene_fov_deg), aspectRatio=1.0,
                znear=cfg.znear, zfar=cfg.zfar)
            cam_node = scene.add(camera, pose=np.eye(4))

            # Bundle bounds for lighting helper
            bounds = (x_min, x_max, floor_y, min(2.8, room_height), z_min, z_max)

            consecutive_cam_fails = 0

            while written < scene_target:
                if consecutive_cam_fails >= cfg.max_cam_fails:
                    print(f"\n  Exhausted camera positions for {scene_dir.name}, moving on.")
                    break

                cam_ok = False
                for _ in range(50):
                    cam_x = rng.uniform(x_min, x_max)
                    cam_z = rng.uniform(z_min, z_max)
                    cam_y = floor_y + rng.uniform(cfg.replica_cam_height_min, cfg.replica_cam_height_max)
                    yaw0 = rng.uniform(-180, 180)
                    pitch0 = rng.uniform(-30, 30)

                    cam0 = camera_pose_at_position([cam_x, cam_y, cam_z], yaw0, pitch0)
                    scene.set_pose(cam_node, pose=cam0)
                    _, depth_check = renderer.render(scene)
                    valid = (depth_check > 0.4) & (depth_check < 10.0)
                    if valid.sum() / depth_check.size > cfg.replica_fov_coverage_min:
                        cam_ok = True
                        break

                if not cam_ok:
                    consecutive_cam_fails += 1
                    continue
                consecutive_cam_fails = 0

                # --- V7: Extreme Viewpoint Perturbation ---
                r_cam = rng.random()
                sign_y = 1 if rng.random() < 0.5 else -1
                sign_p = 1 if rng.random() < 0.5 else -1

                if r_cam < 0.35:
                    dyaw = rng.uniform(0, 5) * sign_y
                    dpitch = rng.uniform(0, 5) * sign_p
                elif r_cam < 0.70:
                    dyaw = rng.uniform(6, 15) * sign_y
                    dpitch = rng.uniform(6, 15) * sign_p
                elif r_cam < 0.90:
                    dyaw = rng.uniform(16, 30) * sign_y  # Moderate Skew
                    dpitch = rng.uniform(16, 30) * sign_p
                else:
                    dyaw = rng.uniform(31, 55) * sign_y  # Extreme Grazing Angle!
                    dpitch = rng.uniform(31, 55) * sign_p

                # Added Roll
                droll = 0.0
                if rng.random() < 0.4:
                    droll = rng.uniform(-15, 15)
                elif rng.random() < 0.1:
                    droll = rng.uniform(-45, 45)

                cam1_x = max(x_min, min(x_max, cam_x + rng.uniform(-0.3, 0.3)))
                cam1_y = max(floor_y + cfg.replica_cam_height_min,
                             min(floor_y + cfg.replica_cam_height_max, cam_y + rng.uniform(-0.1, 0.1)))
                cam1_z = max(z_min, min(z_max, cam_z + rng.uniform(-0.3, 0.3)))

                cam1 = camera_pose_at_position([cam1_x, cam1_y, cam1_z], yaw0 + dyaw, pitch0 + dpitch, droll)
                render_flags = pyrender.RenderFlags.RGBA

                zoom_target = None
                zoom_fov_deg = scene_fov_deg
                if rng.random() < cfg.zoom_prob:
                    zoom_scale = rng.uniform(cfg.zoom_scale_min, cfg.zoom_scale_max)
                    zoomed_rad = 2.0 * math.atan(math.tan(math.radians(scene_fov_deg) / 2.0) / zoom_scale)
                    zoom_fov_deg = max(20.0, min(120.0, math.degrees(zoomed_rad)))
                    zoom_target = 'template' if rng.random() < 0.5 else 'search'

                fov_t_deg = zoom_fov_deg if zoom_target == 'template' else scene_fov_deg
                fov_s_deg = zoom_fov_deg if zoom_target == 'search' else scene_fov_deg

                def render_view(pose, fov_deg):
                    nonlocal cam_node
                    if abs(fov_deg - scene_fov_deg) > 0.5:
                        cam_z = pyrender.PerspectiveCamera(yfov=np.deg2rad(fov_deg), aspectRatio=1.0, znear=cfg.znear,
                                                           zfar=cfg.zfar)
                        scene.remove_node(cam_node)
                        tmp_node = scene.add(cam_z, pose=pose)
                        rgba, depth = renderer.render(scene, flags=render_flags)
                        proj = cam_z.get_projection_matrix(cfg.width, cfg.height)
                        scene.remove_node(tmp_node)
                        cam_node = scene.add(camera, pose=pose)
                    else:
                        scene.set_pose(cam_node, pose=pose)
                        rgba, depth = renderer.render(scene, flags=render_flags)
                        proj = camera.get_projection_matrix(cfg.width, cfg.height)
                    return rgba, depth, proj

                # --- Render Template (Standard Lighting) ---
                setup_lighting(scene, rng, bounds, harsh=False)
                rgba0, depth0, proj0 = render_view(cam0, fov_t_deg)
                view0 = np.linalg.inv(cam0)

                # --- Render Search (Relit 50% of the time) ---
                if rng.random() < 0.5:
                    setup_lighting(scene, rng, bounds, harsh=(rng.random() < 0.5))
                rgba1, depth1, proj1 = render_view(cam1, fov_s_deg)
                view1 = np.linalg.inv(cam1)

                img0 = rgba0[..., :3].copy()
                img1 = rgba1[..., :3].copy()

                patches_from_cam = 0
                attempts = 0
                while (
                        patches_from_cam < cfg.patches_per_camera and attempts < cfg.max_point_attempts and written < scene_target):
                    attempts += 1

                    vidx = rng.randrange(len(candidate_verts))
                    wpt = candidate_verts[vidx]

                    p0 = project_point_px(wpt, view0, proj0, cfg.width, cfg.height)
                    if not p0: continue
                    u0, v0 = p0
                    if not (margin_t <= u0 < cfg.width - margin_t and margin_t <= v0 < cfg.height - margin_t): continue
                    if is_occluded(wpt, view0, depth0, u0, v0, cfg.depth_eps): continue

                    patch_t_rgb = crop_patch_rgb(img0, (u0, v0), cfg.patch_template)

                    black_pixels = np.sum(patch_t_rgb, axis=-1) < 5
                    if np.mean(black_pixels) > 0.02: continue

                    gray = cv2.cvtColor(patch_t_rgb, cv2.COLOR_RGB2GRAY)
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if laplacian_var < cfg.min_texture_variance: continue

                    c_half = cfg.patch_template // 2
                    r_half = cfg.local_window_size // 2
                    local_patch = gray[c_half - r_half: c_half + r_half, c_half - r_half: c_half + r_half]
                    local_var = cv2.Laplacian(local_patch, cv2.CV_64F).var()
                    if local_var < cfg.min_local_variance: continue

                    p1 = project_point_px(wpt, view1, proj1, cfg.width, cfg.height)
                    if not p1: continue
                    u1, v1 = p1
                    if not (0 <= u1 < cfg.width and 0 <= v1 < cfg.height): continue
                    if is_occluded(wpt, view1, depth1, u1, v1, cfg.depth_eps): continue

                    dx_off, dy_off = sample_offset(rng)
                    cx1, cy1 = u1 + dx_off, v1 + dy_off
                    if not (
                            margin_s <= cx1 < cfg.width - margin_s and margin_s <= cy1 < cfg.height - margin_s): continue

                    patch_s_rgb = crop_patch_rgb(img1, (cx1, cy1), cfg.patch_search)

                    if cfg.debug_visualize:
                        patch_t_bgr = cv2.cvtColor(patch_t_rgb, cv2.COLOR_RGB2BGR)
                        patch_s_bgr = cv2.cvtColor(patch_s_rgb, cv2.COLOR_RGB2BGR)

                        pad_debug = (cfg.patch_search - cfg.patch_template) // 2
                        patch_t_bgr_padded = cv2.copyMakeBorder(patch_t_bgr, pad_debug, pad_debug, pad_debug, pad_debug,
                                                                cv2.BORDER_CONSTANT, value=[40, 40, 40])

                        cv2.drawMarker(patch_t_bgr_padded, (cfg.patch_search // 2, cfg.patch_search // 2), (0, 255, 0),
                                       cv2.MARKER_CROSS, 12, 2)
                        target_x = int(round(cfg.patch_search / 2 - dx_off))
                        target_y = int(round(cfg.patch_search / 2 - dy_off))
                        cv2.drawMarker(patch_s_bgr, (target_x, target_y), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
                        cv2.drawMarker(patch_s_bgr, (cfg.patch_search // 2, cfg.patch_search // 2), (255, 100, 0),
                                       cv2.MARKER_TILTED_CROSS, 10, 2)

                        combo = np.hstack((patch_t_bgr_padded, patch_s_bgr))
                        combo = cv2.resize(combo, (384 * 2, 192 * 2), interpolation=cv2.INTER_NEAREST)

                        # cv2.putText(combo, f"Yaw: {dyaw:.1f} | Pitch: {dpitch:.1f} | Roll: {droll:.1f}", (10, 30),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        # cv2.putText(combo, f"Var(G): {laplacian_var:.1f} | Var(L): {local_var:.1f}", (10, 50),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        # zoom_info = f"Zoom->{zoom_target} FOV {scene_fov_deg:.0f}deg->{zoom_fov_deg:.0f}deg" if zoom_target else f"No zoom (FOV {scene_fov_deg:.0f}deg)"
                        # cv2.putText(combo, zoom_info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                        # cv2.putText(combo,
                        #             f"dx={-dx_off:.1f} dy={-dy_off:.1f} | Cam [{patches_from_cam + 1}/{cfg.patches_per_camera}]",
                        #             (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

                        cv2.imshow("Template (Left) vs Search (Right)", combo)
                        if cv2.waitKey(0) == 27:
                            print("\nExiting debug mode...")
                            sys.exit(0)

                    if not cfg.debug_visualize:
                        stem = f"{written:07d}"
                        ref_name = f"{stem}{cfg.template_suffix}"
                        trg_name = f"{stem}{cfg.search_suffix}"
                        write_jpeg(out_split / ref_name, patch_t_rgb, cfg.jpeg_quality)
                        write_jpeg(out_split / trg_name, patch_s_rgb, cfg.jpeg_quality)
                        csv_w.writerow([ref_name, trg_name, float(u1 - cx1), float(v1 - cy1),
                                        dyaw, dpitch, droll, 0, zoom_target or "none", round(zoom_fov_deg, 1)])

                    written += 1
                    patches_from_cam += 1

                    if time.time() - last_print > 1.0:
                        rate = (written - initial_written) / ((time.time() - start_time) + 1e-9)
                        print(
                            f"\r[{split_name.upper()}] {scene_dir.name} | {written}/{target_total} | {rate:.1f} pair/s",
                            end="")
                        last_print = time.time()

            scene.clear()
            del scene, tm, pm, export_scene
            gc.collect()

    print(f"\nDone {split_name}! Total entries: {written}")
    renderer.delete()


def main():
    cfg = Config()
    ensure_dir(cfg.out_root)
    models_train, models_test = split_models_disjoint(cfg)

    if cfg.replica_root.exists():
        all_replica = list_replica_scenes(cfg.replica_root)
        if all_replica:
            n_train = max(1, int(0.8 * len(all_replica)))
            gen_split_replica(cfg, "train", cfg.target_train, all_replica[:n_train], models_train)
            gen_split_replica(cfg, "val", cfg.target_val, all_replica[n_train:], models_test)
        else:
            print("No Replica scenes found inside the directory — skipping.")
    else:
        print(f"Replica root not found ({cfg.replica_root}) — skipping.")


if __name__ == "__main__":
    main()