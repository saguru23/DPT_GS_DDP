import glob
import os
import os.path as osp

import cv2
import numpy as np

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


def _frame_id(path):
    stem = osp.splitext(osp.basename(path))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits)


def _read_replica_pose(line):
    c2w = np.asarray(list(map(float, line.split())), dtype=np.float32).reshape(4, 4)
    c2w[:3, 1] *= -1
    c2w[:3, 2] *= -1
    return c2w


def _load_scene(scene_root, image_dir, intrinsics, depth_scale, num_views):
    results_dir = osp.join(scene_root, image_dir)
    pose_path = osp.join(scene_root, "traj.txt")
    if not osp.isdir(results_dir) or not osp.isfile(pose_path):
        raise FileNotFoundError(f"Invalid Replica scene: {scene_root}")

    image_paths = sorted(
        glob.glob(osp.join(results_dir, "frame*.jpg"))
        + glob.glob(osp.join(results_dir, "frame*.png")),
        key=_frame_id,
    )
    depth_paths = {
        _frame_id(path): path
        for path in glob.glob(osp.join(results_dir, "depth*.png"))
    }
    with open(pose_path, encoding="utf-8") as file:
        pose_lines = file.readlines()

    frames = []
    for image_path in image_paths:
        frame_id = _frame_id(image_path)
        depth_path = depth_paths.get(frame_id)
        if depth_path is None or frame_id >= len(pose_lines):
            continue
        frames.append(
            {
                "image_path": image_path,
                "depth_path": depth_path,
                "camera_pose": _read_replica_pose(pose_lines[frame_id]),
            }
        )
    if len(frames) < num_views:
        raise ValueError(f"Not enough Replica frames in {scene_root}")

    return {
        "root": scene_root,
        "name": osp.basename(scene_root),
        "frames": frames,
        "intrinsics": intrinsics.astype(np.float32),
        "depth_scale": float(depth_scale),
    }


class Replica_Multi(BaseMultiViewDataset):
    """NICE-SLAM Replica loader with RGB, metric depth, poses, and intrinsics."""

    def __init__(
        self,
        *args,
        ROOT,
        image_dir="results",
        max_interval=8,
        depth_scale=6553.5,
        fx=600.0,
        fy=600.0,
        cx=599.5,
        cy=339.5,
        scene_names=None,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.image_dir = image_dir
        self.max_interval = max_interval
        self.depth_scale = depth_scale
        self.intrinsics = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.scene_names = scene_names
        self.video = True
        self.is_metric = True
        super().__init__(*args, **kwargs)
        self._load_data()

    def _scene_roots(self):
        if osp.isfile(osp.join(self.ROOT, "traj.txt")):
            return [self.ROOT]

        if self.scene_names is None:
            names = sorted(
                name
                for name in os.listdir(self.ROOT)
                if osp.isfile(osp.join(self.ROOT, name, "traj.txt"))
            )
        elif isinstance(self.scene_names, str):
            names = [name.strip() for name in self.scene_names.split(",") if name.strip()]
        else:
            names = list(self.scene_names)
        return [osp.join(self.ROOT, name) for name in names]

    def _load_data(self):
        self.scenes = []
        self.start_ids = []
        for scene_root in self._scene_roots():
            scene = _load_scene(
                scene_root,
                self.image_dir,
                self.intrinsics,
                self.depth_scale,
                self.num_views,
            )
            scene_id = len(self.scenes)
            self.scenes.append(scene)
            cut_off = self.num_views if not self.allow_repeat else max(
                self.num_views // 3,
                3,
            )
            for start in range(len(scene["frames"]) - cut_off + 1):
                self.start_ids.append((scene_id, start))
        if not self.start_ids:
            raise ValueError(f"No usable Replica sequences under {self.ROOT}")

    def __len__(self):
        return len(self.start_ids)

    def get_image_num(self):
        return sum(len(scene["frames"]) for scene in self.scenes)

    def _sample_ids(self, scene, start, rng, num_views):
        frames = scene["frames"]
        available = len(frames) - start
        max_interval = min(
            self.max_interval,
            max(1, (available - 1) // max(num_views - 1, 1)),
        )
        interval = int(rng.integers(1, max_interval + 1))
        return [start + i * interval for i in range(num_views)]

    def _get_views(self, idx, resolution, rng, num_views):
        scene_id, start = self.start_ids[idx]
        scene = self.scenes[scene_id]
        image_ids = self._sample_ids(scene, start, rng, num_views)

        views = []
        for view_index, image_id in enumerate(image_ids):
            frame = scene["frames"][image_id]
            image = imread_cv2(frame["image_path"], cv2.IMREAD_COLOR)
            depth = imread_cv2(frame["depth_path"], cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32) / scene["depth_scale"]
            depth[~np.isfinite(depth)] = 0.0
            intrinsics = scene["intrinsics"].copy()
            image, depth, intrinsics = self._crop_resize_if_necessary(
                image,
                depth,
                intrinsics,
                resolution,
                rng=rng,
                info=frame["image_path"],
            )
            views.append(
                {
                    "img": image,
                    "depthmap": depth.astype(np.float32),
                    "camera_pose": frame["camera_pose"].copy(),
                    "camera_intrinsics": intrinsics.astype(np.float32),
                    "dataset": "replica",
                    "label": scene["name"],
                    "instance": frame["image_path"],
                    "is_metric": True,
                    "is_video": True,
                    "quantile": np.array(1.0, dtype=np.float32),
                    "img_mask": True,
                    "ray_mask": False,
                    "camera_only": False,
                    "depth_only": False,
                    "single_view": False,
                    "reset": False,
                }
            )
        assert len(views) == num_views
        return views
