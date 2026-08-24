import json
import os
import os.path as osp

import cv2
import numpy as np

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


def _load_scene(scene_root, num_views):
    colmap_dir = osp.join(scene_root, "colmap")
    images_dir = osp.join(colmap_dir, "images_8")
    transforms_path = osp.join(colmap_dir, "transforms.json")
    with open(transforms_path, encoding="utf-8") as file:
        metadata = json.load(file)

    image_names = set(os.listdir(images_dir))
    frames = []
    for frame in metadata["frames"]:
        basename = osp.basename(frame["file_path"])
        if basename in image_names:
            frames.append(
                {
                    "image_path": osp.join(images_dir, basename),
                    "camera_pose": np.asarray(
                        frame["transform_matrix"], dtype=np.float32
                    ),
                }
            )
    if len(frames) < num_views:
        raise ValueError(f"Not enough images in DL3DV sample scene: {scene_root}")

    first = imread_cv2(frames[0]["image_path"], cv2.IMREAD_COLOR)
    height, width = first.shape[:2]
    scale_x = width / float(metadata["w"])
    scale_y = height / float(metadata["h"])
    intrinsics = np.array(
        [
            [metadata["fl_x"] * scale_x, 0, metadata["cx"] * scale_x],
            [0, metadata["fl_y"] * scale_y, metadata["cy"] * scale_y],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    opengl_to_opencv = np.diag([1, -1, -1, 1]).astype(np.float32)
    for frame in frames:
        frame["camera_pose"] = frame["camera_pose"] @ opengl_to_opencv

    return {
        "root": scene_root,
        "frames": frames,
        "intrinsics": intrinsics,
    }


def _load_direct_images8_scene(scene_root, num_views):
    """Load a DL3DV scene laid out as scene/transforms.json + scene/images_8."""
    images_dir = osp.join(scene_root, "images_8")
    transforms_path = osp.join(scene_root, "transforms.json")
    with open(transforms_path, encoding="utf-8") as file:
        metadata = json.load(file)

    image_names = set(os.listdir(images_dir))
    image_stems = {}
    for image_name in sorted(image_names):
        stem = osp.splitext(image_name)[0]
        image_stems.setdefault(stem, image_name)

    frames = []
    for frame in metadata["frames"]:
        basename = osp.basename(frame["file_path"])
        image_name = basename
        if image_name not in image_names:
            image_name = image_stems.get(osp.splitext(basename)[0])
        if image_name is not None:
            frames.append(
                {
                    "image_path": osp.join(images_dir, image_name),
                    "camera_pose": np.asarray(
                        frame["transform_matrix"], dtype=np.float32
                    ),
                }
            )
    if len(frames) < num_views:
        raise ValueError(
            "Not enough matched images in DL3DV images_8 scene: "
            f"{scene_root} matched={len(frames)} "
            f"frames={len(metadata.get('frames', []))} images={len(image_names)}"
        )

    first = imread_cv2(frames[0]["image_path"], cv2.IMREAD_COLOR)
    height, width = first.shape[:2]
    scale_x = width / float(metadata["w"])
    scale_y = height / float(metadata["h"])
    intrinsics = np.array(
        [
            [metadata["fl_x"] * scale_x, 0, metadata["cx"] * scale_x],
            [0, metadata["fl_y"] * scale_y, metadata["cy"] * scale_y],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    opengl_to_opencv = np.diag([1, -1, -1, 1]).astype(np.float32)
    for frame in frames:
        frame["camera_pose"] = frame["camera_pose"] @ opengl_to_opencv

    return {
        "root": scene_root,
        "frames": frames,
        "intrinsics": intrinsics,
    }


def _get_scene_views(dataset, scene, start, resolution, rng, num_views):
    frames = scene["frames"]
    available = len(frames) - start
    max_interval = min(
        dataset.max_interval,
        max(1, (available - 1) // max(num_views - 1, 1)),
    )
    interval = int(rng.integers(1, max_interval + 1))
    image_ids = [start + index * interval for index in range(num_views)]

    views = []
    for image_id in image_ids:
        frame = frames[image_id]
        image = imread_cv2(frame["image_path"], cv2.IMREAD_COLOR)
        depth = np.ones(image.shape[:2], dtype=np.float32)
        intrinsics = scene["intrinsics"].copy()
        image, depth, intrinsics = dataset._crop_resize_if_necessary(
            image,
            depth,
            intrinsics,
            resolution,
            rng=rng,
            info=image_id,
        )
        views.append(
            {
                "img": image,
                "depthmap": depth,
                "camera_pose": frame["camera_pose"].copy(),
                "camera_intrinsics": intrinsics.astype(np.float32),
                "dataset": "dl3dv-sample",
                "label": osp.basename(scene["root"]),
                "instance": frame["image_path"],
                "is_metric": False,
                "is_video": True,
                "quantile": np.array(0.98, dtype=np.float32),
                "img_mask": True,
                "ray_mask": False,
                "camera_only": True,
                "depth_only": False,
                "single_view": False,
                "reset": False,
            }
        )
    return views


def _read_scene_frame(dataset, scene, image_id, resolution, rng):
    frames = scene["frames"]
    frame = frames[image_id]
    image = imread_cv2(frame["image_path"], cv2.IMREAD_COLOR)
    depth = np.ones(image.shape[:2], dtype=np.float32)
    intrinsics = scene["intrinsics"].copy()
    image, depth, intrinsics = dataset._crop_resize_if_necessary(
        image,
        depth,
        intrinsics,
        resolution,
        rng=rng,
        info=image_id,
    )
    return {
        "img": image,
        "depthmap": depth,
        "camera_pose": frame["camera_pose"].copy(),
        "camera_intrinsics": intrinsics.astype(np.float32),
        "dataset": "dl3dv-sample",
        "label": osp.basename(scene["root"]),
        "instance": frame["image_path"],
        "is_metric": False,
        "is_video": True,
        "quantile": np.array(0.98, dtype=np.float32),
        "img_mask": True,
        "ray_mask": False,
        "camera_only": True,
        "depth_only": False,
        "single_view": False,
        "reset": False,
    }


class DL3DVSample_Multi(BaseMultiViewDataset):
    """Load one raw DL3DV sample scene from colmap/images_8."""

    def __init__(self, *args, ROOT, max_interval=8, **kwargs):
        self.ROOT = ROOT
        self.max_interval = max_interval
        self.video = True
        self.is_metric = False
        super().__init__(*args, **kwargs)
        self._load_scene()

    def _load_scene(self):
        self.scene = _load_scene(self.ROOT, self.num_views)
        self.frames = self.scene["frames"]
        self.intrinsics = self.scene["intrinsics"]
        self.start_ids = np.arange(len(self.frames) - self.num_views + 1)

    def __len__(self):
        return len(self.start_ids)

    def get_image_num(self):
        return len(self.frames)

    def _get_views(self, idx, resolution, rng, num_views):
        start = int(self.start_ids[idx])
        return _get_scene_views(
            self, self.scene, start, resolution, rng, num_views
        )


class DL3DVSampleCollection_Multi(BaseMultiViewDataset):
    """Load every raw DL3DV sample scene below ROOT as one training set."""

    def __init__(self, *args, ROOT, max_interval=8, **kwargs):
        self.ROOT = ROOT
        self.max_interval = max_interval
        self.video = True
        self.is_metric = False
        super().__init__(*args, **kwargs)
        self._load_scenes()

    def _load_scenes(self):
        scene_roots = []
        for name in sorted(os.listdir(self.ROOT)):
            scene_root = osp.join(self.ROOT, name)
            if not osp.isdir(scene_root) or name.startswith("."):
                continue
            transforms_path = osp.join(scene_root, "colmap", "transforms.json")
            images_dir = osp.join(scene_root, "colmap", "images_8")
            if osp.isfile(transforms_path) and osp.isdir(images_dir):
                scene_roots.append(scene_root)
        if not scene_roots:
            raise ValueError(f"No DL3DV sample scenes found below {self.ROOT}")

        self.scenes = [
            _load_scene(scene_root, self.num_views)
            for scene_root in scene_roots
        ]
        self.sample_index = [
            (scene_id, start)
            for scene_id, scene in enumerate(self.scenes)
            for start in range(len(scene["frames"]) - self.num_views + 1)
        ]

    def __len__(self):
        return len(self.sample_index)

    def get_image_num(self):
        return sum(len(scene["frames"]) for scene in self.scenes)

    def get_stats(self):
        return (
            f"{len(self.sample_index)} groups from {len(self.scenes)} scenes "
            f"and {self.get_image_num()} images"
        )

    def _get_views(self, idx, resolution, rng, num_views):
        scene_id, start = self.sample_index[idx]
        return _get_scene_views(
            self,
            self.scenes[scene_id],
            start,
            resolution,
            rng,
            num_views,
        )


class DL3DVImages8_Multi(BaseMultiViewDataset):
    """Load DL3DV folders under ROOT/*/<scene> with images_8 + transforms.json."""

    def __init__(
        self,
        *args,
        ROOT,
        max_interval=8,
        max_scenes=0,
        skip_invalid=True,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.max_interval = max_interval
        self.max_scenes = max_scenes
        self.skip_invalid = skip_invalid
        self.video = True
        self.is_metric = False
        super().__init__(*args, **kwargs)
        self._load_scenes()

    @staticmethod
    def _is_scene_dir(path):
        return (
            osp.isfile(osp.join(path, "transforms.json"))
            and osp.isdir(osp.join(path, "images_8"))
        )

    def _find_scene_roots(self):
        if self._is_scene_dir(self.ROOT):
            return [self.ROOT]

        scene_roots = []
        for name in sorted(os.listdir(self.ROOT)):
            first_level = osp.join(self.ROOT, name)
            if not osp.isdir(first_level) or name.startswith("."):
                continue
            if self._is_scene_dir(first_level):
                scene_roots.append(first_level)
                continue
            for child_name in sorted(os.listdir(first_level)):
                scene_root = osp.join(first_level, child_name)
                if (
                    osp.isdir(scene_root)
                    and not child_name.startswith(".")
                    and self._is_scene_dir(scene_root)
                ):
                    scene_roots.append(scene_root)
        return scene_roots

    def _load_scenes(self):
        scene_roots = self._find_scene_roots()
        if not scene_roots:
            raise ValueError(
                f"No DL3DV images_8 scenes found below {self.ROOT}"
            )

        self.scenes = []
        self.skipped_scenes = []
        rank = int(os.environ.get("RANK", "0"))
        for scene_root in scene_roots:
            try:
                self.scenes.append(
                    _load_direct_images8_scene(scene_root, self.num_views)
                )
            except (OSError, KeyError, ValueError) as error:
                if not self.skip_invalid:
                    raise
                self.skipped_scenes.append((scene_root, str(error)))
                if rank == 0 and len(self.skipped_scenes) <= 20:
                    print(f"Skipping invalid DL3DV images_8 scene: {error}")
            if self.max_scenes and len(self.scenes) >= self.max_scenes:
                break
        if not self.scenes:
            raise ValueError(f"No valid DL3DV images_8 scenes found below {self.ROOT}")
        if rank == 0 and len(self.skipped_scenes) > 20:
            print(
                "Skipped additional invalid DL3DV images_8 scenes: "
                f"{len(self.skipped_scenes) - 20}"
            )

        self.sample_index = [
            (scene_id, start)
            for scene_id, scene in enumerate(self.scenes)
            for start in range(len(scene["frames"]) - self.num_views + 1)
        ]

    def __len__(self):
        return len(self.sample_index)

    def get_image_num(self):
        return sum(len(scene["frames"]) for scene in self.scenes)

    def get_stats(self):
        skipped = (
            f", skipped {len(self.skipped_scenes)} invalid scenes"
            if self.skipped_scenes
            else ""
        )
        return (
            f"{len(self.sample_index)} groups from {len(self.scenes)} scenes "
            f"and {self.get_image_num()} images{skipped}"
        )

    def _get_views(self, idx, resolution, rng, num_views):
        scene_id, start = self.sample_index[idx]
        scene = self.scenes[scene_id]
        frames = scene["frames"]
        available = len(frames) - start
        max_interval = min(
            self.max_interval,
            max(1, (available - 1) // max(num_views - 1, 1)),
        )
        interval = int(rng.integers(1, max_interval + 1))
        image_ids = [start + index * interval for index in range(num_views)]

        views = []
        used_ids = set()
        for image_id in image_ids:
            view = self._read_valid_view(
                scene,
                image_id,
                resolution,
                rng,
                used_ids,
            )
            views.append(view)
            used_ids.add(int(view["idx_in_scene"]))
            del view["idx_in_scene"]
        return views

    def _read_valid_view(self, scene, image_id, resolution, rng, used_ids):
        frames = scene["frames"]
        candidates = sorted(
            range(len(frames)),
            key=lambda candidate: (abs(candidate - image_id), candidate),
        )

        last_error = None
        for candidate in candidates:
            if candidate in used_ids:
                continue
            try:
                view = _read_scene_frame(self, scene, candidate, resolution, rng)
                view["idx_in_scene"] = candidate
                return view
            except OSError as error:
                last_error = error

        raise OSError(
            f"No readable replacement frame in {scene['root']} near {image_id}. "
            f"Last error: {last_error}"
        )
