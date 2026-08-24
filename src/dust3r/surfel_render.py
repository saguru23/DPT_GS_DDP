import torch
import torch.nn as nn
import torch.nn.functional as F

from dust3r.post_process import estimate_focal_knowing_depth


def _standardize_quaternion(quaternion):
    quaternion = F.normalize(quaternion, dim=-1, eps=1e-8)
    return torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)


def _quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quaternion_to_matrix(quaternion):
    quaternion = _standardize_quaternion(quaternion)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y.square() + z.square()),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x.square() + z.square()),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def _pose_encoding_to_camera(pose):
    camera = torch.eye(4, device=pose.device, dtype=pose.dtype)
    camera = camera.expand(pose.shape[0], -1, -1).clone()
    camera[:, :3, :3] = _quaternion_to_matrix(pose[:, 3:7])
    camera[:, :3, 3] = pose[:, :3]
    return camera


def _matrix_to_quaternion(matrix):
    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    q_abs = torch.sqrt(
        torch.stack(
            (
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ),
            dim=-1,
        ).clamp_min(0.0)
    )
    quat_by_component = torch.stack(
        (
            torch.stack(
                (q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01),
                dim=-1,
            ),
            torch.stack(
                (m21 - m12, q_abs[..., 1].square(), m01 + m10, m02 + m20),
                dim=-1,
            ),
            torch.stack(
                (m02 - m20, m01 + m10, q_abs[..., 2].square(), m12 + m21),
                dim=-1,
            ),
            torch.stack(
                (m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3].square()),
                dim=-1,
            ),
        ),
        dim=-2,
    )
    quat_candidates = quat_by_component / (
        2.0 * q_abs[..., None].clamp_min(1e-8)
    )
    best = q_abs.argmax(dim=-1)
    quaternion = quat_candidates[
        F.one_hot(best, num_classes=4).to(dtype=torch.bool)
    ].reshape(*best.shape, 4)
    return _standardize_quaternion(quaternion)


def _quaternion_from_normal(normal):
    valid_normal = normal.norm(dim=-1, keepdim=True) > 0.5
    fallback = torch.zeros_like(normal)
    fallback[..., 2] = -1.0
    z_axis = F.normalize(
        torch.where(valid_normal, normal, fallback), dim=-1, eps=1e-8
    )

    ref = torch.zeros_like(z_axis)
    ref[..., 0] = 1.0
    alt_ref = torch.zeros_like(z_axis)
    alt_ref[..., 1] = 1.0
    ref = torch.where(z_axis[..., :1].abs() > 0.99, alt_ref, ref)

    x_axis = ref - (ref * z_axis).sum(dim=-1, keepdim=True) * z_axis
    x_axis = F.normalize(x_axis, dim=-1, eps=1e-8)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=1e-8)
    rotation = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    return _matrix_to_quaternion(rotation)


def pointmap_normals(points):
    if points.shape[1] < 3 or points.shape[2] < 3:
        return torch.zeros_like(points)

    valid = torch.isfinite(points).all(dim=-1) & (points[..., 2] > 1e-4)
    points = torch.where(valid[..., None], points, torch.zeros_like(points))
    mask = valid[..., None].float()

    point_up = torch.cat([points[:, :1], points[:, :-1]], dim=1)
    point_left = torch.cat([points[:, :, :1], points[:, :, :-1]], dim=2)
    point_bottom = torch.cat([points[:, 1:], points[:, -1:]], dim=1)
    point_right = torch.cat([points[:, :, 1:], points[:, :, -1:]], dim=2)
    mask_up = torch.cat([mask[:, :1], mask[:, :-1]], dim=1)
    mask_left = torch.cat([mask[:, :, :1], mask[:, :, :-1]], dim=2)
    mask_bottom = torch.cat([mask[:, 1:], mask[:, -1:]], dim=1)
    mask_right = torch.cat([mask[:, :, 1:], mask[:, :, -1:]], dim=2)

    up = (point_up - points) * mask_up
    left = (point_left - points) * mask_left
    bottom = (point_bottom - points) * mask_bottom
    right = (point_right - points) * mask_right
    normal = (
        torch.cross(up, left, dim=-1)
        + torch.cross(right, up, dim=-1)
        + torch.cross(bottom, right, dim=-1)
        + torch.cross(left, bottom, dim=-1)
    )
    normal = F.normalize(normal, dim=-1, eps=1e-8)
    normal = torch.where(valid[..., None], normal, torch.zeros_like(normal))
    normal = torch.where(normal[..., 2:3] > 0, -normal, normal)
    return normal


def _predicted_intrinsics(pred):
    points = pred["pts3d_in_self_view"].detach()
    batch_size, height, width, _ = points.shape
    pp = points.new_tensor([width / 2.0, height / 2.0]).expand(batch_size, -1)
    focal = estimate_focal_knowing_depth(
        points,
        pp,
        focal_mode="weiszfeld",
        min_focal=0.1,
        max_focal=10.0,
    ).detach()
    default_focal = points.new_full(
        (batch_size,), max(height, width) / 1.154700538
    )
    focal = torch.where(
        torch.isfinite(focal) & (focal > 0),
        focal,
        default_focal,
    )
    intrinsics = torch.eye(3, device=points.device, dtype=points.dtype)
    intrinsics = intrinsics[None].repeat(batch_size, 1, 1)
    intrinsics[:, 0, 0] = focal
    intrinsics[:, 1, 1] = focal
    intrinsics[:, 0, 2] = pp[:, 0]
    intrinsics[:, 1, 2] = pp[:, 1]
    return intrinsics


def _geometry_mask(pred, stride):
    points = pred["pts3d_in_self_view"].detach()
    mask = torch.isfinite(points).all(dim=-1) & (points[..., 2] > 1e-4)
    mask = mask[:, None].float()
    if stride > 1:
        height, width = points.shape[1:3]
        mask = F.interpolate(
            mask,
            size=(
                (height + stride - 1) // stride,
                (width + stride - 1) // stride,
            ),
            mode="area",
        )
    return mask.permute(0, 2, 3, 1)


def _batch_bool(value, batch_index, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if torch.is_tensor(value):
        if value.numel() == 1:
            return bool(value.item())
        return bool(value.reshape(-1)[batch_index].item())
    try:
        return bool(value[batch_index])
    except (TypeError, IndexError):
        return bool(value)


def _gt_depthmap(gt):
    depth = gt.get("depthmap")
    if depth is None:
        return None
    depth = depth.detach()
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    return depth


def _gt_depth_is_available(gt, batch_index, source):
    if source == "predicted":
        return False
    depth = _gt_depthmap(gt)
    if depth is None:
        if source == "ground_truth":
            raise KeyError("depth_target_source='ground_truth' needs gt depthmap")
        return False
    if source == "auto":
        if _batch_bool(gt.get("camera_only"), batch_index, default=False):
            return False
        if not _batch_bool(gt.get("is_metric"), batch_index, default=True):
            return False
    depth_i = depth[batch_index]
    valid = torch.isfinite(depth_i) & (depth_i > 1e-6)
    if source == "ground_truth" and not valid.any():
        raise ValueError("No valid gt depth pixels for ground_truth depth loss")
    return bool(valid.float().mean() > 0.01)


def _gt_camera_points(gt):
    depth = _gt_depthmap(gt)
    if depth is None:
        return None
    intrinsics = gt["camera_intrinsics"].detach()
    return _camera_points_from_depth(depth, intrinsics)


def _camera_points_from_depth(depth, intrinsics):
    batch_size, height, width = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    xs = xs[None].expand(batch_size, -1, -1)
    ys = ys[None].expand(batch_size, -1, -1)
    z = depth
    x = (xs - intrinsics[:, None, None, 0, 2]) / intrinsics[
        :, None, None, 0, 0
    ].clamp_min(1e-8) * z
    y = (ys - intrinsics[:, None, None, 1, 2]) / intrinsics[
        :, None, None, 1, 1
    ].clamp_min(1e-8) * z
    points = torch.stack((x, y, z), dim=-1)
    valid = torch.isfinite(points).all(dim=-1) & (z > 1e-6)
    return torch.where(valid[..., None], points, torch.zeros_like(points))


def _normal_from_depth(depth, intrinsics):
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    points = _camera_points_from_depth(depth, intrinsics)
    return pointmap_normals(points)


def _render_inputs(gt, stride, intrinsics=None):
    image = (gt["img"] + 1) * 0.5
    height, width = image.shape[-2:]
    if intrinsics is None:
        intrinsics = gt["camera_intrinsics"]
    if stride == 1:
        return (
            image.permute(0, 2, 3, 1),
            intrinsics,
            height,
            width,
        )

    render_height = (height + stride - 1) // stride
    render_width = (width + stride - 1) // stride
    target = F.interpolate(
        image,
        size=(render_height, render_width),
        mode="area",
    ).permute(0, 2, 3, 1)

    scale_x = render_width / width
    scale_y = render_height / height
    intrinsics = intrinsics.clone()
    intrinsics[:, 0, 0] *= scale_x
    intrinsics[:, 1, 1] *= scale_y
    intrinsics[:, 0, 2] = (intrinsics[:, 0, 2] + 0.5) * scale_x - 0.5
    intrinsics[:, 1, 2] = (intrinsics[:, 1, 2] + 0.5) * scale_y - 0.5
    return target, intrinsics, render_height, render_width


def _gaussian_2d_rotations(pred, batch_index, stride=1):
    selection = (batch_index, slice(None, None, stride), slice(None, None, stride))
    rotations = pred.get("gaussian_2d_rotations")
    if rotations is None:
        rotations = pred["gaussian_2d_delta_rotations"]
    return _standardize_quaternion(rotations[selection])


def sample_surfels(
    pred,
    batch_index,
    stride=1,
    scale_multiplier=1.0,
):
    selection = (batch_index, slice(None, None, stride), slice(None, None, stride))
    means = pred["pts3d_in_self_view"].detach()[selection]
    colors = pred["gaussian_2d_rgb"][selection]
    opacities = pred["gaussian_2d_opacity"][selection]
    scales = pred["gaussian_2d_scales"][selection] * scale_multiplier
    rotations = _gaussian_2d_rotations(
        pred,
        batch_index,
        stride=stride,
    )
    valid = (
        torch.isfinite(means).all(dim=-1)
        & torch.isfinite(colors).all(dim=-1)
        & torch.isfinite(opacities).squeeze(-1)
        & torch.isfinite(scales).all(dim=-1)
        & torch.isfinite(rotations).all(dim=-1)
        & (means[..., 2] > 1e-4)
    )
    return (
        means[valid],
        colors[valid],
        opacities[valid],
        scales[valid],
        rotations[valid],
    )


def _render_surfels_gsplat(
    means,
    colors,
    opacities,
    scales,
    rotations,
    intrinsics,
    height,
    width,
    viewmat=None,
):
    from gsplat import rasterization_2dgs  # noqa: WPS433

    if means.numel() == 0:
        raise RuntimeError("No valid 2D Gaussians remain after filtering")
    if means.device.type != "cuda":
        raise RuntimeError("2DGS rasterization requires CUDA tensors")

    scale_z = torch.ones_like(scales[:, :1])
    scales3d = torch.cat([scales, scale_z], dim=-1)
    if viewmat is None:
        viewmat = torch.eye(4, device=means.device, dtype=torch.float32)
    render_colors, alpha, normal, _, dist, median_depth, meta = (
        rasterization_2dgs(
            means=means.float()[None],
            quats=_standardize_quaternion(rotations.float())[None],
            scales=scales3d.float().clamp_min(1e-6)[None],
            opacities=opacities.squeeze(-1).float().clamp(1e-4, 1 - 1e-4)[
                None
            ],
            colors=colors.float().clamp(0, 1)[None, None],
            viewmats=viewmat.float()[None, None],
            Ks=intrinsics.float()[None, None],
            width=width,
            height=height,
            packed=False,
            render_mode="RGB+ED",
            distloss=False,
            depth_mode="expected",
        )
    )
    rgbd = render_colors[0, 0]
    return {
        "rgb": rgbd[..., :3].permute(2, 0, 1),
        "radii": meta["radii"],
        "alpha": alpha[0, 0].permute(2, 0, 1),
        "depth": rgbd[..., 3:].permute(2, 0, 1),
        "normal": normal[0, 0].permute(2, 0, 1),
        "dist": dist[0, 0].permute(2, 0, 1),
        "median_depth": median_depth[0, 0].permute(2, 0, 1),
    }


def render_surfels(
    means,
    colors,
    opacities,
    scales,
    rotations,
    intrinsics,
    height,
    width,
    viewmat=None,
):
    return _render_surfels_gsplat(
        means,
        colors,
        opacities,
        scales,
        rotations,
        intrinsics,
        height,
        width,
        viewmat=viewmat,
    )


def _depth_target(gt, pred, batch_index, stride, use_gt_depth):
    selection = (batch_index, slice(None, None, stride), slice(None, None, stride))
    if use_gt_depth:
        return _gt_depthmap(gt)[selection][..., None]
    return pred["pts3d_in_self_view"].detach()[selection][..., 2:3]


def _normal_target(gt, pred, batch_index, stride, use_gt_depth):
    selection = (batch_index, slice(None, None, stride), slice(None, None, stride))
    if use_gt_depth:
        return pointmap_normals(_gt_camera_points(gt))[selection]
    return pointmap_normals(pred["pts3d_in_self_view"].detach())[selection]


def render_prediction_view(
    gt,
    pred,
    batch_index=0,
):
    intrinsics = _predicted_intrinsics(pred)
    target, intrinsics, height, width = _render_inputs(
        gt, stride=1, intrinsics=intrinsics
    )
    surfels = sample_surfels(
        pred,
        batch_index,
    )
    rendered = render_surfels(
        *surfels,
        intrinsics[batch_index],
        height,
        width,
    )
    return (
        rendered["rgb"].permute(1, 2, 0),
        rendered["depth"].squeeze(0),
        rendered["alpha"].squeeze(0),
        rendered["normal"].permute(1, 2, 0),
        target[batch_index],
    )


class Gaussian2DRenderingLoss(nn.Module):
    def __init__(
        self,
        *,
        render_stride=1,
        rgb_weight=1.0,
        lpips_weight=0.0,
        normal_weight=0.1,
        self_weight=0.5,
        merged_weight=1.0,
        depth_target_source="predicted",
    ):
        super().__init__()
        if render_stride < 1:
            raise ValueError("render_stride must be >= 1")
        if self_weight < 0.0 or merged_weight < 0.0:
            raise ValueError("self_weight and merged_weight must be >= 0")
        if self_weight == 0.0 and merged_weight == 0.0:
            raise ValueError("At least one render loss weight must be > 0")
        if rgb_weight < 0.0 or lpips_weight < 0.0:
            raise ValueError("RGB and LPIPS weights must be >= 0")
        if normal_weight < 0.0:
            raise ValueError("Normal weight must be >= 0")
        if depth_target_source not in {"auto", "ground_truth", "predicted"}:
            raise ValueError(
                "depth_target_source must be 'auto', 'ground_truth', or "
                "'predicted'"
            )
        self.render_stride = render_stride
        self.rgb_weight = rgb_weight
        self.lpips_weight = lpips_weight
        self.normal_weight = normal_weight
        self.self_weight = self_weight
        self.merged_weight = merged_weight
        self.depth_target_source = depth_target_source

        if self.lpips_weight > 0.0:
            try:
                import lpips  # noqa: WPS433
            except ImportError as exc:
                raise ImportError(
                    "Gaussian2DRenderingLoss needs the lpips package when "
                    "lpips_weight > 0. Install lpips in the active TTT3R "
                    "environment, or set --lpips_weight 0."
                ) from exc
            self.lpips_model = lpips.LPIPS(net="alex")
            self.lpips_model.eval()
            for parameter in self.lpips_model.parameters():
                parameter.requires_grad_(False)
        else:
            self.lpips_model = None

    def _lpips_loss(self, rendered, target, mask):
        if self.lpips_model is None:
            return rendered.new_zeros(())
        valid_rendered = rendered * mask + target.detach() * (1.0 - mask)
        rendered_chw = valid_rendered.permute(0, 3, 1, 2).float() * 2.0 - 1.0
        target_chw = target.permute(0, 3, 1, 2).float() * 2.0 - 1.0
        with torch.autocast(device_type=rendered.device.type, enabled=False):
            return self.lpips_model(rendered_chw, target_chw).mean()

    def _image_loss(self, rendered, target, mask):
        mask_rgb = mask.expand_as(rendered)
        mse = (
            (rendered - target).square() * mask_rgb
        ).sum() / mask_rgb.sum().clamp_min(1.0)
        lpips_loss = self._lpips_loss(rendered, target, mask)
        return (
            self.rgb_weight * mse + self.lpips_weight * lpips_loss,
            mse,
            lpips_loss,
        )

    def _normal_loss(self, rendered_normal, target_normal, mask):
        rendered_normal = F.normalize(rendered_normal, dim=-1, eps=1e-8)
        target_normal = F.normalize(target_normal.detach(), dim=-1, eps=1e-8)
        normal_valid = (
            torch.isfinite(rendered_normal).all(dim=-1, keepdim=True)
            & torch.isfinite(target_normal).all(dim=-1, keepdim=True)
            & (rendered_normal.norm(dim=-1, keepdim=True) > 0.5)
            & (target_normal.norm(dim=-1, keepdim=True) > 0.5)
        )
        normal_mask = mask * normal_valid.float()
        cosine = (rendered_normal * target_normal).sum(dim=-1, keepdim=True)
        loss = 1.0 - cosine.abs().clamp(0.0, 1.0)
        return (loss * normal_mask).sum() / normal_mask.sum().clamp_min(1.0)

    def _normal_smooth_loss(self, rendered_normal, mask):
        rendered_normal = F.normalize(rendered_normal, dim=-1, eps=1e-8)
        valid = (
            torch.isfinite(rendered_normal).all(dim=-1, keepdim=True)
            & (rendered_normal.norm(dim=-1, keepdim=True) > 0.5)
        )
        normal_mask = mask * valid.float()

        loss = rendered_normal.new_zeros(())
        if rendered_normal.shape[2] > 1:
            diff_x = (
                rendered_normal[:, :, 1:]
                - rendered_normal[:, :, :-1]
            ).abs().mean(dim=-1, keepdim=True)
            mask_x = normal_mask[:, :, 1:] * normal_mask[:, :, :-1]
            loss = loss + (diff_x * mask_x).sum() / mask_x.sum().clamp_min(1.0)
        if rendered_normal.shape[1] > 1:
            diff_y = (
                rendered_normal[:, 1:]
                - rendered_normal[:, :-1]
            ).abs().mean(dim=-1, keepdim=True)
            mask_y = normal_mask[:, 1:] * normal_mask[:, :-1]
            loss = loss + (diff_y * mask_y).sum() / mask_y.sum().clamp_min(1.0)
        return loss

    def _render_loss(self, rendered, target, target_depth, target_normal, mask):
        rendered_rgb = rendered["rgb"].permute(1, 2, 0)[None]
        loss_rgb, rgb_mse, lpips_loss = self._image_loss(
            rendered_rgb,
            target,
            mask,
        )
        if self.normal_weight > 0.0:
            rendered_normal = rendered["normal"].permute(1, 2, 0)[None]
            normal_consistency = self._normal_loss(
                rendered_normal,
                target_normal,
                mask,
            )
            normal_smooth = self._normal_smooth_loss(rendered_normal, mask)
            normal = normal_consistency + normal_smooth
        else:
            normal_consistency = rgb_mse.new_zeros(())
            normal_smooth = rgb_mse.new_zeros(())
            normal = rgb_mse.new_zeros(())
        loss = loss_rgb + self.normal_weight * normal
        return (
            loss,
            rgb_mse,
            lpips_loss,
            normal,
            normal_consistency,
            normal_smooth,
        )

    def _self_loss(self, gts, preds):
        losses = []
        rgb_mse_losses = []
        lpips_losses = []
        normal_losses = []
        normal_consistency_losses = []
        normal_smooth_losses = []
        alpha_means = []
        gt_depth_flags = []
        batch_size = preds[0]["pts3d_in_self_view"].shape[0]
        for gt, pred in zip(gts, preds):
            render_intrinsics = _predicted_intrinsics(pred)
            target, intrinsics, height, width = _render_inputs(
                gt,
                self.render_stride,
                intrinsics=render_intrinsics,
            )
            geometry_mask = _geometry_mask(
                pred,
                self.render_stride,
            )
            for batch_index in range(batch_size):
                use_gt_depth = _gt_depth_is_available(
                    gt,
                    batch_index,
                    self.depth_target_source,
                )
                surfels = sample_surfels(
                    pred,
                    batch_index,
                    stride=self.render_stride,
                    scale_multiplier=self.render_stride,
                )
                rendered = render_surfels(
                    *surfels,
                    intrinsics[batch_index],
                    height,
                    width,
                )
                target_depth = _depth_target(
                    gt,
                    pred,
                    batch_index,
                    self.render_stride,
                    use_gt_depth,
                )[None]
                target_normal = (
                    _normal_from_depth(
                        target_depth[..., 0],
                        intrinsics[[batch_index]],
                    )
                    if self.normal_weight > 0.0
                    else None
                )
                (
                    loss,
                    rgb_mse,
                    lpips_loss,
                    normal,
                    normal_consistency,
                    normal_smooth,
                ) = self._render_loss(
                    rendered,
                    target[[batch_index]],
                    target_depth,
                    target_normal,
                    geometry_mask[[batch_index]],
                )
                losses.append(loss)
                rgb_mse_losses.append(rgb_mse)
                lpips_losses.append(lpips_loss)
                normal_losses.append(normal)
                normal_consistency_losses.append(normal_consistency)
                normal_smooth_losses.append(normal_smooth)
                alpha_means.append(rendered["alpha"].mean())
                gt_depth_flags.append(
                    target_depth.new_tensor(float(use_gt_depth))
                )
        return (
            torch.stack(losses).mean(),
            torch.stack(rgb_mse_losses).mean(),
            torch.stack(lpips_losses).mean(),
            torch.stack(normal_losses).mean(),
            torch.stack(normal_consistency_losses).mean(),
            torch.stack(normal_smooth_losses).mean(),
            torch.stack(alpha_means).mean(),
            torch.stack(gt_depth_flags).mean(),
        )

    def _merged_loss(self, gts, preds):
        zero = preds[0]["gaussian_2d_rgb"].new_zeros(())
        if len(preds) < 2 or self.merged_weight == 0.0:
            return zero, zero, zero, zero, zero, zero, zero, zero
        if "camera_pose" not in preds[0]:
            raise KeyError("Merged 2DGS render loss requires camera_pose predictions")

        cameras = [
            _pose_encoding_to_camera(pred["camera_pose"].detach().float())
            for pred in preds
        ]
        losses = []
        rgb_mse_losses = []
        lpips_losses = []
        normal_losses = []
        normal_consistency_losses = []
        normal_smooth_losses = []
        alpha_means = []
        gt_depth_flags = []
        batch_size = preds[0]["pts3d_in_self_view"].shape[0]

        for target_index, (gt, pred) in enumerate(zip(gts, preds)):
            render_intrinsics = _predicted_intrinsics(pred)
            target, intrinsics, height, width = _render_inputs(
                gt,
                self.render_stride,
                intrinsics=render_intrinsics,
            )
            geometry_mask = _geometry_mask(
                pred,
                self.render_stride,
            )
            for batch_index in range(batch_size):
                merged = [[] for _ in range(5)]
                for source_index, source_pred in enumerate(preds):
                    if source_index == target_index:
                        continue
                    means, colors, opacities, scales, rotations = sample_surfels(
                        source_pred,
                        batch_index,
                        stride=self.render_stride,
                        scale_multiplier=self.render_stride,
                    )
                    if means.numel() == 0:
                        continue
                    c2w = cameras[source_index][batch_index].to(
                        device=means.device,
                        dtype=means.dtype,
                    )
                    means = means @ c2w[:3, :3].T + c2w[:3, 3]
                    camera_rotation = source_pred["camera_pose"][
                        batch_index,
                        3:7,
                    ].detach().to(device=rotations.device, dtype=rotations.dtype)
                    rotations = _quaternion_multiply(
                        camera_rotation.expand_as(rotations),
                        rotations,
                    )
                    for bucket, value in zip(
                        merged,
                        (means, colors, opacities, scales, rotations),
                    ):
                        bucket.append(value)

                if not merged[0]:
                    continue
                use_gt_depth = _gt_depth_is_available(
                    gt,
                    batch_index,
                    self.depth_target_source,
                )
                target_depth = _depth_target(
                    gt,
                    pred,
                    batch_index,
                    self.render_stride,
                    use_gt_depth,
                )[None]
                target_normal = (
                    _normal_from_depth(
                        target_depth[..., 0],
                        intrinsics[[batch_index]],
                    )
                    if self.normal_weight > 0.0
                    else None
                )
                viewmat = torch.linalg.inv(cameras[target_index][batch_index]).to(
                    device=target_depth.device,
                    dtype=target_depth.dtype,
                )
                rendered = render_surfels(
                    *(torch.cat(bucket) for bucket in merged),
                    intrinsics[batch_index],
                    height,
                    width,
                    viewmat=viewmat,
                )
                (
                    loss,
                    rgb_mse,
                    lpips_loss,
                    normal,
                    normal_consistency,
                    normal_smooth,
                ) = self._render_loss(
                    rendered,
                    target[[batch_index]],
                    target_depth,
                    target_normal,
                    geometry_mask[[batch_index]],
                )
                losses.append(loss)
                rgb_mse_losses.append(rgb_mse)
                lpips_losses.append(lpips_loss)
                normal_losses.append(normal)
                normal_consistency_losses.append(normal_consistency)
                normal_smooth_losses.append(normal_smooth)
                alpha_means.append(rendered["alpha"].mean())
                gt_depth_flags.append(target_depth.new_tensor(float(use_gt_depth)))

        if not losses:
            return zero, zero, zero, zero, zero, zero, zero, zero
        return (
            torch.stack(losses).mean(),
            torch.stack(rgb_mse_losses).mean(),
            torch.stack(lpips_losses).mean(),
            torch.stack(normal_losses).mean(),
            torch.stack(normal_consistency_losses).mean(),
            torch.stack(normal_smooth_losses).mean(),
            torch.stack(alpha_means).mean(),
            torch.stack(gt_depth_flags).mean(),
        )

    def forward(self, gts, preds):
        required = {
            "pts3d_in_self_view",
            "gaussian_2d_rgb",
            "gaussian_2d_opacity",
            "gaussian_2d_scales",
        }
        missing = required - preds[0].keys()
        if missing:
            raise KeyError(f"Missing 2D Gaussian outputs: {sorted(missing)}")
        if (
            "gaussian_2d_rotations" not in preds[0]
            and "gaussian_2d_delta_rotations" not in preds[0]
        ):
            raise KeyError("Missing 2D Gaussian rotations")

        (
            self_loss,
            self_rgb_mse,
            self_lpips,
            self_normal_loss,
            self_normal_consistency,
            self_normal_smooth,
            self_alpha_mean,
            self_gt_depth_ratio,
        ) = self._self_loss(gts, preds)
        (
            merged_loss,
            merged_rgb_mse,
            merged_lpips,
            merged_normal_loss,
            merged_normal_consistency,
            merged_normal_smooth,
            merged_alpha_mean,
            merged_gt_depth_ratio,
        ) = self._merged_loss(gts, preds)
        total = (
            self.self_weight * self_loss
            + self.merged_weight * merged_loss
        )

        stat_weight = self.self_weight
        merged_active = len(preds) >= 2 and self.merged_weight > 0.0
        if merged_active:
            stat_weight = stat_weight + self.merged_weight
        stat_weight = max(stat_weight, 1e-8)
        rgb_mse = (
            self.self_weight * self_rgb_mse
            + (self.merged_weight * merged_rgb_mse if merged_active else 0.0)
        ) / stat_weight
        lpips_loss = (
            self.self_weight * self_lpips
            + (self.merged_weight * merged_lpips if merged_active else 0.0)
        ) / stat_weight
        normal_loss = (
            self.self_weight * self_normal_loss
            + (
                self.merged_weight * merged_normal_loss
                if merged_active
                else 0.0
            )
        ) / stat_weight
        normal_consistency = (
            self.self_weight * self_normal_consistency
            + (
                self.merged_weight * merged_normal_consistency
                if merged_active
                else 0.0
            )
        ) / stat_weight
        normal_smooth = (
            self.self_weight * self_normal_smooth
            + (
                self.merged_weight * merged_normal_smooth
                if merged_active
                else 0.0
            )
        ) / stat_weight
        alpha_mean = (
            self.self_weight * self_alpha_mean
            + (
                self.merged_weight * merged_alpha_mean
                if merged_active
                else 0.0
            )
        ) / stat_weight
        gt_depth_ratio = (
            self.self_weight * self_gt_depth_ratio
            + (
                self.merged_weight * merged_gt_depth_ratio
                if merged_active
                else 0.0
            )
        ) / stat_weight

        scales = torch.cat(
            [pred["gaussian_2d_scales"].reshape(-1, 2) for pred in preds]
        )
        opacities = torch.cat(
            [pred["gaussian_2d_opacity"].reshape(-1) for pred in preds]
        )
        geometry_mask_ratio = torch.stack(
            [
                _geometry_mask(
                    pred,
                    self.render_stride,
                ).mean()
                for pred in preds
            ]
        ).mean()
        return total, {
            "gaussian_2d_loss": float(total.detach()),
            "self_render_loss": float(self_loss.detach()),
            "merged_render_loss": float(merged_loss.detach()),
            "rgb_mse": float(rgb_mse.detach()),
            "lpips_loss": float(lpips_loss.detach()),
            "normal_loss": float(normal_loss.detach()),
            "normal_consistency": float(normal_consistency.detach()),
            "normal_smooth": float(normal_smooth.detach()),
            "mean_scale": float(scales.mean().detach()),
            "mean_opacity": float(opacities.mean().detach()),
            "mean_alpha": float(alpha_mean.detach()),
            "geometry_mask_ratio": float(geometry_mask_ratio.detach()),
            "gt_depth_ratio": float(gt_depth_ratio.detach()),
        }
