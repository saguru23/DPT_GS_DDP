import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization

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


def _ssim_loss(pred, target, mask=None):
    pred = pred.permute(0, 3, 1, 2)
    target = target.permute(0, 3, 1, 2)
    mu_pred = F.avg_pool2d(pred, 3, 1, 1)
    mu_target = F.avg_pool2d(target, 3, 1, 1)
    var_pred = F.avg_pool2d(pred.square(), 3, 1, 1) - mu_pred.square()
    var_target = F.avg_pool2d(target.square(), 3, 1, 1) - mu_target.square()
    covariance = F.avg_pool2d(pred * target, 3, 1, 1) - mu_pred * mu_target
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2 * mu_pred * mu_target + c1) * (2 * covariance + c2)) / (
        (mu_pred.square() + mu_target.square() + c1)
        * (var_pred + var_target + c2)
    ).clamp_min(1e-8)
    loss = ((1 - ssim) * 0.5).clamp(0, 1)
    if mask is None:
        return loss.mean()
    mask = mask.permute(0, 3, 1, 2).expand_as(loss)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


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


def _geometry_mask(pred, stride, confidence_quantile):
    points = pred["pts3d_in_self_view"].detach()
    mask = torch.isfinite(points).all(dim=-1) & (points[..., 2] > 1e-4)
    confidence = pred.get("conf_self")
    if confidence is not None and confidence_quantile > 0:
        confidence = confidence.detach()
        for batch_index in range(len(mask)):
            valid_confidence = confidence[batch_index][
                mask[batch_index] & torch.isfinite(confidence[batch_index])
            ]
            if valid_confidence.numel() > 0:
                threshold = torch.quantile(
                    valid_confidence.float(), confidence_quantile
                )
                mask[batch_index] &= confidence[batch_index] >= threshold

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


def sample_gaussians(pred, batch_index, stride=1, scale_multiplier=1.0):
    selection = (batch_index, slice(None, None, stride), slice(None, None, stride))
    means = pred["pts3d_in_self_view"][selection]
    colors = pred["gaussian_rgb"][selection]
    opacities = pred["gaussian_opacity"][selection].squeeze(-1)
    scales = pred["gaussian_scales"][selection] * scale_multiplier
    rotations = pred["gaussian_rotations"][selection]
    valid = (
        torch.isfinite(means).all(dim=-1)
        & torch.isfinite(colors).all(dim=-1)
        & torch.isfinite(opacities)
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


def render_gaussians(
    means,
    colors,
    opacities,
    scales,
    rotations,
    viewmat,
    intrinsics,
    height,
    width,
):
    if means.numel() == 0:
        raise RuntimeError("No valid Gaussians remain after filtering")
    rgbd, alpha, _ = rasterization(
        means=means.float(),
        quats=_standardize_quaternion(rotations.float()),
        scales=scales.float().clamp_min(1e-6),
        opacities=opacities.float().clamp(1e-4, 1 - 1e-4),
        colors=colors.float().clamp(0, 1),
        viewmats=viewmat.float()[None],
        Ks=intrinsics.float()[None],
        width=width,
        height=height,
        packed=True,
        render_mode="RGB+D",
    )
    return rgbd[..., :3], rgbd[..., 3], alpha


def render_prediction_view(
    gt,
    pred,
    batch_index=0,
    intrinsics_source="predicted",
):
    intrinsics = (
        _predicted_intrinsics(pred)
        if intrinsics_source == "predicted"
        else gt["camera_intrinsics"]
    )
    target, intrinsics, height, width = _render_inputs(
        gt, stride=1, intrinsics=intrinsics
    )
    means, colors, opacities, scales, rotations = sample_gaussians(
        pred, batch_index
    )
    identity = torch.eye(4, device=means.device, dtype=means.dtype)
    rgb, depth, alpha = render_gaussians(
        means,
        colors,
        opacities,
        scales,
        rotations,
        identity,
        intrinsics[batch_index],
        height,
        width,
    )
    return rgb[0], depth[0], alpha[0], target[batch_index]


class GaussianRenderingLoss(nn.Module):
    def __init__(
        self,
        *,
        render_stride=1,
        rgb_weight=1.0,
        ssim_weight=0.2,
        self_weight=1.0,
        merged_weight=1.0,
        scale_reg_weight=1e-2,
        opacity_reg_weight=1e-2,
        scale_reference=1e-2,
        opacity_reference=0.1,
        intrinsics_source="predicted",
        confidence_quantile=0.1,
    ):
        super().__init__()
        if render_stride < 1:
            raise ValueError("render_stride must be >= 1")
        if self_weight < 0.0 or merged_weight < 0.0:
            raise ValueError("self_weight and merged_weight must be >= 0")
        if self_weight == 0.0 and merged_weight == 0.0:
            raise ValueError("At least one render loss weight must be > 0")
        if rgb_weight < 0.0 or ssim_weight < 0.0:
            raise ValueError("RGB and SSIM weights must be >= 0")
        if scale_reg_weight < 0.0 or opacity_reg_weight < 0.0:
            raise ValueError("Regularization weights must be >= 0")
        self.render_stride = render_stride
        self.rgb_weight = rgb_weight
        self.ssim_weight = ssim_weight
        self.self_weight = self_weight
        self.merged_weight = merged_weight
        self.scale_reg_weight = scale_reg_weight
        self.opacity_reg_weight = opacity_reg_weight
        self.scale_reference = scale_reference
        self.opacity_reference = opacity_reference
        if intrinsics_source not in {"predicted", "ground_truth"}:
            raise ValueError(
                "intrinsics_source must be 'predicted' or 'ground_truth'"
            )
        if not 0.0 <= confidence_quantile < 1.0:
            raise ValueError("confidence_quantile must be in [0, 1)")
        self.intrinsics_source = intrinsics_source
        self.confidence_quantile = confidence_quantile

    def _image_loss(self, rendered, target, mask):
        mask_rgb = mask.expand_as(rendered)
        l1 = (
            (rendered - target).abs() * mask_rgb
        ).sum() / mask_rgb.sum().clamp_min(1.0)
        ssim = _ssim_loss(rendered, target, mask)
        return self.rgb_weight * l1 + self.ssim_weight * ssim, l1, ssim

    def _self_loss(self, gts, preds):
        losses = []
        l1_losses = []
        ssim_losses = []
        alpha_means = []
        batch_size = preds[0]["pts3d_in_self_view"].shape[0]
        identity = torch.eye(
            4,
            device=preds[0]["pts3d_in_self_view"].device,
            dtype=preds[0]["pts3d_in_self_view"].dtype,
        )
        for gt, pred in zip(gts, preds):
            render_intrinsics = (
                _predicted_intrinsics(pred)
                if self.intrinsics_source == "predicted"
                else gt["camera_intrinsics"]
            )
            target, intrinsics, height, width = _render_inputs(
                gt,
                self.render_stride,
                intrinsics=render_intrinsics,
            )
            geometry_mask = _geometry_mask(
                pred,
                self.render_stride,
                self.confidence_quantile,
            )
            for batch_index in range(batch_size):
                gaussian = sample_gaussians(
                    pred,
                    batch_index,
                    stride=self.render_stride,
                    scale_multiplier=self.render_stride,
                )
                rendered, _, alpha = render_gaussians(
                    *gaussian,
                    identity,
                    intrinsics[batch_index],
                    height,
                    width,
                )
                loss, l1, ssim = self._image_loss(
                    rendered,
                    target[[batch_index]],
                    geometry_mask[[batch_index]],
                )
                losses.append(loss)
                l1_losses.append(l1)
                ssim_losses.append(ssim)
                alpha_means.append(alpha.mean())
        return (
            torch.stack(losses).mean(),
            torch.stack(l1_losses).mean(),
            torch.stack(ssim_losses).mean(),
            torch.stack(alpha_means).mean(),
        )

    def _merged_loss(self, gts, preds):
        zero = preds[0]["gaussian_rgb"].new_zeros(())
        if len(preds) < 2 or self.merged_weight == 0:
            return zero, zero, zero, zero

        cameras = [
            _pose_encoding_to_camera(pred["camera_pose"].detach().float())
            for pred in preds
        ]
        losses = []
        l1_losses = []
        ssim_losses = []
        alpha_means = []
        batch_size = preds[0]["pts3d_in_self_view"].shape[0]

        for target_index, (gt, pred) in enumerate(zip(gts, preds)):
            target_intrinsics = (
                _predicted_intrinsics(pred)
                if self.intrinsics_source == "predicted"
                else gt["camera_intrinsics"]
            )
            target, intrinsics, height, width = _render_inputs(
                gt,
                self.render_stride,
                intrinsics=target_intrinsics,
            )
            geometry_mask = _geometry_mask(
                pred,
                self.render_stride,
                self.confidence_quantile,
            )
            for batch_index in range(batch_size):
                merged = [[] for _ in range(5)]
                for source_index, source_pred in enumerate(preds):
                    if source_index == target_index:
                        continue
                    means, colors, opacities, scales, rotations = sample_gaussians(
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
                        camera_rotation.expand_as(rotations), rotations
                    )
                    for bucket, value in zip(
                        merged, (means, colors, opacities, scales, rotations)
                    ):
                        bucket.append(value)

                if not merged[0]:
                    continue
                viewmat = torch.linalg.inv(cameras[target_index][batch_index]).to(
                    device=intrinsics.device,
                    dtype=intrinsics.dtype,
                )
                rendered, _, alpha = render_gaussians(
                    *(torch.cat(bucket) for bucket in merged),
                    viewmat,
                    intrinsics[batch_index],
                    height,
                    width,
                )
                loss, l1, ssim = self._image_loss(
                    rendered,
                    target[[batch_index]],
                    geometry_mask[[batch_index]],
                )
                losses.append(loss)
                l1_losses.append(l1)
                ssim_losses.append(ssim)
                alpha_means.append(alpha.mean())
        if not losses:
            return zero, zero, zero, zero
        return (
            torch.stack(losses).mean(),
            torch.stack(l1_losses).mean(),
            torch.stack(ssim_losses).mean(),
            torch.stack(alpha_means).mean(),
        )

    def forward(self, gts, preds):
        required = {
            "pts3d_in_self_view",
            "gaussian_rgb",
            "gaussian_opacity",
            "gaussian_scales",
            "gaussian_rotations",
        }
        missing = required - preds[0].keys()
        if missing:
            raise KeyError(f"Missing Gaussian outputs: {sorted(missing)}")
        if (
            len(preds) >= 2
            and self.merged_weight > 0
            and "camera_pose" not in preds[0]
        ):
            raise KeyError("Merged GS render loss requires camera_pose predictions")

        self_loss, self_rgb_l1, self_ssim, self_alpha_mean = self._self_loss(
            gts,
            preds,
        )
        (
            merged_loss,
            merged_rgb_l1,
            merged_ssim,
            merged_alpha_mean,
        ) = self._merged_loss(gts, preds)
        scales = torch.cat(
            [pred["gaussian_scales"].reshape(-1, 3) for pred in preds]
        )
        opacities = torch.cat(
            [pred["gaussian_opacity"].reshape(-1) for pred in preds]
        )
        scale_reg = (
            scales.clamp_min(1e-8).log()
            - scales.new_tensor(self.scale_reference).log()
        ).square().mean()
        opacity_reg = (opacities - self.opacity_reference).square().mean()
        loss = (
            self.self_weight * self_loss
            + self.merged_weight * merged_loss
            + self.scale_reg_weight * scale_reg
            + self.opacity_reg_weight * opacity_reg
        )
        stat_weight = self.self_weight
        merged_active = len(preds) >= 2 and self.merged_weight > 0.0
        if merged_active:
            stat_weight = stat_weight + self.merged_weight
        stat_weight = max(stat_weight, 1e-8)
        rgb_l1 = (
            self.self_weight * self_rgb_l1
            + (self.merged_weight * merged_rgb_l1 if merged_active else 0.0)
        ) / stat_weight
        ssim = (
            self.self_weight * self_ssim
            + (self.merged_weight * merged_ssim if merged_active else 0.0)
        ) / stat_weight
        alpha_mean = (
            self.self_weight * self_alpha_mean
            + (self.merged_weight * merged_alpha_mean if merged_active else 0.0)
        ) / stat_weight
        geometry_mask_ratio = torch.stack(
            [
                _geometry_mask(
                    pred,
                    self.render_stride,
                    self.confidence_quantile,
                ).mean()
                for pred in preds
            ]
        ).mean()
        return loss, {
            "gaussian_loss": float(loss.detach()),
            "self_render_loss": float(self_loss.detach()),
            "merged_render_loss": float(merged_loss.detach()),
            "rgb_l1": float(rgb_l1.detach()),
            "ssim": float(ssim.detach()),
            "mean_scale": float(scales.mean().detach()),
            "mean_opacity": float(opacities.mean().detach()),
            "mean_alpha": float(alpha_mean.detach()),
            "geometry_mask_ratio": float(geometry_mask_ratio.detach()),
            "scale_reg_loss": float(
                (self.scale_reg_weight * scale_reg).detach()
            ),
            "opacity_reg_loss": float(
                (self.opacity_reg_weight * opacity_reg).detach()
            ),
        }
