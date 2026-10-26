"""Scale-conditioned SAM supervision, following SegAnyGAussians v2.

Reference: Jumpat/SegAnyGAussians, train_contrastive_feature.py and get_scale.py.
Packed masks avoid retaining every full-resolution mask on the GPU.
"""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def mask_scales_from_depth(masks, depth, fovx, fovy):
    h, w = depth.shape
    row, col = torch.meshgrid(torch.arange(h, device=depth.device),
                              torch.arange(w, device=depth.device))
    # Preserve upstream's row/column convention, including its unusual x/y mapping.
    fx, fy = w / (2 * np.tan(fovx / 2)), h / (2 * np.tan(fovy / 2))
    points = torch.stack(((row - w / 2) * depth / fx,
                          (col - h / 2) * depth / fy, depth), dim=-1)
    values = []
    for mask in masks:
        resized = F.interpolate(torch.as_tensor(mask, device=depth.device,
                                dtype=torch.float32)[None, None], (h, w),
                                mode="bilinear", align_corners=False)
        interior = F.conv2d(resized, torch.ones(1, 1, 3, 3, device=depth.device),
                            padding=1)[0, 0] >= 5
        if interior.sum() < 2:
            values.append(float("nan"))
        else:
            values.append(float((2 * points[interior].std(dim=0)).norm()))
    return np.asarray(values, dtype=np.float32)


@torch.no_grad()
def sample_supervision(packed, coverage, areas, scales, height, width,
                       num_rays, upper_scale, device, ray_sample_rate=0.0):
    n = height * width
    probability = ray_sample_rate if ray_sample_rate > 0 else num_rays / n
    covered = torch.from_numpy(np.unpackbits(coverage, count=n).astype(bool)).to(device)
    pixels = torch.where((torch.rand(n, device=device) < probability) & covered)[0]
    if pixels.numel() < 2 or len(scales) == 0:
        return None
    ids = pixels.cpu().numpy()
    membership = ((packed[:, ids // 8] >> (7 - ids % 8)[None]) & 1)
    masks = torch.as_tensor(membership, dtype=torch.float32, device=device)
    scales = torch.as_tensor(scales, device=device)
    scales, order = scales.sort(descending=True)
    masks = masks[order]
    areas = torch.as_tensor(areas, device=device, dtype=torch.float32)[order]
    mean_area = (masks * areas[:, None]).sum(0) / (masks.sum(0) + 1e-9)
    product = mean_area[:, None] * mean_area[None, :]
    weights = (product.max() / product.clamp_min(1e-9)).clamp_min(1)
    span = weights.max() - weights.min()
    weights = (weights - weights.min()) / span * 9 + 1 if span > 0 else torch.ones_like(weights)
    # SAGA uses eight randomly selected scale indices plus two boundary scales.
    indices = torch.zeros(10, dtype=torch.long, device=device)
    chosen = torch.randperm(len(scales), device=device)[:8]
    indices[1:len(chosen) + 1] = chosen
    indices[0], indices[-1] = -1, len(scales) - 1
    sampled_scales = scales[indices].clone()
    sampled_scales[0] = upper_scale * (1 + torch.rand((), device=device))
    smaller = scales[scales < upper_scale]
    second = smaller.max() if smaller.numel() else scales.min()
    mask_indices = torch.arange(len(scales), device=device)[:, None]
    prefix_last = torch.where(masks.bool(), mask_indices.expand_as(masks),
                               torch.full_like(mask_indices.expand_as(masks), -1)).cummax(0).values
    correspondence = []
    for idx, index in enumerate(indices.tolist()):
        above = sampled_scales[idx] >= upper_scale
        lower = second if above else (scales[index + 1] if index != len(scales) - 1 else 0)
        sampled_scales[idx] -= (sampled_scales[idx] - lower) * torch.rand((), device=device)
        if above:
            vectors = masks
        else:
            # Equivalent to SAGA's reverse prefix loop: retain only the last
            # covering mask in the prefix, without hundreds of CUDA launches.
            vectors = masks.clone()
            vectors[:index + 1] = (mask_indices[:index + 1] == prefix_last[index][None]).float()
        correspondence.append((vectors.T @ vectors > 0).float())
    return pixels, torch.stack(correspondence), weights, sampled_scales


@torch.no_grad()
def cosine_statistics(correlation, truth):
    """Unweighted means over all scale-labelled pairs, excluding self pairs.

Use all pairs rather than only hard-mined pairs to avoid mining-selection bias.
Each unordered pixel pair is counted once per scale.
"""
    domain = torch.triu(torch.ones_like(truth, dtype=torch.bool), diagonal=1)
    positive = correlation.detach()[domain & truth.bool()]
    negative = correlation.detach()[domain & ~truth.bool()]
    return {
        "positive_cosine": float(positive.mean()) if positive.numel() else float("nan"),
        "negative_cosine": float(negative.mean()) if negative.numel() else float("nan"),
    }


def contrastive_loss(feature_map, supervision, gate, quantile, mask_shape,
                     rfn=1.0, positive_threshold=0.75, negative_threshold=0.5,
                     scale_aware_dim=-1, collect_cosine=False):
    pixels, truth, weights, scales = supervision
    norm_reg = (1 - feature_map.norm(dim=0).mean()).square()
    resized = F.interpolate(feature_map[None], mask_shape, mode="bilinear",
                            align_corners=False)[0]
    if gate is None:
        vectors = resized.flatten(1).T[pixels][None]
    else:
        normalized_scales = torch.as_tensor(quantile.transform(scales.detach().cpu().numpy()[:, None]),
                                            device=feature_map.device, dtype=feature_map.dtype)
        if 0 < scale_aware_dim < feature_map.shape[0]:
            levels = ((1 - normalized_scales[:, 0]) * scale_aware_dim).long()
            gates = (torch.arange(feature_map.shape[0], device=feature_map.device)[None] <
                     (feature_map.shape[0] - scale_aware_dim + levels)[:, None]).float()
        else:
            gates = gate(normalized_scales)
        vectors = resized.flatten(1).T[pixels][None] * gates[:, None]
    vectors = F.normalize(vectors, dim=-1)
    correlation = vectors @ vectors.transpose(1, 2)
    total = truth.sum(0)
    positive, negative = total == len(truth), total == 0
    mixed = ~(positive | negative)
    budget = mixed.sum().float() / 2
    random_values = torch.rand_like(total)
    select_pos = positive & (random_values < budget / positive.sum().clamp_min(1))
    select_neg = negative & (random_values < budget / negative.sum().clamp_min(1))
    select_pos |= mixed | ((correlation.detach() < positive_threshold) & truth.bool()).any(0)
    select_neg |= mixed | ((correlation.detach() > negative_threshold) & ~truth.bool()).any(0)
    select_pos, select_neg = torch.triu(select_pos, 1), torch.triu(select_neg, 1)
    zero = correlation.sum() * 0
    pull = (-weights[select_pos][None] * truth[:, select_pos] * correlation[:, select_pos]).mean() if select_pos.any() else zero
    push = (weights[select_neg][None] * (1 - truth[:, select_neg]) * F.relu(correlation[:, select_neg])).mean() if select_neg.any() else zero
    stats = {
        "positive_pairs": int(select_pos.sum()), "negative_pairs": int(select_neg.sum()),
        "rendered_norm": float(feature_map.detach().norm(dim=0).mean()),
    }
    if collect_cosine:
        stats.update(cosine_statistics(correlation, truth))
    return pull + push + rfn * norm_reg, stats


@torch.no_grad()
def sample_single_supervision(packed, coverage, areas, height, width, num_rays,
                              device, ray_sample_rate=0.0):
    """Single segment per pixel, no scale-conditioned correspondence."""
    n = height * width
    probability = ray_sample_rate if ray_sample_rate > 0 else num_rays / n
    covered = torch.as_tensor(np.unpackbits(coverage, count=n).astype(bool), device=device)
    pixels = torch.where((torch.rand(n, device=device) < probability) & covered)[0]
    if pixels.numel() < 2 or len(areas) == 0:
        return None
    ids = pixels.cpu().numpy()
    members = ((packed[:, ids // 8] >> (7 - ids % 8)[None]) & 1).astype(bool)
    if not (members.sum(axis=0) == 1).all():
        raise ValueError("Expected one mask per covered pixel; regenerate SAM data")
    labels = torch.as_tensor(members.argmax(axis=0), device=device)
    truth = (labels[:, None] == labels[None, :]).float()[None]
    pixel_areas = torch.as_tensor(areas, device=device, dtype=torch.float32)[labels]
    product = pixel_areas[:, None] * pixel_areas[None, :]
    weights = (product.max() / product.clamp_min(1e-9)).clamp_min(1)
    span = weights.max() - weights.min()
    weights = (weights - weights.min()) / span * 9 + 1 if span > 0 else torch.ones_like(weights)
    return pixels, truth, weights, None
