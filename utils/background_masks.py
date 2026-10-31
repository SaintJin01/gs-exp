"""CPU depth-to-background masks shared by previews and training."""
import cv2
import numpy as np

METHODS = {
    'far_q90': 'Farthest 10% valid depth',
    'far_q80': 'Farthest 20% valid depth',
    'far_q70': 'Farthest 30% valid depth',
    'otsu_log': 'Otsu split in log depth',
    'gap_log': 'Largest log-depth quantile gap',
    'depth_ge_10': 'Raw metric depth >= 10',
    'depth_ge_20': 'Raw metric depth >= 20',
    'invalid_only': 'Negative/nonfinite pixels only (sky hypothesis)',
    'invalid_far_q90': 'Invalid OR farthest 10%',
    'invalid_far_q80': 'Invalid OR farthest 20%',
    'invalid_otsu_log': 'Invalid OR log-Otsu far depth',
    'top_connected': 'Invalid OR farthest 20%, connected to top border',
}


def make_background_mask(depth, method, kind='metric'):
    if method not in METHODS:
        raise ValueError('Unknown mask method: {}'.format(method))
    if kind not in ('metric', 'inverse'):
        raise ValueError('Depth kind must be metric or inverse')
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError('Expected HxW single-channel depth')
    invalid = ~np.isfinite(depth) | (depth < 0)
    valid = ~invalid & (depth > 0)
    distance = depth.copy()
    if kind == 'inverse':
        distance[valid] = 1.0 / np.maximum(depth[valid], 1e-12)
    if method == 'invalid_only':
        return invalid
    base = method.replace('invalid_', '')
    if base == 'top_connected':
        base = 'far_q80'
    mask = np.zeros_like(valid)
    values = distance[valid]
    if values.size:
        if base.startswith('far_q'):
            threshold = np.quantile(values, int(base[5:])/100.)
        elif base.startswith('depth_ge_'):
            if kind != 'metric':
                raise ValueError('Absolute raw-depth methods require metric depth')
            threshold = float(base[9:])
        elif base == 'otsu_log':
            logs = np.log(np.maximum(values, 1e-12))
            lo, hi = np.percentile(logs, [1, 99])
            if hi <= lo:
                threshold = float('inf')
            else:
                bins = np.clip((logs-lo)/(hi-lo)*255, 0, 255).astype(np.uint8)
                cut, _ = cv2.threshold(bins.reshape(-1,1), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                threshold = np.exp(lo + (cut/255.)*(hi-lo))
        elif base == 'gap_log':
            # Upper-half quantile gaps avoid splitting off the nearest surface;
            # ignore the last 1% so isolated extreme estimates do not dominate.
            qs = np.quantile(values, np.linspace(.5, .99, 100))
            gaps = np.diff(np.log(np.maximum(qs, 1e-12)))
            best = int(gaps.argmax())
            threshold = np.sqrt(qs[best]*qs[best+1]) if gaps[best] > 0 else float('inf')
        else:
            raise ValueError(method)
        mask = valid & (distance >= threshold)
    if method.startswith('invalid_') or method == 'top_connected':
        mask |= invalid
    if method == 'top_connected':
        _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        ids = np.unique(labels[0]); ids = ids[ids != 0]
        mask = np.isin(labels, ids) if len(ids) else np.zeros_like(mask)
    return mask

# Preview-only fusion: depth sources can have different scales. Rank fusion avoids
# assuming they use the same physical units; absolute variants use metric raw units.
MIXED_METHODS = {
    'mix_invalid_consensus': 'Both sources invalid only',
    'mix_far_intersection': 'Both invalid OR both sources in farthest 20%',
    'mix_far_union': 'Both invalid OR either valid source in farthest 10%',
    'mix_metric_gate': 'Relative background, vetoed by near metric depth',
    'mix_relative_gate': 'Metric background, vetoed by near relative depth',
    'mix_rank_mean': 'Mean percentile >= 0.80; single valid source >= 0.90',
    'mix_rank_metric70': '70% metric + 30% relative percentile >= 0.80',
    'mix_log_aligned': 'Median-scale aligned geometric depth; farthest 15%',
    'mix_otsu_agree': 'Both invalid OR both log-Otsu far regions',
    'mix_metric_ge5': 'Both invalid OR metric raw depth >= 5',
    'mix_metric_ge10': 'Both invalid OR metric raw depth >= 10',
    'mix_top_connected': 'Top-connected components of mean-rank background',
}


def percentile_rank(depth, valid):
    rank = np.zeros(depth.shape, dtype=np.float32)
    if valid.any():
        values = depth[valid]
        knots = np.quantile(values, np.linspace(0, 1, 257))
        rank[valid] = np.searchsorted(knots, values, side='left') / 256.
    return np.clip(rank, 0, 1)


def make_mixed_masks(relative, metric):
    """Aligned HxW maps, larger means farther. Nonpositive values are unknown."""
    relative = np.asarray(relative, dtype=np.float32)
    metric = np.asarray(metric, dtype=np.float32)
    if relative.shape != metric.shape or relative.ndim != 2:
        raise ValueError('Fusion requires two aligned HxW depth maps')
    rv = np.isfinite(relative) & (relative > 0)
    mv = np.isfinite(metric) & (metric > 0)
    both = rv & mv
    unknown = ~rv & ~mv
    rr = percentile_rank(relative, rv)
    mr = percentile_rank(metric, mv)
    far_r = rv & (rr >= .8)
    far_m = mv & (mr >= .8)
    only_r = rv & ~mv
    only_m = mv & ~rv
    mean = unknown | (both & ((rr+mr)*.5 >= .8)) | (only_r & (rr >= .9)) | (only_m & (mr >= .9))
    weighted = unknown | (both & (.3*rr+.7*mr >= .8)) | (only_r & (rr >= .9)) | (only_m & (mr >= .9))
    result = {
        'mix_invalid_consensus': unknown,
        'mix_far_intersection': unknown | (far_r & far_m),
        'mix_far_union': unknown | (rv & (rr >= .9)) | (mv & (mr >= .9)),
        'mix_metric_gate': (~rv | far_r) & (~mv | (mr >= .7)),
        'mix_relative_gate': (~mv | far_m) & (~rv | (rr >= .7)),
        'mix_rank_mean': mean,
        'mix_rank_metric70': weighted,
        'mix_otsu_agree': unknown | (make_background_mask(relative,'otsu_log') & make_background_mask(metric,'otsu_log')),
        'mix_metric_ge5': unknown | (mv & (metric >= 5)),
        'mix_metric_ge10': unknown | (mv & (metric >= 10)),
    }
    # Align scales from overlapping valid pixels; geometric mean reduces the
    # effect of one source predicting much larger depth than the other.
    scale = float(np.median(metric[both] / relative[both])) if both.any() else 1.
    aligned = np.full(relative.shape, -1., dtype=np.float32)
    aligned[both] = np.sqrt(relative[both]*scale*metric[both])
    aligned[only_r] = relative[only_r]*scale
    aligned[only_m] = metric[only_m]
    valid = rv | mv
    threshold = np.quantile(aligned[valid], .85) if valid.any() else float('inf')
    result['mix_log_aligned'] = unknown | (valid & (aligned >= threshold))
    _, labels = cv2.connectedComponents(mean.astype(np.uint8), connectivity=8)
    ids = np.unique(labels[0]); ids = ids[ids != 0]
    result['mix_top_connected'] = np.isin(labels,ids) if len(ids) else np.zeros_like(mean)
    return result
