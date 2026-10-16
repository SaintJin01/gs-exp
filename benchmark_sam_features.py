"""Benchmark a few real training steps without modifying trained checkpoints."""
import json
import time
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer
import train_sam_features as training
from save_q3_gaussians_all_cameras import camera_from_json_record
from saga_supervision import sample_supervision, contrastive_loss


def main():
    parser = ArgumentParser(__doc__)
    parser.add_argument('-m', '--model_path', required=True)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--backend', default='rgb', choices=('rgb', 'feature'))
    parser.add_argument('--camera_id', type=int, default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--verify', action='store_true', help='compare full-model feature/RGB outputs and gradients')
    args = parser.parse_args()
    torch.manual_seed(0)
    model = Path(args.model_path)
    manifest = json.loads((model / 'sam_feature_data/manifest.json').read_text())
    record = next((r for r in manifest['cameras'] if args.camera_id is None or r['camera_id'] == args.camera_id), None)
    if record is None:
        raise ValueError('Camera not in training manifest')
    camera_record = record['cameras_json_record'].copy()
    # Match Scene's default automatic downscaling exactly.
    original_width, original_height = camera_record['width'], camera_record['height']
    if original_width > 1600:
        factor = original_width / 1600
        camera_record['width'], camera_record['height'] = round(original_width/factor), round(original_height/factor)
        camera_record['fx'] *= camera_record['width']/original_width
        camera_record['fy'] *= camera_record['height']/original_height
    camera = camera_from_json_record(camera_record)
    cloud = training.GaussianModel(3)
    iteration = manifest['iteration']
    cloud.load_ply(str(model / 'point_cloud' / ('iteration_%d' % iteration) / 'point_cloud.ply'))
    cloud.active_sh_degree = 3
    pipe = SimpleNamespace(compute_cov3D_python=False, debug=False, antialiasing=False,
                           feature_backend=args.backend)
    label = model / 'sam_feature_data' / record['file']
    # Pre-fit with every view's scales, as in actual training.
    all_scales = []
    for item in manifest['cameras']:
        with np.load(str(model / 'sam_feature_data' / item['file'])) as data:
            all_scales.append(data['mask_scales'])
    all_scales = np.concatenate(all_scales)
    quantile = QuantileTransformer(output_distribution='uniform').fit(all_scales[:, None])
    neighbors = training.load_or_build_xyz_knn(cloud.get_xyz, 16,
        model/'sam_features'/('xyz_knn_iteration_%06d_k16.npy' % iteration), 200000)
    features = torch.nn.Parameter(torch.randn(len(cloud.get_xyz), 32, device='cuda') * .01)
    gate = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Sigmoid()).cuda()
    optimizer = torch.optim.Adam([features] + list(gate.parameters()), lr=.0025, eps=1e-15)
    parity = None
    if args.verify:
        smoothed = training.smooth_gaussian_features(features, neighbors, .5, 65536)
        pipe.feature_backend = 'feature'
        fast = training.render_feature_map(camera, cloud, pipe, smoothed)
        pipe.feature_backend = 'rgb'
        reference = training.render_feature_map(camera, cloud, pipe, smoothed)
        torch.testing.assert_allclose(fast, reference, rtol=2e-5, atol=2e-6)
        upstream = torch.randn_like(fast)
        actual_grad = torch.autograd.grad(fast, smoothed, upstream)[0]
        expected_grad = torch.autograd.grad(reference, smoothed, upstream)[0]
        relative = float((actual_grad-expected_grad).norm()/expected_grad.norm().clamp_min(1e-12))
        if relative > 1e-4:
            raise AssertionError('Feature gradient relative error: %g' % relative)
        parity = {'output_max_abs_error': float((fast-reference).abs().max()),
                  'gradient_relative_l2_error': relative}
        print('PARITY ' + json.dumps(parity), flush=True)
        del fast, reference, upstream, actual_grad, expected_grad, smoothed
        pipe.feature_backend = args.backend
    rows = []
    print('Benchmark: %d Gaussians, %dx%d render, backend=%s' %
          (len(features), camera.image_width, camera.image_height, args.backend), flush=True)
    for step in range(args.steps + 1):
        row = {}
        def mark(name, started):
            torch.cuda.synchronize()
            row[name] = time.perf_counter() - started
            return time.perf_counter()
        torch.cuda.synchronize()
        started = time.perf_counter()
        packed, coverage, h, w = training.load_packed_masks(str(label))
        with np.load(str(label)) as data:
            areas, scales = data['mask_areas'], data['mask_scales']
        started = mark('load_masks', started)
        supervision = sample_supervision(packed, coverage, areas, scales, h, w, 1000,
                                          float(all_scales.max()), 'cuda')
        started = mark('supervision', started)
        smoothed = training.smooth_gaussian_features(features, neighbors, .5, 65536)
        started = mark('knn_forward', started)
        image = training.render_feature_map(camera, cloud, pipe, smoothed)
        started = mark('render_forward', started)
        loss, _ = contrastive_loss(image, supervision, gate, quantile, (h, w))
        started = mark('loss_forward', started)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        started = mark('backward', started)
        optimizer.step()
        mark('optimizer', started)
        row['total'] = sum(row.values())
        print('step %d %s' % (step, json.dumps(row)), flush=True)
        if step:
            rows.append(row)
        del image, loss, smoothed, supervision
    means = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    print('MEAN ' + json.dumps(means), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps({'backend': args.backend, 'camera_id': record['camera_id'],
                                               'steps': rows, 'mean': means, 'parity': parity}, indent=2))


if __name__ == '__main__':
    main()
