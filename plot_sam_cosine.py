"""Compare saved checkpoints on one fixed training view; this is not training-log recovery."""
import json
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import train_sam_features as training
from cosine_history import CosineHistory
from save_q3_gaussians_all_cameras import camera_from_json_record
from saga_supervision import sample_supervision, contrastive_loss


@torch.no_grad()
def main():
    parser = ArgumentParser(__doc__)
    parser.add_argument('-m', '--model_path', required=True)
    parser.add_argument('-c', '--camera_id', type=int, default=None)
    args = parser.parse_args()
    model = Path(args.model_path)
    folder = model/'sam_features'
    checkpoints = sorted(folder.glob('iteration_*.pt'))
    if not checkpoints:
        raise ValueError('No iteration checkpoints found')
    manifest = json.loads((model/'sam_feature_data/manifest.json').read_text())
    record = next((r for r in manifest['cameras'] if args.camera_id is None or r['camera_id']==args.camera_id), None)
    if record is None:
        raise ValueError('Camera is not in prepared training views')
    config = json.loads((folder/'training_config.json').read_text())
    camera_record = record['cameras_json_record'].copy()
    ow, oh = camera_record['width'], camera_record['height']
    resolution = config.get('resolution', -1)
    if resolution in (1, 2, 4, 8):
        width, height = round(ow/resolution), round(oh/resolution)
    else:
        factor = ow/1600 if resolution == -1 and ow>1600 else (1 if resolution == -1 else ow/resolution)
        width, height = int(ow/factor), int(oh/factor)
    camera_record.update(width=width, height=height, fx=camera_record['fx']*width/ow,
                         fy=camera_record['fy']*height/oh)
    camera = camera_from_json_record(camera_record)
    cloud = training.GaussianModel(3)
    cloud.load_ply(str(model/'point_cloud'/('iteration_%d'%config['source_iteration'])/'point_cloud.ply'))
    pipe = SimpleNamespace(debug=False, antialiasing=config.get('antialiasing',False),
                           compute_cov3D_python=False, feature_backend='feature')
    packed, coverage, height, width = training.load_packed_masks(str(model/'sam_feature_data'/record['file']))
    all_scales=[]
    for r in manifest['cameras']:
        with np.load(str(model/'sam_feature_data'/r['file'])) as data:
            all_scales.append(data['mask_scales'])
    torch.manual_seed(0)
    with np.load(str(model/'sam_feature_data'/record['file'])) as data:
        supervision=sample_supervision(packed,coverage,data['mask_areas'],data['mask_scales'],
            height,width,1000,float(np.concatenate(all_scales).max()),'cuda')
    if supervision is None:
        raise ValueError('Not enough covered rays in this camera')
    output = folder/'cosine_eval'
    history=CosineHistory(output)
    for path in checkpoints:
        checkpoint=torch.load(str(path),map_location='cpu')
        if checkpoint.get('training_format')!='saga_scale_conditioned_v2':
            raise ValueError('Checkpoint lacks scale-conditioned training metadata: %s'%path)
        features=checkpoint['features'].cuda()
        if len(features)!=len(cloud.get_xyz) or checkpoint['source_iteration']!=config['source_iteration']:
            raise ValueError('Checkpoint geometry mismatch')
        k=checkpoint['training']['smooth_k']
        neighbors=training.load_or_build_xyz_knn(cloud.get_xyz,k,
            folder/('xyz_knn_iteration_%06d_k%d.npy'%(config['source_iteration'],k)),200000)
        smoothed=training.smooth_gaussian_features(features,neighbors,-1,65536)
        image=training.render_feature_map(camera,cloud,pipe,smoothed,config.get('white_background',False))
        gate=torch.nn.Sequential(torch.nn.Linear(1,features.shape[1]),torch.nn.Sigmoid()).cuda()
        gate.load_state_dict(checkpoint['scale_gate'])
        _,stats=contrastive_loss(image,supervision,gate,checkpoint['quantile_transform'],
            (height,width),scale_aware_dim=checkpoint['training'].get('scale_aware_dim',-1),collect_cosine=True)
        history.record(checkpoint['iteration'],record['camera_id'],stats['positive_cosine'],stats['negative_cosine'])
        print('%d: positive=%.5f negative=%.5f'%(checkpoint['iteration'],stats['positive_cosine'],stats['negative_cosine']),flush=True)
        del features,smoothed,image,neighbors,gate,checkpoint
    history.plot(alpha=1,title='Positive / negative cosine across training checkpoints',
        note='Fixed training camera %d | same sampled rays and scales | all K neighbors | self-pairs excluded'%record['camera_id'])
    print('Saved %s'%history.plot_path,flush=True)


if __name__=='__main__':
    main()
