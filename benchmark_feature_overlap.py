"""Small real-data benchmark; never modifies clustering outputs."""
import argparse
import cProfile
import importlib.util
import io
import json
import pstats
import time
from pathlib import Path
import numpy as np
from plyfile import PlyData
import cluster_gaussians_feature_overlap as graph
from cluster_gaussian_features import normalized_features
from cluster_gaussians_twice import gaussian_geometry


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--ply_file',required=True)
    parser.add_argument('--feature_file',required=True)
    parser.add_argument('--reference_script',required=True)
    parser.add_argument('--points',type=int,default=2000)
    parser.add_argument('--output',default=None)
    args=parser.parse_args()
    vertex=PlyData.read(args.ply_file)['vertex'].data
    vertex=vertex[:min(args.points,len(vertex))].copy()
    xyz=np.stack([vertex[k] for k in ('x','y','z')],axis=1).astype(np.float64)
    features=graph.load_features(Path(args.feature_file))
    features=normalized_features(features,slice(0,len(vertex)))
    geometry=gaussian_geometry(vertex,3)
    spec=importlib.util.spec_from_file_location('overlap_reference',args.reference_script)
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    results={};outputs=[]
    for name,module in (('before',reference),('after',graph)):
        profile=cProfile.Profile();started=time.perf_counter();profile.enable()
        labels,count,stats=module.cluster_overlap_graph(xyz,geometry,features,.7,1,2048)
        profile.disable();elapsed=time.perf_counter()-started
        results[name]=dict(seconds=elapsed,clusters=count,**stats);outputs.append(labels)
        stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats('tottime').print_stats(10)
        print(name,json.dumps(results[name]),stream.getvalue(),flush=True)
    old,new=outputs
    pairs=np.unique(np.stack((old,new),axis=1),axis=0)
    identical=len(pairs)==len(np.unique(old))==len(np.unique(new))
    results['same_partition']=identical
    results['speedup']=results['before']['seconds']/results['after']['seconds']
    print('RESULT',json.dumps(results),flush=True)
    if not identical:raise AssertionError('Optimized graph partition differs')
    if args.output:Path(args.output).write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
