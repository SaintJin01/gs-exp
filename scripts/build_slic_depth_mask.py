#!/usr/bin/env python3
import argparse, csv, json, math, shutil
from pathlib import Path
import numpy as np
from PIL import Image

def parse_args():
    p=argparse.ArgumentParser(description="Build RGBA background inputs using SLIC-level relative depth and pixel-level metric depth.")
    p.add_argument("--source",required=True,type=Path);p.add_argument("--far_fraction",type=float,default=0.30);p.add_argument("--min_meters",type=float,default=5.0)
    p.add_argument("--output",type=Path);return p.parse_args()

def main():
    args=parse_args();source=args.source.resolve();output=(args.output or source/"compensation/depth_masked_slic_top30_min5m").resolve()
    if not 0<args.far_fraction<=1:raise ValueError("far_fraction must be in (0,1]")
    if output.exists() and any(output.iterdir()):raise RuntimeError("output must be empty: {}".format(output))
    output.mkdir(parents=True,exist_ok=True);preview=output/"preview";preview.mkdir()
    rgb_dir=source/"compensation/images";rel_dir=source/"depths_relative_da3mono";metric_dir=source/"depths_metric";slic_dir=source/"slic/labels";sky_dir=source/"sky"
    names=sorted(p.name for p in rgb_dir.glob("*.png"))
    if not names:raise RuntimeError("no compensated PNG images in {}".format(rgb_dir))
    rows=[];total_pixels=0;kept_pixels=0;kept_sky=0;kept_non_sky=0
    for index,name in enumerate(names):
        stem=Path(name).stem;rgb=np.asarray(Image.open(rgb_dir/name).convert("RGB"));rel=np.load(rel_dir/(stem+".npy"));metric=np.load(metric_dir/(stem+".npy"));labels=np.load(slic_dir/(stem+".npy"));sky=np.asarray(Image.open(sky_dir/name).convert("RGBA"))[...,3]>0
        shape=rgb.shape[:2]
        if rel.shape!=shape or metric.shape!=shape or labels.shape!=shape or sky.shape!=shape:raise ValueError("shape mismatch: {}".format(name))
        valid_rel=np.isfinite(rel)&(rel!=0)&(~sky);valid_labels=labels[valid_rel].astype(np.int64);valid_depth=rel[valid_rel].astype(np.float64);label_count=int(labels.max())+1
        counts=np.bincount(valid_labels,minlength=label_count);sums=np.bincount(valid_labels,weights=valid_depth,minlength=label_count);valid_ids=np.flatnonzero(counts>0);segment_depth=sums[valid_ids]/counts[valid_ids]
        if not len(valid_ids):raise RuntimeError("no valid SLIC relative-depth segments: {}".format(name))
        keep_count=max(1,int(math.ceil(len(valid_ids)*args.far_fraction)));order=np.argsort(segment_depth)[::-1];kept_ids=valid_ids[order[:keep_count]].astype(labels.dtype)
        selected_superpixels=np.isin(labels,kept_ids);metric_far=np.isfinite(metric)&(metric>args.min_meters);non_sky=selected_superpixels&metric_far&(~sky);mask=non_sky|sky
        rgba=np.dstack((rgb,np.where(mask,255,0).astype(np.uint8)));Image.fromarray(rgba,"RGBA").save(output/name,compress_level=3)
        if index%10==0:shutil.copy2(output/name,preview/name)
        total_pixels+=mask.size;kept_pixels+=int(mask.sum());kept_sky+=int(sky.sum());kept_non_sky+=int(non_sky.sum())
        rows.append({"image":name,"valid_superpixels":len(segment_depth),"kept_superpixels":keep_count,"kept_superpixel_fraction":keep_count/float(len(segment_depth)),"metric_far_non_sky_pixels":int(non_sky.sum()),"sky_pixels":int(sky.sum()),"final_pixels":int(mask.sum()),"final_fraction":float(mask.mean())})
        if (index+1)%20==0 or index+1==len(names):print("SLIC depth mask {}/{}".format(index+1,len(names)),flush=True)
    with (output/"image_statistics.csv").open("w",newline="",encoding="utf8") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    manifest={"rgb_source":str(rgb_dir),"relative_depth":str(rel_dir),"metric_depth":str(metric_dir),"slic_labels":str(slic_dir),"sky":str(sky_dir),"rule":"keep farthest {:.1%} of SLIC superpixels ranked by mean valid relative depth; inside retained superpixels keep metric depth > {:.3f}m pixel-wise; always keep sky".format(args.far_fraction,args.min_meters),"image_count":len(names),"total_pixels":total_pixels,"kept_pixels":kept_pixels,"kept_fraction":kept_pixels/float(total_pixels),"kept_non_sky_pixels":kept_non_sky,"kept_sky_pixels":kept_sky}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8");print(json.dumps(manifest,indent=2))

if __name__=="__main__":main()
