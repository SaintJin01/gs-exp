#!/usr/bin/env python3
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

TOPS=(10,20,30,40,50);DEPTHS=(5,8,10,12);RATIOS=(5,10,25,50)
EXCLUDED_RANGES=((108,173),(181,183),(219,279),(281,282))

def parse_args():
    p=argparse.ArgumentParser(description="Build an 80-variant SLIC foreground-removal ablation grid.")
    p.add_argument("--source",required=True,type=Path);p.add_argument("--output",type=Path);return p.parse_args()

def variant_name(top,depth,ratio):return "top{:02d}_depth{:02d}m_ratio{:02d}".format(top,depth,ratio)

def excluded(stem):
    index=int(stem);return any(start<=index<=end for start,end in EXCLUDED_RANGES)

def contact_sheet(preview,output):
    paths=sorted(preview.glob("*.png"));tw,th,lh,cols=240,134,20,5;rows=math.ceil(len(paths)/cols);canvas=Image.new("RGB",(tw*cols,(th+lh)*rows),"white");draw=ImageDraw.Draw(canvas)
    for i,path in enumerate(paths):
        image=Image.open(path).convert("RGB");image.thumbnail((tw,th),Image.Resampling.LANCZOS);x=(i%cols)*tw;y=(i//cols)*(th+lh);tile=Image.new("RGB",(tw,th),"black");tile.paste(image,((tw-image.width)//2,(th-image.height)//2));canvas.paste(tile,(x,y+lh));draw.text((x+4,y+3),path.name,fill="black")
    canvas.save(output,quality=90)

def main():
    args=parse_args();source=args.source.resolve();output=(args.output or source/"compensation/slic_based").resolve()
    if output.exists() and any(output.iterdir()):raise RuntimeError("output must be empty: {}".format(output))
    rgb_dir=source/"compensation/images_artifact_safe";preview_rgb_dir=rgb_dir;rel_dir=source/"depths_relative_da3mono";metric_dir=source/"depths_metric";slic_dir=source/"slic/labels"
    all_names=sorted(p.name for p in rgb_dir.glob("*.png"));names=[name for name in all_names if not excluded(Path(name).stem)]
    if not names:raise RuntimeError("no compensated PNG images: {}".format(rgb_dir))
    variants={}
    for top in TOPS:
        for depth in DEPTHS:
            for ratio in RATIOS:
                name=variant_name(top,depth,ratio);root=output/name;(root/"masks").mkdir(parents=True);(root/"preview").mkdir();variants[(top,depth,ratio)]={"name":name,"root":root,"rows":[],"kept":0,"removed":0,"rejected_segments":0}
    for index,name in enumerate(names):
        stem=Path(name).stem;rgb=np.asarray(Image.open(preview_rgb_dir/name).convert("RGB"));rel=np.load(rel_dir/(stem+".npy"));metric=np.load(metric_dir/(stem+".npy"));labels=np.load(slic_dir/(stem+".npy")).astype(np.int64)
        shape=rgb.shape[:2]
        if rel.shape!=shape or metric.shape!=shape or labels.shape!=shape:raise ValueError("shape mismatch: {}".format(name))
        label_count=int(labels.max())+1;segment_size=np.bincount(labels.ravel(),minlength=label_count);rel_valid=np.isfinite(rel)&(rel!=0);rel_values=rel[rel_valid]
        if not rel_values.size:raise RuntimeError("no valid relative depth: {}".format(name))
        relative_bad={top:rel_valid&(rel<np.percentile(rel_values,100-top)) for top in TOPS};metric_bad={depth:np.isfinite(metric)&(metric>0)&(metric<=float(depth)) for depth in DEPTHS}
        for top in TOPS:
            for depth in DEPTHS:
                bad_pixels=relative_bad[top]|metric_bad[depth];bad_count=np.bincount(labels[bad_pixels],minlength=label_count)
                for ratio in RATIOS:
                    item=variants[(top,depth,ratio)];reject_segment=(bad_count/np.maximum(segment_size,1))>=(ratio/100.0);final_bad=reject_segment[labels];keep=~final_bad
                    Image.fromarray((keep*255).astype(np.uint8),"L").save(item["root"]/"masks"/name,compress_level=3)
                    if index%10==0:
                        baked=np.zeros_like(rgb);baked[keep]=rgb[keep];Image.fromarray(baked,"RGB").save(item["root"]/"preview"/name,compress_level=3)
                    kept=int(keep.sum());removed=int(final_bad.sum());segments=int(reject_segment.sum());item["kept"]+=kept;item["removed"]+=removed;item["rejected_segments"]+=segments
                    item["rows"].append({"image":name,"candidate_pixels":int(bad_pixels.sum()),"kept_pixels":kept,"removed_pixels":removed,"kept_fraction":kept/float(keep.size),"rejected_superpixels":segments,"total_superpixels":int(np.count_nonzero(segment_size))})
        if (index+1)%10==0 or index+1==len(names):print("SLIC grid {}/{}".format(index+1,len(names)),flush=True)
    summary={"source":str(source),"output":str(output),"rgb_source":str(rgb_dir),"definition":{"candidate_mask":"(valid relative depth outside farthest n_top%) OR (valid metric depth <= n_depth meters); saturation is completely ignored","superpixel_rule":"reject complete SLIC superpixel when at least n_ratio% of all its pixels are in candidate_mask","final_mask":"remove rejected SLIC superpixels only; no pixel-wise removal"},"hyperparameters":{"n_top":TOPS,"n_depth":DEPTHS,"n_ratio":RATIOS},"excluded_ranges":[list(x) for x in EXCLUDED_RANGES],"excluded_images":len(all_names)-len(names),"image_count":len(names),"variants":{}}
    for key,item in variants.items():
        with (item["root"]/"image_statistics.csv").open("w",newline="",encoding="utf8") as stream:
            writer=csv.DictWriter(stream,fieldnames=item["rows"][0].keys());writer.writeheader();writer.writerows(item["rows"])
        contact_sheet(item["root"]/"preview",item["root"]/"preview_contact_sheet.jpg")
        total=item["kept"]+item["removed"];record={"n_top":key[0],"n_depth":key[1],"n_ratio":key[2],"kept_pixels":item["kept"],"removed_pixels":item["removed"],"kept_fraction":item["kept"]/float(total),"sum_rejected_superpixels_per_image":item["rejected_segments"]};(item["root"]/"manifest.json").write_text(json.dumps(record,indent=2),encoding="utf8");summary["variants"][item["name"]]=record
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    with (output/"summary.csv").open("w",newline="",encoding="utf8") as stream:
        rows=list(summary["variants"].values());writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    print("completed {} variants at {}".format(len(variants),output))

if __name__=="__main__":main()
