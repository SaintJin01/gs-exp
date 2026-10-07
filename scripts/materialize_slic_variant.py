#!/usr/bin/env python3
import argparse, json, shutil, time
from pathlib import Path
import numpy as np
from PIL import Image

EXCLUDED_RANGES=((108,173),(181,183),(219,279),(281,282))

def excluded(path):
    index=int(path.stem);return any(start<=index<=end for start,end in EXCLUDED_RANGES)

def main():
    parser=argparse.ArgumentParser(description="Materialize one slic_based variant as compensation/images and compensation/depth_masked.")
    parser.add_argument("--source",required=True,type=Path);parser.add_argument("--variant",default="top50_depth05m_ratio50");args=parser.parse_args();source=args.source.resolve();comp=source/"compensation";variant=comp/"slic_based"/args.variant;artifact=comp/"images_artifact_safe"
    masks=[path for path in sorted((variant/"masks").glob("*.png")) if not excluded(path)]
    if not masks:raise RuntimeError("no masks: {}".format(variant/"masks"))
    token=str(int(time.time()));images_tmp=comp/(".images_slic_tmp_"+token);depth_tmp=comp/(".depth_masked_slic_tmp_"+token);images_tmp.mkdir();depth_tmp.mkdir();records=[]
    try:
        for index,mask_path in enumerate(masks):
            source_image=artifact/mask_path.name
            if not source_image.is_file():raise FileNotFoundError(source_image)
            rgb=np.asarray(Image.open(source_image).convert("RGB"));mask=np.asarray(Image.open(mask_path).convert("L"))
            if rgb.shape[:2]!=mask.shape:raise ValueError("shape mismatch: {}".format(mask_path.name))
            shutil.copy2(source_image,images_tmp/mask_path.name);rgba=np.dstack((rgb,mask));Image.fromarray(rgba.astype(np.uint8),"RGBA").save(depth_tmp/mask_path.name,compress_level=3)
            records.append({"image":mask_path.name,"active_pixels":int(np.count_nonzero(mask)),"total_pixels":int(mask.size)})
            if (index+1)%20==0 or index+1==len(masks):print("materialized {}/{}".format(index+1,len(masks)),flush=True)
        manifest={"variant":args.variant,"variant_root":str(variant),"rgb_source":str(artifact),"image_count":len(masks),"excluded_images":301-len(masks),"active_pixels":sum(x["active_pixels"] for x in records),"total_pixels":sum(x["total_pixels"] for x in records),"records":records}
        (images_tmp/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8");(depth_tmp/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
        backups=[]
        for name,tmp in (("images",images_tmp),("depth_masked",depth_tmp)):
            destination=comp/name
            if destination.exists():
                backup=comp/(name+"_before_"+args.variant+"_"+token);destination.rename(backup);backups.append(str(backup))
            tmp.rename(destination)
        print(json.dumps({"images":str(comp/"images"),"depth_masked":str(comp/"depth_masked"),"backups":backups,"image_count":len(masks)},indent=2))
    except Exception:
        shutil.rmtree(images_tmp,ignore_errors=True);shutil.rmtree(depth_tmp,ignore_errors=True);raise

if __name__=="__main__":main()
