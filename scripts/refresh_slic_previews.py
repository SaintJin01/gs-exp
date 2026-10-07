#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from build_slic_grid import contact_sheet

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--source",required=True,type=Path);args=parser.parse_args();source=args.source.resolve();root=source/"compensation/slic_based";names=sorted(p.name for p in (source/"compensation/images").glob("*.png"))[::10]
    for index,variant in enumerate(sorted(p for p in root.iterdir() if p.is_dir())):
        preview=variant/"preview";preview.mkdir(exist_ok=True)
        for name in names:
            stem=Path(name).stem;rgb=np.asarray(Image.open(source/"images"/(stem+".jpg")).convert("RGB"));keep=np.asarray(Image.open(variant/"masks"/name).convert("L"))>0;baked=np.zeros_like(rgb);baked[keep]=rgb[keep];Image.fromarray(baked,"RGB").save(preview/name,compress_level=3)
        contact_sheet(preview,variant/"preview_contact_sheet.jpg")
        if (index+1)%10==0:print("preview variants {}/80".format(index+1),flush=True)

if __name__=="__main__":main()
