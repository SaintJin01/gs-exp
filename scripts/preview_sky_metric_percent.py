"""Preview only: sky OR ((metric >= n) AND/OR farthest m% per image)."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from preview_background_masks import tile, sheet, write_image


def make_masks(relative, metric, sky, thresholds, percents):
    # Compute relative quantiles on its native grid so upsampling cannot bias ranks.
    small_sky = ~np.isfinite(relative) | (relative <= 0)
    rv = np.isfinite(relative)&(relative>0)&~small_sky
    mv = np.isfinite(metric)&(metric>0)
    far = {}
    cuts = {}
    for m in percents:
        cutoff = float(np.quantile(relative[rv],1-m/100.)) if rv.any() else float('inf')
        cuts[m] = cutoff if np.isfinite(cutoff) else None
        small = rv & (relative>=cutoff)
        far[m] = cv2.resize(small.astype(np.uint8),(metric.shape[1],metric.shape[0]),interpolation=cv2.INTER_NEAREST)>0
    for mode in ('and','or'):
        for n in thresholds:
            metric_far = mv & (metric>=n)
            for m in percents:
                combined = metric_far & far[m] if mode=='and' else metric_far | far[m]
                mask = sky | combined
                yield 'sky_n{:g}_m{:g}_{}'.format(n,m,mode),mode,n,m,mask,cuts[m]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--n',nargs='+',type=float,default=[3,5,10])
    parser.add_argument('--m',nargs='+',type=float,default=[10,20,30])
    parser.add_argument('--samples',type=int,default=12)
    args=parser.parse_args()
    if any(not np.isfinite(x) or x<=0 for x in args.n) or any(not 0<x<100 for x in args.m):
        parser.error('Require finite n > 0 and 0 < m < 100')
    cv2.setNumThreads(1)
    root=Path(args.source);out=Path(args.output);dest=out/'sky_metric_percent'
    dest.mkdir(parents=True,exist_ok=True)
    files=sorted((root/'depths').glob('*.npy'))
    if not files: raise ValueError('No relative depth files')
    sample_ids=set(np.linspace(0,len(files)-1,min(args.samples,len(files))).round().astype(int))
    contacts={};stats=[];views=[]
    for i,path in enumerate(files):
        relative=np.load(str(path),allow_pickle=False)
        metric=np.load(str(root/'depths_metric'/path.name),allow_pickle=False)
        if relative.ndim!=2 or metric.ndim!=2:
            raise ValueError('Expected HxW depth maps: '+path.stem)
        # Dataset sky is encoded by invalid values in depths, not a sky PNG.
        sky_native = ~np.isfinite(relative) | (relative <= 0)
        sky = cv2.resize(sky_native.astype(np.uint8),
                        (metric.shape[1],metric.shape[0]),interpolation=cv2.INTER_NEAREST)>0
        rgb=None;grids={'and':[],'or':[]}
        if i in sample_ids:
            image_path=next((root/'images'/(path.stem+s) for s in ('.jpg','.png','.JPG') if (root/'images'/(path.stem+s)).is_file()),None)
            if image_path is None: raise FileNotFoundError(path.stem)
            rgb=cv2.imread(str(image_path))
            if rgb is None: raise ValueError(str(image_path))
            rgb=cv2.resize(rgb,(512,round(512*metric.shape[0]/metric.shape[1])),interpolation=cv2.INTER_AREA)
            small_sky=cv2.resize(sky.astype(np.uint8),(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST)>0
            overlay=rgb.copy();overlay[small_sky]=(overlay[small_sky]*.4+np.array([40,230,40])*.6).astype(np.uint8)
            write_image(dest/'views'/(path.stem+'_source.jpg'),sheet([tile(rgb,'Original '+path.stem),tile(overlay,'Depth invalid = sky | {:.1f}%'.format(100*sky.mean()))],2))
        for name,mode,n,m,mask,cut in make_masks(relative,metric,sky,args.n,args.m):
            mask_path=dest/'masks'/name/(path.stem+'.png');mask_path.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(mask_path),mask.astype(np.uint8)*255,[cv2.IMWRITE_PNG_COMPRESSION,1]):
                raise OSError(str(mask_path))
            # Invalid-depth sky is upsampled by nearest neighbor and must never be excluded.
            if not mask[sky].all(): raise AssertionError('Sky exclusion: '+name)
            coverage=float(mask.mean())
            stats.append(dict(image=path.stem,method=name,mode=mode,n=n,m=m,relative_cutoff=cut,
                              sky_coverage=float(sky.mean()),coverage=coverage))
            if rgb is not None:
                small=cv2.resize(mask.astype(np.uint8),(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST)>0
                overlay=rgb.copy();overlay[small]=(overlay[small]*.4+np.array([40,230,40])*.6).astype(np.uint8)
                grids[mode].append(tile(overlay,'n={:g}, m={:g}% | {:.1f}%'.format(n,m,100*coverage)))
                contacts.setdefault(name,[]).append(tile(overlay,'{} | {:.1f}%'.format(path.stem,100*coverage)))
        if rgb is not None:
            for mode in grids: write_image(dest/'views'/(path.stem+'_'+mode+'.jpg'),sheet(grids[mode],len(args.m)))
            views.append(path.stem)
        if (i+1)%25==0: print('Sky+metric+percent: {}/{}'.format(i+1,len(files)),flush=True)
    for name,tiles in contacts.items(): write_image(dest/'methods'/(name+'.jpg'),sheet(tiles,4))
    with (dest/'coverage.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stats[0]));writer.writeheader();writer.writerows(stats)
    (dest/'manifest.json').write_text(json.dumps(dict(n=args.n,m=args.m,views=views,count=len(files),
        sky='depths/<stem>.npy: nonfinite or <= 0; nearest resize to metric resolution; always included',
        relative_percent='farthest m percent of valid non-sky native relative-depth pixels, per image',
        metric='raw depths_metric >= n; no physical unit calibration assumed',
        formulas={'and':'sky | (metric_far & relative_far)','or':'sky | (metric_far | relative_far)'},
        training='preview only; training pipeline unchanged'),indent=2))
    page=['<!doctype html><meta charset="utf-8"><title>Sky + metric n + relative m%</title>',
          '<style>body{background:#181818;color:#eee;font:16px sans-serif;margin:24px;line-height:1.6}img{max-width:100%}select,button{font-size:17px;padding:8px;margin:5px;background:#333;color:white;border:1px solid #777}a{color:#8bd0ff}section{margin:30px 0}.sticky{position:sticky;top:0;background:#181818;padding:10px;border-bottom:1px solid #555}</style>',
          '<h1>Sky 항상 포함 + metric ≥ n + 이미지별 먼 m%</h1>',
          '<p>초록색 = 배경. depths의 invalid 값(이 데이터의 -1, 그 외 0 이하·비유한 값)을 sky로 사용합니다. Sky PNG는 사용하지 않으며, invalid 영역은 모든 후보에 항상 포함합니다.</p>',
          '<p>m%는 각 이미지의 <b>하늘을 제외한 유효 일반 depth</b> 중 가장 먼 비율입니다. n은 metric 원본 값 기준이며 단위 보정을 추가하지 않았습니다. 무효 depth는 하늘인 경우를 제외하고 해당 depth 조건을 충족하지 않습니다.</p>',
          '<p><b>AND:</b> sky ∪ [(metric ≥ n) ∩ (일반 depth 먼 m%)]<br><b>OR:</b> sky ∪ [(metric ≥ n) ∪ (일반 depth 먼 m%)]</p>',
          '<p>격자에서 행은 n='+str(args.n)+', 열은 m='+str(args.m)+'입니다. metric 해상도로 (depths의 invalid 영역은 nearest 확대) '+str(len(files))+'장의 binary mask를 sky_metric_percent/masks/에 저장했습니다. 학습 옵션과 파이프라인은 바꾸지 않았습니다.</p>',
          '<div class="sticky"><button onclick="move(-1)">이전 뷰</button><select id="view">'+''.join('<option>'+v+'</option>' for v in views)+'</select><button onclick="move(1)">다음 뷰</button><select id="mode"><option value="and">AND: 두 조건 모두</option><option value="or">OR: 한 조건 이상</option></select></div>',
          '<h2 id="label"></h2><img id="source"><br><a id="full" target="_blank"><img id="grid"></a>',
          '<h2>조합별 전체 샘플 비교</h2>']
    for name in contacts:
        page.append('<details><summary>'+name+'</summary><a href="sky_metric_percent/methods/'+name+'.jpg"><img loading="lazy" src="sky_metric_percent/methods/'+name+'.jpg"></a></details>')
    page.append('''<script>
const v=document.getElementById('view'), m=document.getElementById('mode');
function update(){let base='sky_metric_percent/views/'+v.value;document.getElementById('source').src=base+'_source.jpg';document.getElementById('grid').src=base+'_'+m.value+'.jpg';document.getElementById('full').href=base+'_'+m.value+'.jpg';document.getElementById('label').textContent=v.value+' / '+m.value.toUpperCase();}
function move(delta){v.selectedIndex=(v.selectedIndex+delta+v.options.length)%v.options.length;update();}
v.onchange=update;m.onchange=update;update();
</script>''')
    (out/'index.html').write_text('\n'.join(page),encoding='utf-8')
    print('Saved {} combinations x {} masks; all sky pixels included'.format(2*len(args.n)*len(args.m),len(files)),flush=True)


if __name__=='__main__': main()
