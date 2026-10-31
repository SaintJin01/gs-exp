"""Generate depth mask alternatives, full-resolution masks and HTML contact sheets."""
import argparse
import csv
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.background_masks import MIXED_METHODS, make_background_mask, make_mixed_masks


def tile(image, title, width=360):
    h = round(image.shape[0]*width/image.shape[1])
    out = np.zeros((h+38,width,3), dtype=np.uint8)
    out[38:] = cv2.resize(image,(width,h),interpolation=cv2.INTER_AREA)
    cv2.putText(out,title,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
    return out


def sheet(tiles, columns):
    h,w = tiles[0].shape[:2]
    out = np.full((((len(tiles)+columns-1)//columns)*h,columns*w,3),25,dtype=np.uint8)
    for i,t in enumerate(tiles):
        out[(i//columns)*h:(i//columns+1)*h,(i%columns)*w:(i%columns+1)*w] = cv2.resize(t,(w,h))
    return out


def write_image(path, image):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),image):
        raise OSError('Could not write '+str(path))


def depth_color(depth):
    valid = np.isfinite(depth) & (depth > 0)
    vis = np.zeros(depth.shape, np.uint8)
    if valid.any():
        logs = np.log(depth[valid]); lo,hi = np.percentile(logs,[1,99])
        vis[valid] = np.clip((logs-lo)/max(hi-lo,1e-8)*255,0,255).astype(np.uint8)
    color = cv2.applyColorMap(vis,cv2.COLORMAP_TURBO)
    color[~valid] = [255,0,255]
    return color


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--samples',type=int,default=12)
    parser.add_argument('--metric-depths',default='depths_metric')
    args=parser.parse_args()
    root=Path(args.source);out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    depths=sorted((root/'depths').glob('*.npy'))
    if not depths: raise ValueError('No .npy depth maps found')
    samples=set(np.linspace(0,len(depths)-1,min(args.samples,len(depths))).round().astype(int).tolist())
    methods = dict(baseline_relative='depths only: invalid OR farthest 10%',
                   baseline_metric='depths_metric only: invalid OR farthest 10%')
    methods.update(MIXED_METHODS)
    contacts={m:[] for m in methods};views=[];stats=[];scales=[]
    for i,path in enumerate(depths):
        depth=np.load(str(path),allow_pickle=False)
        metric_path=root/args.metric_depths/path.name
        metric_native=np.load(str(metric_path),allow_pickle=False)
        if depth.ndim != 2 or metric_native.ndim != 2:
            raise ValueError('Expected HxW depth: '+str(path))
        # Nearest sampling avoids mixing negative invalid pixels into valid values.
        metric=cv2.resize(metric_native.astype(np.float32),(depth.shape[1],depth.shape[0]),interpolation=cv2.INTER_NEAREST)
        valid=np.isfinite(depth)&np.isfinite(metric)&(depth>0)&(metric>0)
        scales.append(dict(image=path.stem,relative_shape=list(depth.shape),metric_shape=list(metric_native.shape),
                           metric_per_relative_median=float(np.median(metric[valid]/depth[valid])) if valid.any() else None))
        masks=make_mixed_masks(depth,metric)
        masks['baseline_relative']=make_background_mask(depth,'invalid_far_q90')
        masks['baseline_metric']=make_background_mask(metric,'invalid_far_q90')
        rgb=None
        if i in samples:
            image_path=next((root/'images'/(path.stem+s) for s in ('.jpg','.png','.JPG') if (root/'images'/(path.stem+s)).is_file()),None)
            if image_path is None: raise FileNotFoundError(path.stem)
            rgb=cv2.imread(str(image_path))
            if rgb is None: raise ValueError(str(image_path))
            rgb=cv2.resize(rgb,(depth.shape[1],depth.shape[0]),interpolation=cv2.INTER_AREA)
            view_tiles=[tile(rgb,'Original '+path.stem),tile(depth_color(depth),'depths / log depth'),
                        tile(depth_color(metric),'depths_metric / log depth')]
        for method in methods:
            mask=masks[method]
            write_image(out/'mixed'/'masks'/method/(path.stem+'.png'),mask.astype(np.uint8)*255)
            coverage=float(mask.mean())
            stats.append(dict(image=path.stem,method=method,coverage=coverage))
            if rgb is not None:
                overlay=rgb.copy()
                overlay[mask]=(overlay[mask].astype(np.float32)*.4+np.array([40,230,40])*.6).astype(np.uint8)
                t=tile(overlay,'{} | {:.1f}%'.format(method,100*coverage))
                view_tiles.append(t)
                contacts[method].append(tile(overlay,'{} | {:.1f}%'.format(path.stem,100*coverage)))
        if rgb is not None:
            rel='mixed/views/'+path.stem+'.jpg';write_image(out/rel,sheet(view_tiles,4));views.append(rel)
        if (i+1)%50==0: print('Mixed masks: {}/{}'.format(i+1,len(depths)),flush=True)
    for method,tiles in contacts.items(): write_image(out/'mixed'/'methods'/(method+'.jpg'),sheet(tiles,4))
    with (out/'mixed'/'coverage.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['image','method','coverage']);writer.writeheader();writer.writerows(stats)
    (out/'mixed'/'manifest.json').write_text(json.dumps(dict(source=str(root),relative='depths',metric=args.metric_depths,
        methods=methods,samples=views,image_count=len(depths),alignment=scales,
        mask_resolution='depths native resolution; metric resized by nearest neighbor',
        invalid_policy='Negative, zero, or nonfinite are unknown; consensus-invalid is a background hypothesis'),indent=2))
    descriptions = {
        'baseline_relative':'기존 depths만 사용: 무효값 + 유효값 중 가장 먼 10%.',
        'baseline_metric':'depths_metric만 사용: 무효값 + 유효값 중 가장 먼 10%.',
        'mix_invalid_consensus':'두 depth가 모두 무효값인 픽셀만 선택. 하늘 후보 기준.',
        'mix_far_intersection':'두 depth 모두 먼 20%에 포함될 때 선택. 두 값 모두 무효인 영역도 포함.',
        'mix_far_union':'어느 한 depth라도 먼 10%이면 선택. 두 값 모두 무효인 영역도 포함.',
        'mix_metric_gate':'depths의 무효값/먼 20% 후보에서 metric의 가까운 70%를 제외.',
        'mix_relative_gate':'metric의 무효값/먼 20% 후보에서 depths의 가까운 70%를 제외.',
        'mix_rank_mean':'두 거리 순위의 평균이 80% 이상. 한쪽만 유효하면 그쪽 순위 90% 이상.',
        'mix_rank_metric70':'metric 순위 70% + depths 순위 30%로 결합. 합산 순위 80% 이상.',
        'mix_log_aligned':'겹치는 유효 픽셀의 비율 중앙값으로 depths 스케일을 metric에 맞춘 뒤 기하평균. 가장 먼 15% 선택.',
        'mix_otsu_agree':'각 depth의 log-Otsu가 모두 먼 영역으로 분류한 곳만 선택.',
        'mix_metric_ge5':'두 depth가 모두 무효이거나, metric 원본 값이 5 이상.',
        'mix_metric_ge10':'두 depth가 모두 무효이거나, metric 원본 값이 10 이상.',
        'mix_top_connected':'평균 순위 기준의 후보 중 영상 상단에 연결된 영역만 선택.'}
    text=['<!doctype html><meta charset="utf-8"><title>Mixed depth background masks</title>',
          '<style>body{background:#181818;color:#eee;font:16px sans-serif;margin:28px;line-height:1.6}img{max-width:100%}a{color:#87ccff}section{margin:35px 0}nav a{display:inline-block;margin:5px 14px 5px 0}</style>',
          '<h1>depths + depths_metric 배경 마스크 비교</h1>',
          '<p>초록색 = 선택한 배경. depth 그림의 자홍색 = 무효값. 각 depth 그림은 독립적으로 색을 정규화했으므로 색이 같아도 같은 거리가 아닙니다.</p>',
          '<p>배경 Gaussian 학습 파이프라인은 유지합니다. 마스크 생성·비교용 학습 옵션만 제거했습니다. 아래 혼합 기준은 미리보기 전용이며 아직 학습 마스크로 선택하지 않았습니다.</p>',
          '<p>depths(512×288) 좌표에 metric(1959×1090)을 nearest 방식으로 맞췄습니다. 공통 영상 좌표를 가정하며 별도 기하 보정은 하지 않았습니다. 두 depth 모두 큰 값이 먼 거리인 것으로 해석합니다. 무효값을 배경으로 보는 것은 실험 가설입니다.</p>',
          '<p>두 단독 기준 + 혼합 기준 12개. 전체 {}개 뷰의 마스크는 mixed/masks/에 저장했습니다. 기존 단독 기준 결과 폴더도 남겨 두었습니다.</p>'.format(len(depths)),
          '<nav>'+''.join('<a href="#{}">{}</a>'.format(m,m) for m in methods)+'</nav><h2>기준별 비교</h2>']
    for method in methods:
        text.append('<section id="{}"><h3>{}</h3><p>{}</p><a href="mixed/methods/{}.jpg"><img loading="lazy" src="mixed/methods/{}.jpg"></a></section>'.format(method,method,html.escape(descriptions[method]),method,method))
    text.append('<h2>뷰별 모든 기준</h2>')
    for view in views: text.append('<section><h3>{}</h3><a href="{}"><img loading="lazy" src="{}"></a></section>'.format(Path(view).stem,view,view))
    (out/'index.html').write_text('\n'.join(text),encoding='utf-8')
    print('Saved {} methods x {} masks, {} sample comparisons to {}'.format(len(methods),len(depths),len(views),out))


if __name__=='__main__': main()
