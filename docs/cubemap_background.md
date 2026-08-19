# Cubemap 배경 파이프라인: 생성부터 3DGS 학습 통합까지

이 문서는 두 부분으로 구성된다.

1. **Cubemap 생성** — `~/work/environment_map_baking`가 촬영 이미지 시퀀스에서 환경맵 cubemap을 굽는 **순서**
2. **렌더 파이프라인 수정** — 그 cubemap을 3DGS(gs-exp) rasterizer 안에서 "가장 뒤 배경"으로 깔고 학습/렌더하는 방법

**현재 방식은 direction-only 환경맵이다.** cubemap을 무한 원경(skybox)으로 보고 픽셀 레이의 **방향으로만** 조회한다. per-texel height/parallax 보정 경로는 렌더러·베이킹 양쪽에서 제거되었다(복원용 요약은 [부록 A](#부록-a--제거된-heightparallax-경로-복원-시-참고)).

기준 데이터셋은 Tanks & Temples *train* (`/home/sj/work/gs-dataset/tandt/train`)이며, cubemap은 `<source>/cubemaps/cubemap/`에 둔다(= `train_cubemap_bg.py`/`render_cubemap_bg.py`의 `--cubemap` 기본 위치).

---

## Part 1 — Cubemap 생성 (environment_map_baking)

### 1.1 생성 순서

```
COLMAP sparse ─▶ (1) poses.json ─▶ (2) DA3 per-image depth (.npy, sky = -1)
                                            │
   images/ + poses.json + depths/ ─▶ (3) envmap-bake ─▶ 6면 PNG + distance.npy + meta.json
                                            │              (배경 분리 → rotation-only 투영
                                            │               → median color 누적 → seam blending)
                                            └─▶ (4) distance.npy 삭제 → color-only cubemap
```

```bash
conda activate envmap-da3
DS=/home/sj/work/gs-dataset/tandt/train

# (1) COLMAP sparse → poses.json  (undistorted PINHOLE intrinsics + 카메라 축 변환)
envmap-colmap-poses --sparse $DS/sparse/0 --out $DS/poses.json

# (2) Depth Anything 3로 per-image depth  (sky = -1 sentinel)
envmap-da3-depths \
  --images $DS/images --poses $DS/poses.json --out $DS/depths \
  --device cuda --process-res 378 --window-size 24 --overlap 2

# (3) 베이킹 — images/(undistorted) + rotation-only + seam-blend
envmap-bake \
  --images $DS/images --poses $DS/poses.json --depths $DS/depths \
  --out $DS/cubemaps/cubemap \
  --face-size 1024 --projection pinhole \
  --background-depth-factor 4 \
  --min-samples 2 --rotation-only --seam-blend
  # 블렌딩: 텍셀당 최대 5프레임 reservoir의 median, 가중치 없음(모든 배경 픽셀 동등).

# (4) height 제거 — direction-only 렌더는 distance를 안 쓰므로 color-only로 정리
#     (envmap-bake는 rotation-only에서도 distance.npy를 남기므로 여기서 지운다)
rm -f $DS/cubemaps/cubemap/distance.npy
python - "$DS/cubemaps/cubemap/meta.json" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]); m = json.loads(p.read_text())
m.pop("distance_map", None); m.pop("distance_convention", None)
p.write_text(json.dumps(m, indent=2))
PY
```

두 가지 필수 선택(아래 §1.5·§1.6에서 이유 상술):

> **① `--images`는 반드시 `images/`(COLMAP undistort 결과) — `input/`(왜곡 원본)이 아니다.**
> `poses.json`의 intrinsics는 `sparse/0`의 undistorted **PINHOLE**이다. `input/`은 렌즈
> 왜곡(k1,k2)과 다른 principal point를 가진 **OPENCV 원본**이라, 이걸 PINHOLE intrinsics로
> 구우면 텍셀 색이 잘못된 방향에 배치된다. gs-exp가 `-i images`로 학습/렌더하므로 **반드시
> 같은 `images/`로** 구워야 한다. (T&T train 실측: `input` 30px vs `images` 5px 오정렬.
> 왜곡·해상도 갭이 클수록 오차가 커진다.)

> **② `--rotation-only`로 굽는다.** direction-only 렌더와 기하적으로 일관된 투영이다
> (카메라 회전만으로 방향 매핑). depth-warp/`--heightfield-warp`는 제거된 parallax 경로용이며
> direction-only 렌더와 섞으면 오히려 정렬이 나빠진다([부록 A](#부록-a--제거된-heightparallax-경로-복원-시-참고)).

### 1.2 좌표 규약

- **World**: COLMAP world를 그대로 쓴다. `poses.json`은 **카메라 축 규약만** 바꾼다
  (COLMAP `+X right, +Y down, +Z forward` → 베이커 `+X right, +Y up, +Z forward`).
  World 자체는 불변이므로 **gs-exp의 world와 동일**하다 (`envmap_baking/colmap.py`):

  ```python
  # COLMAP camera: +X right, +Y down, +Z forward.  Baker: +X right, +Y up, +Z forward.
  y_up_to_colmap_camera = np.diag([1.0, -1.0, 1.0])
  camera_to_world[:3, :3] = colmap_camera_to_world @ y_up_to_colmap_camera
  ```

- **Cubemap 공간**: 카메라 평균 up 벡터를 `+Y`로 맞추는 회전 `world_to_cubemap`을 world에
  적용한 공간이다. 이 회전이 `meta.json`에 저장되고, 소비자는 항상
  `dir_cube = world_to_cubemap @ dir_world`로 방향을 바꾼 뒤 face를 조회한다.
  **`meta.json`이 없으면 방향을 복원할 수 없다.**

### 1.3 Depth 규약 (step 2 산출물)

DA3 depth는 `.npy`로 저장된다.

- **sky 픽셀 = `-1` sentinel** — 베이커는 `depth < 0`을 "무한 거리 배경"으로 취급한다.
- **저해상도 허용** — 이미지보다 작으면 로드 시 **nearest** 리사이즈한다(bilinear는 `-1`과
  유효 depth를 섞어 경계에 가짜 중간값을 만든다, `envmap_baking/io.py`).

### 1.4 배경 분리 (step 3 내부)

배경 텍셀만 cubemap에 남긴다 (`envmap_baking/pipeline.py`):

```python
threshold = max(trajectory_radius * depth_factor, 1e-6)
sky  = np.isfinite(depth) & (depth < 0)              # sky는 항상 배경
mask = (np.isfinite(depth) & (depth >= threshold)) | sky
```

- `trajectory_radius` = 궤적 중심에서 카메라 거리의 95퍼센타일 (T&T train ≈ 5.37).
- `--background-depth-factor 4` → threshold ≈ 21.5 (depth 상위 ~6%만 통과). 3/4/5 중 4가
  "원경 위주 + 언덕 기슭 구조물 유지"의 균형점이었다.
- **rotation-only여도 depth는 필요하다** — 배경/전경을 가르는 데 쓰이기 때문. (색 투영 자체엔
  depth를 안 쓴다.)

### 1.5 투영과 누적 (step 3 내부)

- **투영 = rotation-only**: 배경 픽셀의 카메라 레이를 **pose 회전만으로** world 방향으로
  보낸다(depth-warp 없음). 텍셀 색 = "그 방향으로 본 색" → 무한 원경 direction-only 렌더와
  정확히 일치한다.
- **face/UV 매핑**: `envmap_baking/cubemap.py:directions_to_faces_uv` (major-axis, face 순서
  `posx negx posy negy posz negz`).
- **reservoir median (가중치 없음)**: 텍셀마다 그 방향을 처음 본 **최대 5개 프레임**의
  샘플을 유지하고 채널별 median을 취한다(`CubemapAccumulator`, `MEDIAN_RESERVOIR_SIZE=5`).
  가중치는 없다 — 모든 배경 픽셀이 동등하게 기여하고, 슬롯은 도착 순서대로 채워진다.
  `--min-samples 2`로 단일 샘플 텍셀은 버린다.
  (**왜 전-샘플이 아니라 reservoir인가**: 배경은 시차가 있어 같은 점이 프레임마다 다른
  텍셀에 떨어진다. 궤적 전체의 **모든** 샘플을 median하면 넓은 시차가 섞여 건물이 뭉개진다.
  시간적으로 가까운 5프레임으로 제한하면 시차 폭이 작아 선명하게 유지된다 — fountain
  실측에서 전-샘플 median은 확연히 흐렸다. 이전의 depth 신뢰도·cos^p 중심 **가중치**는 제거됨.)

### 1.6 Seam 처리: multi-band blending (`--seam-blend`)

median만으로는 프레임 간 노출차 때문에 하늘이 다각형 패치워크로 갈라진다. Hedman et al. 2017
방식의 2단계로 감춘다 (`envmap_baking/seam_blend.py`):

1. **픽셀당 단일 소스 라벨링** — median이 고른 샘플의 소스 프레임을 그 텍셀 라벨로 삼고,
   최근접 채움 + median 필터로 despeckle.
2. **라플라시안 피라미드 블렌딩** — 각 소스 레이어의 라플라시안 피라미드 × 라벨 마스크의
   가우시안 피라미드를 레벨별로 누적. 저주파(노출차)는 넓게, 고주파(디테일)는 seam 근처에서만
   전환 → 경계선·이중상이 사라진다. face 경계는 `--seam-padding`(기본 64px)만큼 이웃 면까지
   확장해 블렌딩 후 잘라내므로 면 사이 seam이 새로 생기지 않는다.

남은 한계: 프레임 간 노출차 자체는 하늘에 부드러운 얼룩으로 남는다(별도 exposure alignment 필요).

### 1.7 왜 step 4에서 distance를 지우나

`envmap-bake`는 항상 per-texel distance(`distance.npy`, height)를 함께 굽고 meta에
`distance_map`/`distance_convention`을 넣는다. 이는 제거된 parallax 렌더 경로의 입력이었다
([부록 A](#부록-a--제거된-heightparallax-경로-복원-시-참고)). **direction-only 렌더는 이걸
읽지 않으므로** 혼동을 막기 위해 삭제해 color-only cubemap으로 정리한다.

### 1.8 출력물

```
cubemaps/cubemap/
  posx.png negx.png posy.png negy.png posz.png negz.png   # 1024² faces (color만)
  coverage.png                                             # 커버리지 시트
  meta.json                                                # 아래 키
```

`meta.json` 키: `world_to_cubemap`(렌더러가 쓰는 유일한 값), `face_size`, `bake_origin`,
`background_radius`, `warp`("rotation"), `convention`. `bake_origin`/`background_radius`는
현재 렌더러가 쓰지 않는 잔여 정보다(parallax 복원 시에만 필요).

---

## Part 2 — 렌더 파이프라인 수정 (gs-exp)

배경을 별도 패스로 합성하지 않는다. **rasterizer가 픽셀별로** cubemap 배경을 채운다.

### 2.1 핵심 아이디어: rasterizer의 배경 합성 지점

3DGS forward는 픽셀별로 gaussians를 front-to-back 순회하며 투과율 `T`를 누적하고 마지막에
배경을 더한다:

```
C = Σᵢ cᵢ·αᵢ·Tᵢ + T_final · bg        (Tᵢ₊₁ = Tᵢ(1-αᵢ))
```

이건 "배경을 먼저 깔고 그 위에 alpha blending"과 수학적으로 동일하다. 원하는 합성 구조가 이미
커널 안에 있으므로, `bg`를 픽셀 공통 (3,) 상수에서 **per-pixel (3,H,W) 이미지로 확장**하는 것이
정확한 수정이다 — 별도 패스도, 근사도 없다.

### 2.2 CUDA 패치 (`submodules/diff-gaussian-rasterization`)

인덱싱 두 줄이 전부다.

`cuda_rasterizer/forward.cu` — 픽셀별 배경 합성:

```cuda
// before: out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch * H * W + pix_id];
```

`cuda_rasterizer/backward.cu` — 배경이 alpha gradient에 주는 기여항:

```cuda
float bg_dot_dpixel = 0;
for (int i = 0; i < C; i++)
    bg_dot_dpixel += bg_color[i * H * W + pix_id] * dL_dpixel[i];
dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
```

**하위 호환**: Python 래퍼(`diff_gaussian_rasterization/__init__.py`)가 (3,) 상수 bg를
(3,H,W)로 자동 broadcast하므로 기존 `train.py`/`render.py`는 그대로 동작한다.

```python
bg = raster_settings.bg
if bg.dim() == 1:
    bg = bg[:, None, None].expand(-1, raster_settings.image_height, raster_settings.image_width)
raster_settings = raster_settings._replace(bg=bg.contiguous())
```

CUDA 수정 후 재빌드:

```bash
conda activate gaussian_splatting
pip install ./submodules/diff-gaussian-rasterization --no-build-isolation
```

**동치 검증** (per-pixel bg 단일 패스 vs "black 렌더 + (1−α)·bg" 2패스):

| 항목 | 최대 차이 |
|---|---|
| forward | 3.6e-07 |
| grad means3D | 4.4e-04 (스케일 3.1e+02) |
| grad colors | 3.2e-05 (스케일 3.9e+01) |
| grad opacity | 7.6e-05 (스케일 5.3e+01) |
| 상수 bg broadcast | 0 (비트 동일) |

float atomic 누적 순서 차이 수준으로, 두 공식이 동치임을 확인했다.

### 2.3 Direction-only cubemap 샘플러 (`exp_utils/cubemap_bg.py`)

`CubemapBackground.background_for_view(view)`가 카메라 한 대의 모든 픽셀에 대해 cubemap을
평가해 (3,H,W) 배경을 만든다. 세 단계뿐이다:

```python
# 1) 픽셀 → 카메라 레이 (COLMAP 축: +X right, +Y down, +Z forward)
directions_cam = torch.stack([(xs + 0.5 - W*0.5)/fx,
                              (ys + 0.5 - H*0.5)/fy,
                              torch.ones_like(xs)], dim=-1)
# 2) 카메라 → world → cubemap 공간
#    world_view_transform은 W2C의 전치라 좌상단 3x3이 곧 camera-to-world 회전이다.
directions = directions_cam @ view.world_view_transform[:3, :3].T
directions = directions @ self.world_to_cubemap.T          # meta.json의 회전
# 3) 방향으로 bilinear 조회 (parallax/offset 없음 = 무한 원경)
return self._sample(self.faces, directions)
```

- face/UV 매핑(`_directions_to_faces_uv`)은 베이커의 `directions_to_faces_uv`를 그대로 포팅해
  규약을 맞춘다. `grid_sample(align_corners=False)`의 `gx = 2u−1`이 베이킹된 텍셀 센터와 정렬된다.
- 카메라 위치(`camera_center`)·`bake_origin`·`distance.npy`는 **쓰지 않는다.** 로드하는 meta
  값은 `world_to_cubemap`뿐이다.

### 2.4 학습 (`train_cubemap_bg.py`)

`train.py`와의 차이는 `[CUBEMAP]` 주석 지점뿐. 핵심은 렌더 호출:

```python
# [CUBEMAP] rasterizer가 각 픽셀의 남은 투과율을 그 레이의 cubemap 값으로 채운다.
bg = cubemap_bg(viewpoint_cam)
render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                    use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
```

- **단일 패스**: rasterizer가 합성·gradient를 모두 처리 → 일반 학습과 거의 같은 비용.
- **학습 효과**: GT가 배경인 픽셀의 loss가 cubemap 기준으로 계산되어, gaussians가 하늘/원경을
  재구성할 유인이 사라지고 그 위 전경 gaussian의 opacity가 눌린다(배경 항이 `dL_dalpha`에 들어감).
- **캐시**: 카메라별 배경은 최초 1회만 계산해 uint8 CPU 캐시(`CachedCubemapBackground`).
- `--random_background`는 경고 후 무시, `white_background`도 효과 없음. densification/pruning/
  depth 정규화/checkpoint 로직은 `train.py` 그대로.

```bash
conda activate gaussian_splatting
python train_cubemap_bg.py -s /home/sj/work/gs-dataset/tandt/train -m output/<name> --eval \
  --cubemap /home/sj/work/gs-dataset/tandt/train/cubemaps/cubemap
# --cubemap 기본값 = <source_path>/cubemaps/cubemap (위 경로면 생략 가능)
# loss map 덤프 지점: --loss_map_iterations 500 2500   (기본값; 비우면 끔)
```

### 2.5 렌더링 (`render_cubemap_bg.py`)

학습과 동일한 단일 패스 합성으로 학습-렌더 일관성을 유지하고, 원본 `render.py`와 같은 폴더
구조로 저장한다:

```bash
python render_cubemap_bg.py -m output/<name> --skip_train \
  --cubemap /home/sj/work/gs-dataset/tandt/train/cubemaps/cubemap
# 출력: <model>/{train,test}/ours_<iter>/renders/  및  gt/
```

`renders/`(합성 결과)와 `gt/`(원본)를 `{05d}.png`로 저장하므로 `metrics.py`가 기대하는
레이아웃과 일치한다. `--cubemap` 기본값은 `None`이 아니라 `""`이다(3DGS `get_combined_args`가
`None` CLI 값을 cfg 병합에서 떨어뜨리기 때문).

### 2.6 파일 요약

| 파일 | 역할 |
|---|---|
| `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu` | per-pixel bg 합성 (1줄) |
| `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu` | per-pixel bg alpha gradient |
| `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py` | (3,) bg 자동 broadcast |
| `exp_utils/cubemap_bg.py` | 레이 생성 + direction-only cubemap 샘플러 (`CubemapBackground`) |
| `train_cubemap_bg.py` | cubemap 배경 학습 (`--loss_map_iterations`로 loss map 덤프 지점) |
| `render_cubemap_bg.py` | cubemap 배경 렌더 (출력: `renders/` + `gt/`) |
| `<dataset>/cubemaps/cubemap/{6 faces,coverage,meta.json}` | color-only 환경맵 (distance 없음) |

### 2.7 알려진 한계

- **미관측 영역**: cubemap coverage 구멍(주로 −Y 바닥, 미촬영 방향)은 검은색으로 남는다.
  학습 중 그 픽셀은 gaussians가 채운다. hole filling(inpainting)은 베이킹 후속 과제.
- **노출차**: 프레임 간 노출차로 인한 하늘의 부드러운 밝기 얼룩은 seam blending이 아닌 exposure
  alignment의 몫.
- **parallax 없음**: direction-only는 배경을 무한 원경으로 가정한다. 배경이 가깝고 카메라
  이동이 큰 씬(예: DL3DV, 배경 ≈18m)은 뷰마다 `~offset/거리`만큼 배경이 어긋난다. 이걸 줄이려면
  [부록 A](#부록-a--제거된-heightparallax-경로-복원-시-참고)의 height/parallax 경로가 필요하다.
- `use_trained_exp`(train_test_exp) 사용 시 노출 보정이 합성 이후에 적용되어 배경에도 걸린다
  (기본값 False에서는 무관).

---

## 부록 A — 제거된 height/parallax 경로 (복원 시 참고)

배경이 가까운 씬에서 뷰별 시차를 보정하려면 아래 경로를 되살린다. 원리: cubemap을 bake origin
중심의 **radial height field** `r(û)`로 보고, 각 픽셀 레이 `o + t·d`와 그 표면의 교점 방향으로
색을 조회한다(카메라 offset `o`가 배경 거리 `r`보다 훨씬 작아 몇 번 반복이면 수렴).

되살릴 때 **베이킹·렌더를 짝으로** 바꿔야 한다(둘 중 하나만 바꾸면 정렬이 깨진다):

- **베이킹**: `--rotation-only` 대신 **`--heightfield-warp`**(2-pass consensus) 또는 depth-warp로
  굽고, step 4의 distance 삭제를 **생략**한다. 텍셀 색이 "bake origin에서 본 색"이 되고
  `distance.npy`(per-texel metric distance; `+inf`=sky, `NaN`=미커버, disparity=1/r로 누적)가 남는다.
- **렌더**: `cubemap_bg.py`에 per-texel parallax를 복원한다. `distance.npy`를 disparity로 로드하고
  (sky→0, hole→`1/background_radius`), 고정점 반복으로 레이-구 far-root 교차를 푼다:

  ```
  toward = d
  반복(8회):  r = 1/sample_disparity(toward);  r = max(r, 1.05·|o|)
             t = −b + √(b² − (|o|² − r²)),  b = d·o
             toward = normalize(o + t·d)
  ```

  distance가 없으면 단일 `background_radius` 구 교차(skydome), `bake_origin`도 없으면 direction-only로
  fallback. 반복 횟수는 스카이라인 불연속 텍셀 수렴을 위해 8이 적절했다(3회는 T&T train에서 배경
  텍셀 ~1.5%가 최대 30px 미수렴).

**경험적 결과**:
- 합성 씬(바닥 평면 + 방향성 sky, offset 3): parallax 23.1 dB vs skydome 11.7 dB vs direction-only 14.0 dB.
- T&T train(배경 ≈28m, 멀다): held-out 정렬에서 parallax의 이점이 작다(궤적 위 평가 + 원거리 배경).
- DL3DV(배경 ≈18m, 가깝다): direction-only 잔여 오차 ~10px. input/images 베이크 차이는 무의미했고
  (거의 동일 intrinsics), 이 오차는 시차라서 parallax 경로라야 실질적으로 줄어든다.
- **일관성 규칙**: depth-warp/heightfield 베이킹 ↔ parallax 렌더, rotation-only 베이킹 ↔ direction
  렌더. 교차 조합은 나쁘다.
