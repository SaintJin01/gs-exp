# Cubemap 배경 파이프라인: 생성부터 3DGS 학습 통합까지

이 문서는 두 부분으로 구성된다.

1. **Cubemap 생성** — `~/work/environment_map_baking` 프로젝트가 촬영 이미지 시퀀스에서 환경맵 cubemap을 굽는 방법
2. **렌더 파이프라인 수정** — 이 cubemap을 3DGS(gs-exp)의 rasterizer 안에서 "가장 뒤 배경"으로 깔고 학습/렌더하는 방법

기준 데이터셋은 Tanks & Temples *train* (`/home/sj/work/gs-dataset/tandt/train`)이며, 표준 cubemap은 `cubemaps/cubemap/`에 있다.

---

## Part 1 — Cubemap 생성 (environment_map_baking)

### 1.1 전체 흐름

```
COLMAP sparse ──▶ poses.json ──▶ DA3 per-image depth (.npy, sky = -1)
                                        │
images + poses + depths ──▶ envmap-bake ┴─▶ 6면 cubemap PNG + distance.npy + meta.json
                                             (배경 분리 → rotation-only 투영
                                              → median 누적 (color + 1/거리)
                                              → 라벨링 → multi-band seam blending)
```

실행 커맨드 (표준 설정):

```bash
conda activate envmap-da3

# 1. COLMAP sparse 모델에서 포즈 추출
envmap-colmap-poses \
  --sparse /home/sj/work/gs-dataset/tandt/train/sparse/0 \
  --out    /home/sj/work/gs-dataset/tandt/train/poses.json

# 2. Depth Anything 3로 per-image depth 생성 (sky = -1 sentinel)
envmap-da3-depths \
  --images /home/sj/work/gs-dataset/tandt/train/images \
  --poses  /home/sj/work/gs-dataset/tandt/train/poses.json \
  --out    /home/sj/work/gs-dataset/tandt/train/depths \
  --device cuda --process-res 378 --window-size 24 --overlap 2

# 3. Cubemap 베이킹 (multi-band seam blending 포함)
envmap-bake \
  --images /home/sj/work/gs-dataset/tandt/train/images \
  --poses  /home/sj/work/gs-dataset/tandt/train/poses.json \
  --depths /home/sj/work/gs-dataset/tandt/train/depths \
  --out    /home/sj/work/gs-dataset/tandt/train/cubemaps/cubemap \
  --face-size 1024 --projection pinhole \
  --background-depth-factor 4 --blend-mode median \
  --view-center-weight-power 4 --min-samples 2 \
  --seam-blend
```

> **반드시 `images/`(COLMAP undistort 결과)로 구울 것 — `input/`(왜곡 원본)이 아니라.**
> `poses.json`의 intrinsics는 `sparse/0`의 **PINHOLE(undistorted, 1959×1090, cx=979.5,
> cy=545, 왜곡계수 없음)** 이다. 반면 `input/`은 `distorted` 모델의 **OPENCV 원본
> (1920×1080, cx=960, cy=540, k1=-0.06, k2=0.04)** 이다. `input/`을 PINHOLE intrinsics로
> 구우면 렌즈 왜곡과 principal point가 무시되어 텍셀 색이 잘못된 방향에 배치되고, gs-exp는
> `images/`(undistorted)로 렌더하므로 배경이 **~30px 어긋난다**(T&T train 산 능선 실측:
> `input` 30px → `images` 5px). 이 오차는 parallax 보정으로 잡히지 않는다(시차가 아니라
> 베이킹 기하 오차이기 때문). 학습에 쓰는 `-i/--images`와 **같은 폴더**로 구워야 한다.

> **`--rotation-only`를 쓰지 말 것** (parallax 렌더링과 함께라면): per-texel distance
> parallax 보정(§2.4)은 "텍셀 = bake origin에서 본 색"이라는 central projection을
> 가정한다. depth-warp 투영(기본값)이 그 가정을 만족한다. rotation-only로 구우면
> 텍셀 색이 기여 카메라 위치 기준이라 보정이 오히려 정렬을 악화시킨다
> (실측: T&T train에서 best-shift 오정렬 17px vs direction-only 5px).

### 1.2 좌표 규약

- **World**: COLMAP world를 그대로 사용한다. `poses.json`은 카메라 축 규약만 바꾼다
  (COLMAP 카메라 `+X right, +Y down, +Z forward` → 베이커 카메라 `+X right, +Y up, +Z forward`).
  World 자체는 바뀌지 않으므로 **gs-exp의 world와 동일**하다.

  `envmap_baking/colmap.py`:

  ```python
  # COLMAP camera coordinates are +X right, +Y down, +Z forward.
  # The baker uses +X right, +Y up, +Z forward.
  y_up_to_colmap_camera = np.diag([1.0, -1.0, 1.0])
  camera_to_world[:3, :3] = colmap_camera_to_world @ y_up_to_colmap_camera
  ```

- **Cubemap 공간**: 카메라들의 평균 up 벡터를 `+Y`로 정렬하는 회전(`world_to_cubemap`)을
  world에 적용한 공간이다. 이 회전은 베이킹 결과와 함께 `meta.json`으로 저장된다
  (`envmap_baking/pipeline.py`):

  ```python
  meta = {
      "world_to_cubemap": world_to_cubemap.tolist(),
      "face_size": config.face_size,
      "convention": "dir_cubemap = world_to_cubemap @ dir_world; ...",
  }
  (config.output_dir / "meta.json").write_text(json.dumps(meta, indent=2), ...)
  ```

  **이 파일이 없으면 gs-exp 쪽에서 cubemap 방향을 복원할 수 없다.** 소비자는 항상
  `dir_cube = world_to_cubemap @ dir_world`로 방향을 변환한 뒤 face를 조회한다.

### 1.3 Depth 규약: sky = -1 sentinel

DA3 depth는 `.npy`로 저장되며 두 가지 특성이 있다.

- **sky 픽셀은 `-1`** — DA3의 sky mask가 가리키는 픽셀에 sentinel을 기록한다.
  베이커는 `depth < 0`을 "무한 거리 배경"으로 취급한다.
- **저해상도(288×512) 허용** — 이미지(1080×1920)보다 작으면 로드 시 **nearest** 리사이즈한다.
  bilinear를 쓰면 -1과 유효 depth가 섞여 경계에 잘못된 중간값이 생기기 때문이다
  (`envmap_baking/io.py`):

  ```python
  if depth.shape != shape:
      # Nearest keeps sentinel values (sky = -1) from bleeding into valid depths.
      resized = Image.fromarray(depth.astype(np.float32), mode="F").resize(
          (shape[1], shape[0]), resample=Image.Resampling.NEAREST)
  ```

### 1.4 배경 분리

픽셀이 배경으로 인정되는 조건 (`envmap_baking/pipeline.py`):

```python
threshold = max(trajectory_radius * depth_factor, 1e-6)
sky = np.isfinite(depth) & (depth < 0)
mask = np.isfinite(depth) & (depth >= threshold)
mask |= sky            # sky는 threshold와 무관하게 항상 배경
```

- `trajectory_radius`: 카메라 궤적 중심으로부터 각 카메라 거리의 95퍼센타일 (T&T train ≈ 5.37)
- `--background-depth-factor 4` → threshold ≈ **21.5** (depth 분포의 상위 ~6%만 통과)
- factor 3/4/5를 비교한 결과 4가 "원경 위주 + 언덕 기슭 구조물 유지"의 균형점이었다

sky 픽셀의 블렌딩 가중치는 무한 거리 극한값인 1.0으로 처리한다:

```python
confidence = depth / (depth + parallax_baseline)
# Sky pixels carry depth < 0 and are treated as infinitely far away.
confidence[np.isfinite(depth) & (depth < 0)] = 1.0
```

### 1.5 투영과 누적

- **투영 (기본값 = depth-warp)**: 배경 픽셀을 depth로 world 점으로 올린 뒤 bake origin
  기준 방향으로 재투영한다 (`world_directions_from_depth`). 텍셀이 "bake origin에서 본
  색"이 되어 per-texel distance parallax 보정(§2.4)과 기하적으로 일관된다.
  `--rotation-only`(카메라 회전만 사용)는 depth 노이즈는 피하지만 parallax 보정과
  결합하면 정렬이 깨지므로 direction-only 렌더링에서만 쓴다.
- 방향 → cubemap face/UV 매핑은 `envmap_baking/cubemap.py:directions_to_faces_uv`
  (major-axis 방식, face 순서 `posx negx posy negy posz negz`).
- **median 누적**: 텍셀마다 가중치 상위 5개 샘플을 reservoir로 유지하고 출력 시 median을
  취한다. 움직이는 전경 잔여물 같은 outlier가 제거된다. `--min-samples 2`로 단일 샘플
  텍셀은 버린다.
- 가중치 = depth 신뢰도 × 이미지 중심 가중치(`cos^4`) × 배경 마스크.

### 1.6 Seam 처리: multi-band blending (Casual 3D Photography 방식)

median 블렌딩만으로는 프레임 간 노출 차이 때문에 하늘이 다각형 패치워크로 갈라진다.
`--seam-blend`는 Hedman et al. 2017의 2단계 구조를 따른다.

**단계 A — 픽셀당 단일 소스 라벨링** (`envmap_baking/cubemap.py:label_maps`)

그래프컷 대신 "median이 고른 샘플의 라벨"을 쓴다. reservoir 슬롯마다 소스 프레임
인덱스를 저장해 두고, median 색에 가장 가까운 샘플의 프레임을 그 텍셀의 소스로 삼는다.
median의 outlier 제거 효과가 라벨링에 그대로 이어진다:

```python
colors = self.median_colors[select][:, :count].astype(np.int16)
median = np.median(colors, axis=1, keepdims=True)
distance = np.abs(colors - median).sum(axis=-1)
slots = np.argmin(distance, axis=1)
labels[select] = self.median_labels[select][np.arange(len(slots)), slots]
```

라벨 맵은 최근접 채움 + median 필터로 despeckle한다
(`envmap_baking/seam_blend.py:clean_label_maps`).

**단계 B — 라플라시안 피라미드 블렌딩** (`envmap_baking/seam_blend.py`)

각 소스 프레임을 face에 재투영한 레이어의 **라플라시안 피라미드**와, 그 라벨 마스크의
**가우시안 피라미드**를 레벨별로 곱해 누적한다:

```python
def add_layer(self, face, layer, layer_valid, mask):
    filled = push_pull_fill(layer, layer_valid)      # 스플랫 홀 채움
    bands = laplacian_pyramid(filled, self.levels)
    mask_bands = gaussian_pyramid(mask.astype(np.float32), self.levels)
    for level in range(self.levels):
        self.numerator[face][level] += bands[level] * mask_bands[level][..., None]
        self.denominator[face][level] += mask_bands[level]
```

- 저주파(노출 차)는 넓게, 고주파(디테일)는 seam 바로 근처에서만 전환 → 경계선이 사라지고
  이중상(ghosting)이 없다.
- 프레임별로 증분 누적하므로 메모리는 소스 수와 무관 (피라미드 1세트).
- **face 경계**: 각 면을 `--seam-padding`(기본 64px)만큼 이웃 면 방향까지 확장해 렌더한 뒤
  블렌딩하고 잘라낸다 (`padded_face_directions` / `pad_label_map`). face 사이에 새 seam이
  생기지 않는다.

남은 한계: 프레임 간 노출 차이 자체가 하늘에 부드러운 얼룩으로 남는다(별도의 exposure
alignment 필요). 라벨 배치는 그래프컷이 아닌 근사라 능선 근처에 소소한 얼룩이 있을 수 있다.

### 1.7 Per-texel distance (height) 맵

Cubemap을 (color, height)로 확장한다 — height는 **bake origin에서 그 텍셀의 배경
표면까지의 metric distance**다. 렌더 쪽 parallax 보정(§2.4)의 입력이 된다.

- **누적**: 배경 픽셀마다 depth로 world 점을 만들고(`p = cam_origin + ray·depth`)
  `r = |p − bake_origin|`을 계산, **inverse distance(disparity = 1/r)** 를 color와
  동일한 가중치·median reservoir 구조로 병렬 누적한다
  (`pipeline.py:make_disparities`, `cubemap.py:distance_maps`).
- **disparity 공간을 쓰는 이유**: sky(depth = −1)는 disparity 0으로 자연스럽게
  표현되고(무한 거리의 극한), sky·원경이 섞인 텍셀의 median/mean이 잘 정의되며,
  렌더 쪽 bilinear 보간도 스카이라인에서 부드럽게 열화된다.
- **출력**: `distance.npy`, float32 `(6, face_size, face_size)`, 텍셀별 metric
  distance. `+inf` = sky, `NaN` = 미커버. meta.json에 `distance_map` /
  `distance_convention` 필드가 추가된다.
- `background_radius`(전역 skydome 반경, distance 맵이 없는 소비자용 fallback)는
  이제 distance 맵의 유한값 median으로 계산한다 — 별도의 프레임 재순회 패스가
  사라졌다.

### 1.8 출력물

```
cubemaps/cubemap/
  posx.png negx.png posy.png negy.png posz.png negz.png   # 1024² faces
  distance.npy                                             # per-texel distance (height)
  coverage.png                                             # 커버리지 시트
  meta.json                                                # world_to_cubemap, bake_origin,
                                                           # background_radius, distance_map
```

---

## Part 2 — 렌더 파이프라인 수정 (gs-exp)

### 2.1 핵심 아이디어: rasterizer의 배경 합성 지점

3DGS rasterizer의 forward는 픽셀별로 gaussians를 **front-to-back**으로 순회하며
투과율 `T`를 누적하고, 마지막에 배경을 더한다:

```
C = Σᵢ cᵢ·αᵢ·Tᵢ + T_final · bg        (Tᵢ₊₁ = Tᵢ(1-αᵢ))
```

이 식은 "배경을 먼저 칠하고 그 위에 back-to-front alpha blending"한 결과와 수학적으로
동일하다. 즉 원하는 합성 구조는 이미 CUDA 커널 안에 있고, 문제는 `bg`가 픽셀 공통 (3,)
상수라는 것뿐이었다. 그래서 **`bg`를 per-pixel (3,H,W) 이미지로 확장**하는 것이 정확한
수정이다 — 별도 합성 패스도, 근사도 없다.

### 2.2 CUDA 패치 (`submodules/diff-gaussian-rasterization`)

수정은 인덱싱 두 줄이 전부다.

`cuda_rasterizer/forward.cu` — 픽셀별 배경 합성:

```cuda
// before: out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch * H * W + pix_id];
```

`cuda_rasterizer/backward.cu` — 배경이 alpha gradient에 주는 기여항도 동일하게:

```cuda
// Account for fact that alpha also influences how much of
// the background color is added if nothing left to blend
float bg_dot_dpixel = 0;
for (int i = 0; i < C; i++)
    bg_dot_dpixel += bg_color[i * H * W + pix_id] * dL_dpixel[i];
dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
```

**하위 호환**: Python 래퍼(`diff_gaussian_rasterization/__init__.py`)가 (3,) 상수 bg를
자동으로 (3,H,W)로 broadcast하므로 기존 `train.py`/`render.py`는 수정 없이 동작한다:

```python
# The CUDA kernels composite a per-pixel (3, H, W) background behind the
# Gaussians; broadcast a constant (3,) background color for compatibility.
bg = raster_settings.bg
if bg.dim() == 1:
    bg = bg[:, None, None].expand(-1, raster_settings.image_height, raster_settings.image_width)
raster_settings = raster_settings._replace(bg=bg.contiguous())
```

CUDA 수정 후에는 재빌드가 필요하다:

```bash
conda activate gaussian_splatting
pip install ./submodules/diff-gaussian-rasterization --no-build-isolation
```

**검증** (무작위 gaussians, per-pixel bg 단일 패스 vs "black 렌더 + (1−α)·bg" 2패스 합성):

| 항목 | 최대 차이 |
|---|---|
| forward | 3.6e-07 |
| grad means3D | 4.4e-04 (값 스케일 3.1e+02) |
| grad colors | 3.2e-05 (스케일 3.9e+01) |
| grad opacity | 7.6e-05 (스케일 5.3e+01) |
| 상수 bg broadcast | 0 (비트 동일) |

float atomic 누적 순서 차이 수준으로, 두 공식이 동치임을 확인했다.

### 2.3 Cubemap 샘플러 (`exp_utils/cubemap_bg.py`)

카메라 한 대의 모든 픽셀에 대해 cubemap을 평가해 (3,H,W) 배경 이미지를 만든다.

**픽셀 → world 레이**: COLMAP 카메라 축(`+X right, +Y down, +Z forward`)으로 레이를 만들고
camera-to-world 회전으로 world로 보낸다. 3DGS의 `world_view_transform`은 W2C의 전치라서
좌상단 3×3이 곧 camera-to-world 회전이다:

```python
directions_cam = torch.stack([
    (xs + 0.5 - width * 0.5) / fx,
    (ys + 0.5 - height * 0.5) / fy,
    torch.ones_like(xs),
], dim=-1)
# world_view_transform is the transposed world-to-camera matrix, so its
# upper-left 3x3 block is already the camera-to-world rotation.
cam_to_world = view.world_view_transform[:3, :3].to(self.device)
directions = directions_cam @ cam_to_world.T
directions = directions @ self.world_to_cubemap.T      # meta.json의 회전
```

**Cubemap 샘플링**: face/UV 매핑은 베이커의 `directions_to_faces_uv`를 torch로 그대로
포팅했고(규약 일치 보장), face별로 `grid_sample` bilinear 조회한다. `align_corners=False`
기준 `gx = 2u − 1`이 베이킹된 텍셀 센터와 정확히 정렬된다. `_sample(faces, directions)`은
채널 수와 무관하게 동작해 color faces `(6,3,S,S)`와 disparity faces `(6,1,S,S)`를 같은
코드로 조회한다:

```python
# grid_sample with align_corners=False: gx = 2u - 1 samples the texel
# grid at continuous position u * S - 0.5, matching baked texel centers.
grid = torch.stack([u * 2.0 - 1.0, v * 2.0 - 1.0], dim=-1)
sampled = F.grid_sample(faces[f:f+1], grid.unsqueeze(0),
                        mode="bilinear", padding_mode="border", align_corners=False)
```

### 2.4 Parallax 보정: radial height field 교차

Cubemap은 bake origin 기준 방향별 텍스처라서, 카메라가 origin에서 `|o|`만큼 떨어져
있으면 방향 조회는 최대 `~|o|/r` 라디안 어긋난다(r = 배경 거리). §1.7의 per-texel
distance 맵이 있으면 배경을 **bake origin 중심의 radial height field** `r(û)`로
취급하고, 각 픽셀 레이 `o + t·d`와의 교점을 **고정점 반복**으로 푼다
(`CubemapBackground._parallax_directions`):

```
u₀ = d
반복 (기본 3회):
  rₖ = distance 맵을 uₖ 방향으로 bilinear 샘플 (disparity → 1/disparity)
  tₖ = 반경 rₖ 구와 레이의 far-root 교차:  t = −b + √(b² − (|o|² − rₖ²)),  b = d·o
  uₖ₊₁ = normalize(o + tₖ·d)
최종 color는 u_N 방향으로 샘플
```

- **수렴**: 배경은 항상 `r ≫ |o|`이므로 보정각이 작아 고정점이 빠르게 수렴한다.
  합성 씬 검증에서 3회 = 8회(수렴 완료)였다.
- **안정성**: 반경을 `1.05·|o|`로 하한 클램프해 카메라가 항상 구 내부에 있게 하면
  far-root가 항상 양수라 반복이 접히지 않는다. sky(disparity 0)는 반경 ~10⁶으로
  방향 샘플링에 수렴한다. 미커버 텍셀(NaN)은 로드 시 `background_radius`로 채워
  skydome처럼 동작한다.
- **비용**: 배경 1장당 disparity 조회 3회 + color 조회 1회 = grid_sample 4라운드.
  학습에서는 카메라별 1회 계산 후 캐시되므로 무시 가능하다.
- distance 맵이 없으면 기존 **skydome**(단일 `background_radius` 구 교차)으로,
  bake_origin도 없으면 방향 샘플링으로 fallback한다.

**합성 씬 검증** (바닥 평면 y=−5 + 방향성 sky, 카메라 offset 3, 480×360):

| 모드 | mean&#124;err&#124; | P95 | PSNR |
|---|---|---|---|
| per-texel parallax (3 steps) | 0.0134 | 0.022 | 23.1 dB |
| skydome (const radius) | 0.1475 | 0.660 | 11.7 dB |
| direction only | 0.1182 | 0.494 | 14.0 dB |

잔여 오차는 지평선 1~2행에 집중되며(텍셀 해상도 한계), 그 밴드를 제외하면
mean err 0.0021 — uint8 양자화 바닥 수준이다.

**실측 검증** (T&T train, 8프레임마다 1장을 베이킹에서 제외하고 그 held-out 뷰 5장으로
평가; 사진의 유한 depth 배경 픽셀에서 노출 보상 mean|err|):

| 베이킹 \ 샘플링 | direction | skydome | parallax |
|---|---|---|---|
| rotation-only | 0.154 | 0.198 | 0.193 |
| **depth-warp** | 0.169 | 0.164 | **0.154** |

- **일관된 조합만 동작한다**: depth-warp 베이킹 + parallax 렌더링(우하단)과
  rotation-only + direction(좌상단)이 각각의 기하 모델 안에서 최선이고, 교차 조합은
  명확히 나쁘다.
- best-shift 정렬 측정에서 depth-warp + parallax는 최대 offset 뷰(6.7)에서 **0px
  오정렬**(direction-only는 15px). 노출·블렌딩 소프트함이 만드는 오차 바닥(~0.13)
  때문에 mean|err| 차이는 작아 보이지만, 기하 정렬은 명확히 개선된다.
- rotation-only + direction과 수치가 비슷한 이유: 평가 뷰가 카메라 궤적 위에 있어
  direction 조회의 시차 오차가 제한적이기 때문. 궤적에서 더 벗어나는 novel view일수록
  (3DGS 평가 상황) origin 중심 모델 + parallax의 이점이 커진다.

### 2.5 학습 (`train_cubemap_bg.py`)

`train.py`와의 차이는 `[CUBEMAP]` 주석 지점뿐이다. 핵심은 렌더 호출 한 줄:

```python
# [CUBEMAP] The rasterizer fills each pixel's remaining transmittance
# with the cubemap evaluated along that pixel's ray.
bg = cubemap_bg(viewpoint_cam)
render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                    use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
```

- **단일 패스**: rasterizer가 배경 합성과 gradient를 모두 처리하므로 일반 학습과 거의 같은
  비용이다 (alpha 2차 패스 불필요).
- **학습 효과**: GT가 배경인 픽셀에서 loss가 cubemap 기준으로 계산되므로, gaussians가
  하늘/원경을 재구성할 유인이 사라지고, 그 위에 뜬 전경 gaussian은 opacity가 눌린다
  (배경 기여항이 `dL_dalpha`에 들어가는 것이 backward 패치의 역할).
- 카메라별 배경은 최초 1회만 계산해 uint8 CPU 캐시로 재사용한다:

  ```python
  class CachedCubemapBackground:
      def __call__(self, viewpoint_cam):
          key = viewpoint_cam.image_name
          if key not in self.cache:
              with torch.no_grad():
                  background = self.cubemap.background_for_view(viewpoint_cam)
              self.cache[key] = (background * 255.0).byte().cpu()
          return self.cache[key].cuda().float() / 255.0
  ```

- test 평가(`training_report`)와 SIBR GUI 프리뷰도 같은 cubemap 배경으로 렌더한다.
- `--random_background`는 무의미하므로 경고 후 무시, `white_background`도 효과 없음.
- densification / pruning / depth 정규화 / checkpoint 로직은 `train.py` 그대로.

실행:

```bash
conda activate gaussian_splatting
python train_cubemap_bg.py -s /home/sj/work/gs-dataset/tandt/train -m output/<name> --eval
# cubemap 위치 변경: --cubemap <dir>   (기본: <source_path>/cubemaps/cubemap)
```

### 2.6 렌더링 (`render_cubemap_bg.py`)

학습과 동일한 단일 패스 합성으로 학습-렌더 일관성을 유지한다:

```python
background = cubemap.background_for_view(view)
composite = render(view, gaussians, pipeline, background,
                   use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
```

```bash
python render_cubemap_bg.py -m output/<name> --skip_train
# 출력: <model>/test/ours_<iter>/renders_cubemap/
```

주의: `--cubemap` 인자의 기본값은 `None`이 아니라 `""`이다 — 3DGS의 `get_combined_args`가
`None` 값 CLI 인자를 cfg 병합에서 떨어뜨리기 때문.

### 2.7 파일 요약

| 파일 | 역할 |
|---|---|
| `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu` | per-pixel bg 합성 (1줄) |
| `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu` | per-pixel bg alpha gradient (1줄) |
| `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py` | (3,) bg 자동 broadcast |
| `exp_utils/cubemap_bg.py` | 레이 생성 + cubemap 샘플러 + parallax 보정 (`CubemapBackground`) |
| `train_cubemap_bg.py` | cubemap 배경 학습 스크립트 |
| `render_cubemap_bg.py` | cubemap 배경 렌더 스크립트 |
| `<dataset>/cubemaps/cubemap/meta.json` | `world_to_cubemap`, `bake_origin`, `background_radius`, `distance_map` |
| `<dataset>/cubemaps/cubemap/distance.npy` | per-texel 배경 거리 (height) 맵 |

### 2.8 알려진 한계

- parallax 보정은 **bake origin 기준 radial height field 모델**이다. 고정점 반복은
  스카이라인 같은 급격한 거리 불연속에서 교차점 앞뒤로 진동할 수 있고(고정 3회로
  비용 상한), 배경끼리의 상호 가림(disocclusion)으로 cubemap에 아예 없는 내용은
  복원할 수 없다. 합성 검증에서 이 오차는 지평선 1~2px 밴드에 국한됐다.
- DA3 depth의 프레임별 스케일 오차가 distance 맵의 노이즈로 남는다. median 누적이
  outlier는 걸러주지만, 저주파 바이어스는 남을 수 있다.
- cubemap의 미관측 영역(coverage 구멍)은 검은색으로 남는다 — hole filling(inpainting)은
  베이킹 파이프라인의 후속 과제.
- 프레임 간 노출 차이로 인한 하늘의 부드러운 밝기 얼룩은 seam blending이 아닌 exposure
  alignment의 몫이다.
- `use_trained_exp`(train_test_exp) 사용 시 노출 보정이 합성 이후 이미지에 적용되므로
  배경에도 노출이 걸린다. 기본 설정(False)에서는 무관하다.
