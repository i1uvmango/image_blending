# Laplacian Pyramid Image Blending

OpenCV 없이 numpy와 PIL만을 사용하여 구현한 Laplacian Pyramid 기반 이미지 블렌딩 시스템입니다.

## 기능

이 구현은 다음 8단계로 구성되어 있습니다:

1. **이미지 로드, YUV 변환, Y 채널 분리**: RGB 이미지를 YUV 컬러공간으로 변환하고 밝기(Y) 채널만 추출
2. **Custom Gaussian Filtering & Downsampling**: 시그마와 커널 크기를 파라미터로 받아 Gaussian 커널 생성 및 다운샘플링
3. **Gaussian Pyramid 생성**: 반복문을 사용하여 여러 레벨의 Gaussian 피라미드 생성
4. **Laplacian Pyramid 생성**: Gaussian 피라미드로부터 디테일 정보를 담은 Laplacian 피라미드 생성
5. **Mask 정의 및 Pyramid-level Blending**: 부드러운 그라데이션 mask를 사용한 피라미드 레벨별 블렌딩
6. **Pyramid Collapse**: 블렌딩된 Laplacian 피라미드를 복원하여 최종 이미지 생성
7. **YUV + UV 재결합, 최종 RGB 복원**: 블렌딩된 Y 채널과 원본 U, V 채널을 결합하여 RGB로 변환
8. **Kernel/Stride 변화 실험**: 다양한 파라미터 조합에 대한 실험 및 시각화

## 설치

```bash
pip install -r requirements.txt
```

필요한 패키지:
- numpy >= 1.20.0
- Pillow >= 8.0.0
- matplotlib >= 3.3.0
- scipy >= 1.6.0

## 사용법

### Step별 중간 결과 저장 (추천!)

각 단계별로 중간 결과를 모두 저장하는 함수입니다:

```python
from image_blending import blend_images_with_steps

# Step별 결과를 저장하며 블렌딩
results = blend_images_with_steps(
    image_A_path='image_A.jpg',
    image_B_path='image_B.jpg',
    output_dir='output',      # 결과 저장 디렉토리
    levels=5,
    sigma=1.0,
    blend_line=None,          # None이면 중간선
    gradient_width=50,
    save_intermediate=True    # 중간 결과 저장 여부
)

# 결과는 'output' 디렉토리에 저장됩니다:
# - step1_y_channel/: Y 채널 추출 결과
# - step2_gaussian_pyramid_A/, step2_gaussian_pyramid_B/: Gaussian 피라미드
# - step3_laplacian_pyramid_A/, step3_laplacian_pyramid_B/: Laplacian 피라미드
# - step4_mask/: Mask 및 Mask 피라미드
# - step5_blended_laplacian/: 블렌딩된 Laplacian 피라미드
# - step6_reconstructed/: 복원된 Y 채널
# - step7_final/: 최종 RGB 결과
```

### 기본 블렌딩 (최종 결과만)

최종 결과만 필요한 경우:

```python
from image_blending import blend_images
from PIL import Image

# 두 이미지 블렌딩
result = blend_images(
    image_A_path='image_A.jpg',
    image_B_path='image_B.jpg',
    levels=5,              # 피라미드 레벨 수
    sigma=1.0,             # Gaussian 표준편차
    blend_line=None,       # None이면 중간선, 정수면 해당 x 좌표
    gradient_width=50      # 그라데이션 폭
)

# 결과 저장
result_img = Image.fromarray(result)
result_img.save('blended_result.jpg')
```

### 개별 함수 사용

```python
from image_blending import (
    extract_y_channel,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    create_gradient_mask,
    blend_laplacian_pyramids,
    collapse_laplacian_pyramid
)

# Y 채널 추출
Y, U, V, shape = extract_y_channel('image.jpg')

# Gaussian Pyramid 생성
gaussian_pyramid = build_gaussian_pyramid(Y, levels=5, sigma=1.0)

# Laplacian Pyramid 생성
laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid, sigma=1.0)

# Mask 생성
mask = create_gradient_mask((height, width), blend_line=300, gradient_width=50)
```

### 파라미터 실험

```python
from image_blending import experiment_gaussian_parameters, visualize_experiments

Y, _, _, _ = extract_y_channel('image.jpg')

# 다양한 파라미터로 실험
experiments = experiment_gaussian_parameters(
    image=Y,
    sigmas=[0.5, 1.0, 2.0],
    kernel_sizes=[5, 9, 15],
    strides=[2, 3, 4]
)

# 결과 시각화
visualize_experiments(Y, experiments, 'experiments.png')
```

## 주요 함수 설명

### `blend_images_with_steps(image_A_path, image_B_path, output_dir='output', ...)`
Step별 중간 결과를 모두 저장하며 블렌딩하는 함수입니다. **추천!**

**파라미터:**
- `image_A_path`, `image_B_path`: 블렌딩할 두 이미지 경로
- `output_dir`: 결과 저장 디렉토리 (기본값: 'output')
- `levels`: 피라미드 레벨 수 (기본값: 5)
- `sigma`: Gaussian 필터의 표준편차 (기본값: 1.0)
- `kernel_size`: 커널 크기 (None이면 자동 계산)
- `blend_line`: 블렌딩 라인 x 좌표 (None이면 중간)
- `gradient_width`: 그라데이션 폭 (픽셀 단위)
- `save_intermediate`: 중간 결과 저장 여부 (기본값: True)

**반환값:** 각 step별 결과를 담은 딕셔너리

### `blend_images(image_A_path, image_B_path, levels=5, sigma=1.0, ...)`
전체 블렌딩 파이프라인을 실행하는 메인 함수입니다. 최종 결과만 반환합니다.

**파라미터:**
- `image_A_path`, `image_B_path`: 블렌딩할 두 이미지 경로
- `levels`: 피라미드 레벨 수 (기본값: 5)
- `sigma`: Gaussian 필터의 표준편차 (기본값: 1.0)
- `kernel_size`: 커널 크기 (None이면 자동 계산)
- `blend_line`: 블렌딩 라인 x 좌표 (None이면 중간)
- `gradient_width`: 그라데이션 폭 (픽셀 단위)

**반환값:** 블렌딩된 RGB 이미지 (numpy array)

### `create_gradient_mask(shape, blend_line=None, gradient_width=50)`
부드러운 그라데이션 mask를 생성합니다. 0~1 범위의 값을 가지며, 블렌딩 경계를 부드럽게 만듭니다.

### `build_gaussian_pyramid(image, levels, sigma=1.0, ...)`
입력 이미지로부터 Gaussian 피라미드를 생성합니다.

### `build_laplacian_pyramid(gaussian_pyramid, sigma=1.0, ...)`
Gaussian 피라미드로부터 Laplacian 피라미드를 생성합니다. 각 레벨은 디테일 정보를 담고 있습니다.

## 주의사항

- 이미지 크기가 다를 경우, 더 작은 크기로 맞춰집니다.
- U, V 채널은 첫 번째 이미지의 것을 사용합니다 (필요시 수정 가능).
- 피라미드 레벨 수는 이미지 크기에 따라 적절히 조정해야 합니다.

## 파일 구조

```
.
├── image_blending.py      # 메인 구현 파일
├── example_usage.py       # 사용 예제
├── requirements.txt       # 필요한 패키지 목록
└── README.md             # 이 파일
```

## 참고

- Laplacian Pyramid 블렌딩은 Burt & Adelson (1983)의 방법을 기반으로 합니다.
- Y 채널만 블렌딩하여 색 왜곡을 방지합니다.
- Mask의 그라데이션 폭을 조정하여 블렌딩 경계의 부드러움을 제어할 수 있습니다.

