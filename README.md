# ROI 기반 Eye Blending - Laplacian Pyramid 방식

OpenCV 없이 NumPy와 PIL만을 사용하여 구현한 ROI(Region of Interest) 기반 Laplacian Pyramid 이미지 블렌딩 시스템입니다.

hand 이미지의 특정 ROI 영역에 eye 이미지를 자연스럽게 합성하는 프로젝트입니다.

## 주요 특징

- **ROI 기반 블렌딩**: hand 이미지의 특정 영역에만 eye 이미지를 합성
- **Eye 윤곽만 블렌딩**: eye의 Laplacian 절댓값으로 edge mask를 생성하여 눈 윤곽만 블렌딩 대상으로 설정
- **Padding 없음**: ROI 크기에 정확히 맞춰 crop하여 padding 없이 처리
- **YUV 색공간**: 색 왜곡을 방지하기 위해 Y, U, V 채널 모두 처리
- **단계별 시각화**: 각 단계의 중간 결과를 모두 저장하여 과정을 추적 가능
- **경계 처리 방법 비교**: Gaussian Blur와 Feathering 방법을 비교하여 최적의 결과 선택

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

### 기본 실행

```bash
python src/roi_eye_blending.py
```

### 함수로 직접 호출

```python
from src.roi_eye_blending import roi_eye_blend

result = roi_eye_blend(
    hand_path='res/hand.jpg',
    eye_path='res/eye2.jpg',
    roi_coords=(318, 416, 224, 393),  # (y_start, y_end, x_start, x_end)
    output_dir='test2',
    levels=5,
    sigma=1.0,
    sigma_blur=3.0
)
```

## 프로젝트 진행 과정

### 1단계: 이미지 전처리
- **목표**: hand.jpg 이미지의 회전을 제거하고 640x480으로 리사이즈
- **방법**: EXIF orientation 정보를 사용하여 회전 제거, 비율 유지하며 리사이즈
- **결과**: 전처리된 hand.jpg 이미지 생성

### 2단계: ROI 기반 블렌딩 초기 구현
- **목표**: hand 이미지의 특정 ROI 영역에 eye 이미지를 합성
- **방법**: 
  - ROI 영역을 crop하여 블렌딩
  - Laplacian Pyramid 방식 사용
  - Padding 없이 crop으로 shape 맞춤
- **문제점**: ROI 경계에서 색상 차이가 두드러짐

### 3단계: Edge Mask 기반 블렌딩
- **목표**: eye의 눈 윤곽만 블렌딩 대상으로 설정
- **방법**:
  - eye의 Laplacian 절댓값으로 edge mask 생성
  - Edge map을 normalize하고 Gaussian blur로 부드럽게 처리
  - Mask Pyramid 생성하여 각 레벨에서 블렌딩
- **결과**: 눈 윤곽만 자연스럽게 블렌딩됨

### 4단계: 색상 왜곡 문제 해결
- **문제**: Gaussian Pyramid 생성 시 색상 왜곡 발생
- **해결**:
  - Y, U, V 채널을 모두 처리하여 색상 정보 보존
  - 업샘플링 시 음수값 처리 (정규화 후 리사이즈, 원래 범위로 복원)
- **결과**: 색상 왜곡 없이 자연스러운 블렌딩

### 5단계: ROI 경계 처리 방법 비교
- **문제**: ROI와 hand 간의 경계가 뚜렷하게 나타남
- **해결 방법 비교**:
  - **방법 4 (Gaussian Blur)**: ROI 삽입 후 경계 영역에 선택적 Gaussian blur 적용
  - **방법 1 (Feathering)**: ROI 경계에 그라데이션 마스크를 적용하여 부드럽게 블렌딩
- **결과**: 두 방법 모두 비교하여 최적의 결과 선택 가능

### 6단계: 코드 정리 및 최적화
- **목표**: test2에 해당하는 코드만 남기고 정리
- **작업**:
  - test2와 관련 없는 파일 삭제
  - src 폴더로 코드 정리
  - 모든 step이 포함되어 있는지 확인

## 블렌딩 과정 및 인사이트

### Step 1: 이미지 준비 및 ROI crop

**과정:**
- hand 이미지에서 ROI 영역을 crop
- eye 이미지를 ROI 크기에 정확히 맞춰 리사이즈 (padding 없음, 비율 무시)
- RGB를 YUV로 변환

**인사이트:**
- **Padding vs Crop 선택**: 초기에는 padding을 사용했지만, ROI 경계에서 색상 차이가 두드러지는 문제가 발생했습니다. padding을 제거하고 정확히 ROI 크기로 crop하니 경계가 더 자연스러워졌습니다.
- **비율 무시 리사이즈**: eye 이미지의 원본 비율을 유지하면 ROI 내부에 빈 공간이 생기거나 잘리는 문제가 발생합니다. ROI 크기에 정확히 맞춰 리사이즈하면 전체 영역을 활용할 수 있습니다.
- **YUV 색공간 선택**: RGB에서 직접 블렌딩하면 색상 왜곡이 발생합니다. YUV 색공간으로 변환하여 밝기(Y)와 색상(U, V)을 분리하면 색 왜곡 없이 자연스러운 블렌딩이 가능합니다.

**출력**: `test2/step1_preparation/roi_crop.png` - Hand ROI와 Eye ROI 비교

---

### Step 2: Gaussian Pyramid 생성 (Y, U, V 모두)

**과정:**
- hand와 eye 각각에 대해 Y, U, V 채널별로 Gaussian Pyramid 생성
- 각 레벨에서 Gaussian 필터 적용 후 다운샘플링
- 여러 스케일의 이미지 표현 생성

**인사이트:**
- **Y, U, V 모두 처리**: 초기에는 Y 채널만 블렌딩했지만, U, V 채널도 함께 처리하니 색상 정보가 더 잘 보존됩니다.
- **Gaussian Pyramid의 역할**: 다양한 스케일에서 이미지를 표현하여, 세밀한 디테일부터 전체적인 구조까지 모두 고려한 블렌딩이 가능합니다.
- **레벨 수 선택**: 레벨이 너무 많으면 작은 이미지에서 정보 손실이 발생하고, 너무 적으면 세밀한 블렌딩이 어렵습니다. ROI 크기에 따라 적절히 조정해야 합니다 (기본값: 5).

**출력**:
- `test2/step2_gaussian_pyramid_hand/pyramid.png` - Hand Gaussian Pyramid 시각화
- `test2/step2_gaussian_pyramid_hand/level_*.png` - 각 레벨별 RGB 이미지
- `test2/step2_gaussian_pyramid_eye/pyramid.png` - Eye Gaussian Pyramid 시각화
- `test2/step2_gaussian_pyramid_eye/level_*.png` - 각 레벨별 RGB 이미지

---

### Step 3: Laplacian Pyramid 생성

**과정:**
- Gaussian Pyramid로부터 Laplacian Pyramid 생성
- 각 레벨은 해당 스케일에서의 디테일 정보를 담고 있음
- Laplacian = 현재 레벨 - 업샘플링된 다음 레벨

**인사이트:**
- **Laplacian Pyramid의 의미**: Laplacian Pyramid는 각 스케일에서의 "차이" 정보를 담고 있습니다. 이는 블렌딩에 필요한 디테일 정보를 추출하는 핵심입니다.
- **업샘플링 시 주의사항**: U, V 채널은 음수값을 가질 수 있어서, 업샘플링 시 정규화 후 리사이즈하고 원래 범위로 복원해야 색 왜곡을 방지할 수 있습니다.
- **디테일 보존**: Laplacian Pyramid를 사용하면 원본 이미지를 완벽하게 복원할 수 있으며, 각 레벨의 디테일 정보를 독립적으로 블렌딩할 수 있습니다.

**출력**:
- `test2/step3_laplacian_pyramid_hand/pyramid.png` - Hand Laplacian Pyramid 시각화
- `test2/step3_laplacian_pyramid_hand/level_*.png` - 각 레벨별 Laplacian 이미지
- `test2/step3_laplacian_pyramid_eye/pyramid.png` - Eye Laplacian Pyramid 시각화
- `test2/step3_laplacian_pyramid_eye/level_*.png` - 각 레벨별 Laplacian 이미지

---

### Step 4: Edge Mask 생성 (eye의 Laplacian 절댓값 기반)

**과정:**
- eye의 Laplacian 절댓값을 합산하여 edge map 생성
- Edge map을 normalize하고 Gaussian blur로 부드럽게 처리
- 눈 윤곽만 블렌딩 대상이 되도록 mask 생성
- Mask Pyramid 생성

**인사이트:**
- **Laplacian 절댓값의 의미**: Laplacian은 변화량을 나타내므로, 절댓값을 취하면 edge(경계) 정보를 얻을 수 있습니다. eye 이미지에서 눈 윤곽이 가장 큰 변화량을 가지므로, 이를 이용해 눈 영역만 추출할 수 있습니다.
- **Gaussian blur의 필요성**: Edge map만 사용하면 경계가 너무 날카로워 블렌딩 결과가 부자연스럽습니다. Gaussian blur를 적용하여 부드러운 그라데이션 mask를 만들면 자연스러운 블렌딩이 가능합니다.
- **Mask Pyramid**: 각 레벨에 맞는 mask를 생성하여, 모든 스케일에서 일관된 블렌딩이 이루어지도록 합니다.
- **sigma_blur 조정**: sigma_blur가 너무 작으면 mask가 날카로워지고, 너무 크면 눈 영역이 너무 넓어집니다. 적절한 값(기본값: 3.0)을 찾는 것이 중요합니다.

**출력**:
- `test2/step4_edge_mask/edge_mask.png` - Edge mask 시각화
- `test2/step4_edge_mask/mask_pyramid.png` - Mask Pyramid 시각화
- `test2/step4_edge_mask/mask_level_*.png` - 각 레벨별 mask 이미지

---

### Step 5: Laplacian Pyramid Blending

**과정:**
- 각 레벨에서 `blended = eye * mask + hand * (1 - mask)` 형태로 가중합 수행
- hand, eye, mask 각각의 Laplacian Pyramid를 사용하여 블렌딩
- Y, U, V 채널 모두 동일한 방식으로 블렌딩

**인사이트:**
- **레벨별 블렌딩의 장점**: 각 스케일에서 독립적으로 블렌딩하면, 세밀한 디테일부터 전체적인 구조까지 모두 자연스럽게 합성됩니다. 단순히 원본 크기에서만 블렌딩하면 일부 스케일에서 부자연스러운 결과가 나올 수 있습니다.
- **가중합 공식**: `eye * mask + hand * (1 - mask)`는 mask 값에 따라 eye와 hand를 선형 보간합니다. mask가 1에 가까우면 eye가 더 많이 반영되고, 0에 가까우면 hand가 더 많이 반영됩니다.
- **Y, U, V 모두 블렌딩**: Y 채널만 블렌딩하면 색상 정보가 손실될 수 있습니다. U, V 채널도 함께 블렌딩하면 색상이 더 자연스럽게 보존됩니다.
- **크기 맞추기**: 각 레벨에서 hand, eye, mask의 크기가 다를 수 있으므로, 최소 크기로 맞춰서 블렌딩해야 오류를 방지할 수 있습니다.

**출력**:
- `test2/step5_blended_laplacian/pyramid_Y.png` - Blended Laplacian Pyramid (Y) 시각화
- `test2/step5_blended_laplacian/level_*.png` - 각 레벨별 블렌딩된 Laplacian 이미지

---

### Step 6: Pyramid Collapse 및 ROI 삽입

**과정:**
- 블렌딩된 Laplacian Pyramid를 복원하여 원본 크기로 복원
- Y, U, V 채널을 YUV에서 RGB로 변환
- 블렌딩된 ROI 영역을 원본 hand 이미지에 삽입
- 경계 처리 방법 비교 (Gaussian Blur vs Feathering)

**인사이트:**
- **Pyramid Collapse**: 마지막 레벨부터 시작하여 역순으로 업샘플링하고 Laplacian을 더하면 원본 크기로 복원됩니다. 이 과정에서 모든 스케일의 디테일 정보가 복원됩니다.
- **ROI 삽입 시 경계 문제**: 블렌딩된 ROI를 원본 hand 이미지에 삽입할 때, 경계 부분에서 색상 차이가 두드러질 수 있습니다. 이를 해결하기 위해 두 가지 방법을 비교했습니다:
  - **방법 4 (Gaussian Blur)**: ROI 삽입 후 경계 영역에 선택적 Gaussian blur 적용. 경계가 부드러워지지만 전체적으로 약간 흐릿해질 수 있음.
  - **방법 1 (Feathering)**: ROI 경계에 그라데이션 마스크를 적용하여 부드럽게 블렌딩. 경계가 자연스럽게 블렌딩되며 디테일 보존에 유리함.
- **색상 보존**: YUV에서 RGB로 변환할 때 클리핑(0~255 범위)을 적용하여 색상이 왜곡되지 않도록 합니다.

**출력**:
- `test2/step6_final/comparison.png` - 세 가지 방법 비교 (원본, Gaussian Blur, Feathering)
- `test2/step6_final/blended_result_original.jpg` - 경계 처리 없는 원본
- `test2/step6_final/blended_result_method4_blur.jpg` - 방법 4 (Gaussian Blur) 결과
- `test2/step6_final/blended_result_method1_feathering.jpg` - 방법 1 (Feathering) 결과

---

## 출력 디렉토리 구조

```
test2/
├── step1_preparation/
│   └── roi_crop.png              # Hand ROI와 Eye ROI 비교
├── step2_gaussian_pyramid_hand/
│   ├── pyramid.png               # Gaussian Pyramid 시각화
│   └── level_*.png                # 각 레벨별 RGB 이미지
├── step2_gaussian_pyramid_eye/
│   ├── pyramid.png               # Gaussian Pyramid 시각화
│   └── level_*.png                # 각 레벨별 RGB 이미지
├── step3_laplacian_pyramid_hand/
│   ├── pyramid.png               # Laplacian Pyramid 시각화
│   └── level_*.png                # 각 레벨별 Laplacian 이미지
├── step3_laplacian_pyramid_eye/
│   ├── pyramid.png               # Laplacian Pyramid 시각화
│   └── level_*.png                # 각 레벨별 Laplacian 이미지
├── step4_edge_mask/
│   ├── edge_mask.png             # Edge mask 시각화
│   ├── mask_pyramid.png          # Mask Pyramid 시각화
│   └── mask_level_*.png           # 각 레벨별 mask 이미지
├── step5_blended_laplacian/
│   ├── pyramid_Y.png             # Blended Laplacian Pyramid 시각화
│   └── level_*.png                # 각 레벨별 블렌딩된 Laplacian 이미지
└── step6_final/
    ├── comparison.png             # 세 가지 방법 비교
    ├── blended_result_original.jpg        # 경계 처리 없는 원본
    ├── blended_result_method4_blur.jpg    # 방법 4 (Gaussian Blur) 결과
    └── blended_result_method1_feathering.jpg  # 방법 1 (Feathering) 결과
```

## 파라미터 설명

- `hand_path`: hand 이미지 경로
- `eye_path`: eye 이미지 경로
- `roi_coords`: ROI 좌표 `(y_start, y_end, x_start, x_end)`
  - 예: `(318, 416, 224, 393)` - 가로(224~393), 세로(318~416) 영역
- `output_dir`: 결과 저장 디렉토리 (기본값: 'test2')
- `levels`: 피라미드 레벨 수 (기본값: 5)
  - ROI 크기에 따라 조정 필요
- `sigma`: Gaussian 필터의 표준편차 (기본값: 1.0)
  - 값이 클수록 더 부드러운 블러링
- `sigma_blur`: Edge mask blur 표준편차 (기본값: 3.0)
  - 값이 클수록 mask가 더 부드러워짐

## 주요 함수

### `roi_eye_blend(hand_path, eye_path, roi_coords, output_dir='test2', ...)`
ROI 영역에 eye 이미지를 Laplacian Pyramid로 블렌딩하는 메인 함수입니다.

**파라미터:**
- `hand_path`: hand 이미지 경로
- `eye_path`: eye 이미지 경로
- `roi_coords`: ROI 좌표 `(y_start, y_end, x_start, x_end)`
- `output_dir`: 결과 저장 디렉토리 (기본값: 'test2')
- `levels`: 피라미드 레벨 수 (기본값: 5)
- `sigma`: Gaussian 필터의 표준편차 (기본값: 1.0)
- `sigma_blur`: Edge mask blur 표준편차 (기본값: 3.0)

**반환값:** 블렌딩된 RGB 이미지 (numpy array)

### `prepare_roi_images(hand_path, eye_path, roi_coords)`
hand와 eye 이미지를 준비하고 ROI 영역에 맞춥니다. padding 없이 crop하여 처리합니다.

### `create_edge_mask_from_laplacian(laplacian_pyramid, sigma_blur=3.0)`
eye의 Laplacian 절댓값으로 edge mask를 생성합니다. 눈 윤곽만 블렌딩 대상이 되도록 합니다.

### `blend_laplacian_pyramids(laplacian_hand, laplacian_eye, mask_pyramid)`
두 Laplacian Pyramid를 마스크로 블렌딩합니다. 각 레벨에서 `eye * mask + hand * (1 - mask)` 형태로 가중합을 수행합니다.

### `collapse_laplacian_pyramid(laplacian_pyramid, sigma=1.0)`
Laplacian Pyramid를 복원하여 원본 이미지 크기로 재구성합니다.

## 핵심 인사이트 요약

1. **YUV 색공간 사용**: RGB에서 직접 블렌딩하면 색상 왜곡이 발생하므로, YUV 색공간으로 변환하여 밝기와 색상을 분리하여 처리합니다.

2. **Padding 제거**: ROI 경계에서 색상 차이가 두드러지는 문제를 해결하기 위해 padding을 제거하고 정확히 ROI 크기로 crop합니다.

3. **Laplacian 절댓값으로 Edge Mask 생성**: eye의 Laplacian 절댓값을 이용하여 눈 윤곽만 추출하고, Gaussian blur로 부드러운 mask를 생성합니다.

4. **레벨별 블렌딩**: 각 스케일에서 독립적으로 블렌딩하여 세밀한 디테일부터 전체적인 구조까지 모두 자연스럽게 합성합니다.

5. **Y, U, V 모두 처리**: Y 채널만 블렌딩하면 색상 정보가 손실되므로, U, V 채널도 함께 블렌딩합니다.

6. **업샘플링 시 음수값 처리**: U, V 채널은 음수값을 가질 수 있으므로, 업샘플링 시 정규화 후 리사이즈하고 원래 범위로 복원해야 합니다.

7. **경계 처리 방법 비교**: ROI 삽입 시 경계 문제를 해결하기 위해 Gaussian Blur와 Feathering 방법을 비교하여 최적의 결과를 선택합니다.

## 참고

- Laplacian Pyramid 블렌딩은 Burt & Adelson (1983)의 방법을 기반으로 합니다.
- ROI 기반 블렌딩은 eye의 Laplacian 절댓값으로 edge mask를 생성하여 눈 윤곽만 블렌딩 대상으로 설정합니다.
- 각 단계의 중간 결과를 모두 저장하여 과정을 추적하고 디버깅할 수 있습니다.
- 경계 처리 방법을 비교하여 최적의 결과를 선택할 수 있습니다.
