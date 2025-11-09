"""
Laplacian Pyramid Image Blending
OpenCV 없이 numpy와 PIL만 사용하여 구현
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import os


# ============================================================================
# Step 1: 이미지 로드, YUV 변환, Y 채널 분리
# ============================================================================

def load_image_rgb(image_path):
    """
    이미지를 RGB로 로드 (EXIF orientation 자동 적용)
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        numpy array (H, W, 3) - RGB 이미지
    """
    from PIL import ImageOps
    
    img = Image.open(image_path)
    # EXIF orientation 정보가 있으면 자동으로 회전 적용
    img = ImageOps.exif_transpose(img)
    img = img.convert('RGB')
    return np.array(img, dtype=np.float64)


def rgb_to_yuv(rgb_image):
    """
    RGB 이미지를 YUV 컬러공간으로 변환
    
    Args:
        rgb_image: numpy array (H, W, 3) - RGB 이미지
        
    Returns:
        tuple: (Y, U, V) 각각 (H, W) 형태의 numpy array
    """
    R, G, B = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
    
    # YUV 변환 공식
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.14713 * R - 0.28886 * G + 0.436 * B
    V = 0.615 * R - 0.51499 * G - 0.10001 * B
    
    return Y, U, V


def yuv_to_rgb(Y, U, V):
    """
    YUV를 RGB로 변환
    
    Args:
        Y, U, V: 각각 (H, W) 형태의 numpy array
        
    Returns:
        numpy array (H, W, 3) - RGB 이미지
    """
    R = Y + 1.13983 * V
    G = Y - 0.39465 * U - 0.58060 * V
    B = Y + 2.03211 * U
    
    # 클리핑 (0-255 범위)
    rgb = np.stack([R, G, B], axis=2)
    rgb = np.clip(rgb, 0, 255)
    
    return rgb


def extract_y_channel(image_path):
    """
    이미지를 로드하고 YUV로 변환한 후 Y 채널만 추출
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        tuple: (Y, U, V, shape) - Y 채널, U 채널, V 채널, 원본 이미지 shape
    """
    rgb_image = load_image_rgb(image_path)
    Y, U, V = rgb_to_yuv(rgb_image)
    shape = rgb_image.shape[:2]  # (H, W)
    
    return Y, U, V, shape


# ============================================================================
# Step 2: Custom Gaussian Filtering & Downsampling
# ============================================================================

def create_gaussian_kernel(sigma, kernel_size=None):
    """
    2D Gaussian 커널 생성
    
    Args:
        sigma: Gaussian 표준편차
        kernel_size: 커널 크기 (None이면 6*sigma+1로 자동 설정)
        
    Returns:
        numpy array (kernel_size, kernel_size) - 정규화된 Gaussian 커널
    """
    if kernel_size is None:
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
    
    # 중심점
    center = kernel_size // 2
    
    # 좌표 그리드 생성
    x, y = np.meshgrid(np.arange(kernel_size) - center, 
                       np.arange(kernel_size) - center)
    
    # Gaussian 공식
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # 정규화
    kernel = kernel / np.sum(kernel)
    
    return kernel


def gaussian_filter_2d(image, sigma, kernel_size=None):
    """
    Gaussian 필터를 적용한 컨볼루션
    
    Args:
        image: 입력 이미지 (H, W)
        sigma: Gaussian 표준편차
        kernel_size: 커널 크기
        
    Returns:
        numpy array (H, W) - 필터링된 이미지
    """
    kernel = create_gaussian_kernel(sigma, kernel_size)
    
    # 2D 컨볼루션
    filtered = convolve2d(image, kernel, mode='same', boundary='symm')
    
    return filtered


def downsample(image, stride=2):
    """
    이미지를 stride만큼 다운샘플링
    
    Args:
        image: 입력 이미지 (H, W)
        stride: 다운샘플링 배율 (기본값 2)
        
    Returns:
        numpy array - 다운샘플링된 이미지
    """
    return image[::stride, ::stride]


def gaussian_filter_and_downsample(image, sigma, kernel_size=None, stride=2):
    """
    Gaussian 필터링 후 다운샘플링
    
    Args:
        image: 입력 이미지 (H, W)
        sigma: Gaussian 표준편차
        kernel_size: 커널 크기
        stride: 다운샘플링 배율
        
    Returns:
        numpy array - 필터링 및 다운샘플링된 이미지
    """
    filtered = gaussian_filter_2d(image, sigma, kernel_size)
    downsampled = downsample(filtered, stride)
    return downsampled


# ============================================================================
# Step 3: Gaussian Pyramid 생성
# ============================================================================

def build_gaussian_pyramid(image, levels, sigma=1.0, kernel_size=None, stride=2):
    """
    Gaussian Pyramid 생성 (재귀가 아닌 반복문 이용)
    
    Args:
        image: 입력 이미지 (H, W) - Y 채널
        levels: 피라미드 레벨 수
        sigma: Gaussian 표준편차
        kernel_size: 커널 크기
        stride: 다운샘플링 배율
        
    Returns:
        list: Gaussian pyramid (각 레벨의 이미지 리스트)
    """
    pyramid = [image.copy()]
    current_image = image.copy()
    
    for i in range(levels - 1):
        # Gaussian 필터링 후 다운샘플링
        current_image = gaussian_filter_and_downsample(
            current_image, sigma, kernel_size, stride
        )
        pyramid.append(current_image)
    
    return pyramid


# ============================================================================
# Step 4: Laplacian Pyramid 생성
# ============================================================================

def upsample(image, scale_factor=2, method='bilinear'):
    """
    이미지를 업샘플링 (보간)
    
    Args:
        image: 입력 이미지 (H, W)
        scale_factor: 업샘플링 배율
        method: 보간 방법 ('bilinear', 'nearest')
        
    Returns:
        numpy array - 업샘플링된 이미지
    """
    h, w = image.shape
    new_h, new_w = h * scale_factor, w * scale_factor
    
    # PIL을 사용한 보간
    img_pil = Image.fromarray(image.astype(np.uint8))
    
    if method == 'bilinear':
        resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
    else:
        resized = img_pil.resize((new_w, new_h), Image.NEAREST)
    
    return np.array(resized, dtype=np.float64)


def build_laplacian_pyramid(gaussian_pyramid, sigma=1.0, kernel_size=None):
    """
    Gaussian Pyramid로부터 Laplacian Pyramid 생성
    
    Args:
        gaussian_pyramid: Gaussian pyramid 리스트
        sigma: 업샘플링 후 smoothing을 위한 Gaussian 표준편차
        kernel_size: 커널 크기
        
    Returns:
        list: Laplacian pyramid (각 레벨의 이미지 리스트)
    """
    laplacian_pyramid = []
    
    for i in range(len(gaussian_pyramid) - 1):
        # 현재 레벨
        current = gaussian_pyramid[i]
        
        # 다음 레벨을 업샘플링
        next_level = gaussian_pyramid[i + 1]
        upsampled = upsample(next_level, scale_factor=2)
        
        # 크기 맞추기
        h_current, w_current = current.shape
        h_upsampled, w_upsampled = upsampled.shape
        
        if h_upsampled != h_current:
            upsampled = upsampled[:h_current, :]
        if w_upsampled != w_current:
            upsampled = upsampled[:, :w_current]
        
        # Gaussian smoothing
        upsampled_smooth = gaussian_filter_2d(upsampled, sigma, kernel_size)
        
        # Laplacian = 현재 레벨 - 업샘플링된 다음 레벨
        laplacian = current - upsampled_smooth
        laplacian_pyramid.append(laplacian)
    
    # 마지막 레벨은 Gaussian 그대로 사용 (low frequency)
    laplacian_pyramid.append(gaussian_pyramid[-1].copy())
    
    return laplacian_pyramid


# ============================================================================
# Step 5: Mask 정의 및 Pyramid-level Blending
# ============================================================================

def create_gradient_mask(shape, blend_line=None, gradient_width=50):
    """
    부드러운 그라데이션 mask 생성
    
    Args:
        shape: 이미지 shape (H, W)
        blend_line: 블렌딩 라인 위치 (None이면 중간)
        gradient_width: 그라데이션 폭 (픽셀)
        
    Returns:
        numpy array (H, W) - 0~1 범위의 mask
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float64)
    
    if blend_line is None:
        blend_line = w // 2
    
    # 좌표 그리드 생성 (2D)
    y, x = np.ogrid[:h, :w]
    
    # 그라데이션 계산
    distance = np.abs(x - blend_line)
    
    # 그라데이션 적용 (2D 배열로)
    mask = np.where(distance < gradient_width, 
                   0.5 * (1 + np.cos(np.pi * distance / gradient_width)),
                   np.where(x < blend_line, 1.0, 0.0))
    
    return mask


def create_mask_pyramid(mask, levels, sigma=1.0, kernel_size=None):
    """
    Mask의 Gaussian pyramid 생성 (각 해상도 레벨에 맞춰)
    
    Args:
        mask: 원본 mask (H, W) - 2D 배열
        levels: 피라미드 레벨 수
        sigma: Gaussian 표준편차
        kernel_size: 커널 크기
        
    Returns:
        list: 각 레벨에 맞는 mask pyramid
    """
    # mask가 2D 배열인지 확인
    if len(mask.shape) != 2:
        raise ValueError(f"mask must be 2D array, got shape {mask.shape}")
    
    mask_pyramid = [mask.copy().astype(np.float64)]
    current_mask = mask.copy().astype(np.float64)
    
    for i in range(levels - 1):
        # Gaussian 필터링 후 다운샘플링
        current_mask = gaussian_filter_2d(current_mask, sigma, kernel_size)
        current_mask = downsample(current_mask, stride=2)
        mask_pyramid.append(current_mask)
    
    return mask_pyramid


def blend_laplacian_pyramids(laplacian_A, laplacian_B, mask_pyramid):
    """
    두 Laplacian pyramid를 mask를 사용하여 블렌딩
    
    Args:
        laplacian_A: 첫 번째 이미지의 Laplacian pyramid
        laplacian_B: 두 번째 이미지의 Laplacian pyramid
        mask_pyramid: 각 레벨에 맞는 mask pyramid
        
    Returns:
        list: 블렌딩된 Laplacian pyramid
    """
    blended_pyramid = []
    
    for i in range(len(laplacian_A)):
        # 각 레벨의 mask 크기에 맞추기
        mask = mask_pyramid[i]
        
        # 크기 맞추기
        h_A, w_A = laplacian_A[i].shape
        h_B, w_B = laplacian_B[i].shape
        h_mask, w_mask = mask.shape
        
        # 최소 크기로 맞추기
        h_min = min(h_A, h_B, h_mask)
        w_min = min(w_A, w_B, w_mask)
        
        lap_A = laplacian_A[i][:h_min, :w_min]
        lap_B = laplacian_B[i][:h_min, :w_min]
        mask_resized = mask[:h_min, :w_min]
        
        # 블렌딩: mask * A + (1 - mask) * B
        blended = mask_resized * lap_A + (1 - mask_resized) * lap_B
        blended_pyramid.append(blended)
    
    return blended_pyramid


# ============================================================================
# Step 6: Pyramid Collapse (Blended Image 복원)
# ============================================================================

def collapse_laplacian_pyramid(laplacian_pyramid, sigma=1.0, kernel_size=None):
    """
    Laplacian pyramid를 복원하여 최종 이미지 생성
    
    Args:
        laplacian_pyramid: Laplacian pyramid 리스트
        sigma: 업샘플링 후 smoothing을 위한 Gaussian 표준편차
        kernel_size: 커널 크기
        
    Returns:
        numpy array (H, W) - 복원된 Y 채널 이미지
    """
    # 가장 낮은 해상도부터 시작
    reconstructed = laplacian_pyramid[-1].copy()
    
    # 하위 레벨부터 상위 레벨로 복원
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        # 업샘플링
        upsampled = upsample(reconstructed, scale_factor=2)
        
        # 크기 맞추기
        target = laplacian_pyramid[i]
        if upsampled.shape[0] != target.shape[0]:
            upsampled = upsampled[:target.shape[0], :]
        if upsampled.shape[1] != target.shape[1]:
            upsampled = upsampled[:, :target.shape[1]]
        
        # Gaussian smoothing
        upsampled_smooth = gaussian_filter_2d(upsampled, sigma, kernel_size)
        
        # Laplacian 더하기
        reconstructed = upsampled_smooth + target
    
    return reconstructed


# ============================================================================
# Step 7: YUV + UV 재결합, 최종 RGB 복원
# ============================================================================

def combine_yuv_channels(Y_blended, U_blended, V_blended):
    """
    블렌딩된 Y, U, V 채널을 RGB로 변환
    
    Args:
        Y_blended: 블렌딩된 Y 채널 (H, W)
        U_blended: 블렌딩된 U 채널 (H, W) - 이미 올바른 크기
        V_blended: 블렌딩된 V 채널 (H, W) - 이미 올바른 크기
        
    Returns:
        numpy array (H, W, 3) - RGB 이미지
    """
    # Y, U, V는 이미 같은 크기로 블렌딩되어 있으므로 직접 RGB 변환
    rgb = yuv_to_rgb(Y_blended, U_blended, V_blended)
    
    return rgb


# ============================================================================
# 메인 블렌딩 파이프라인
# ============================================================================

def blend_images(image_A_path, image_B_path, levels=5, sigma=1.0, 
                 kernel_size=None, blend_line=None, gradient_width=50,
                 output_dir=None, save_intermediate=False):
    """
    두 이미지를 Laplacian pyramid로 블렌딩하는 전체 파이프라인
    
    Args:
        image_A_path: 첫 번째 이미지 경로
        image_B_path: 두 번째 이미지 경로
        levels: 피라미드 레벨 수
        sigma: Gaussian 표준편차
        kernel_size: 커널 크기
        blend_line: 블렌딩 라인 위치
        gradient_width: 그라데이션 폭
        output_dir: 중간 결과 저장 디렉토리 (None이면 저장 안 함)
        save_intermediate: 중간 결과 저장 여부
        
    Returns:
        numpy array - 블렌딩된 RGB 이미지
    """
    # Step 1: 이미지 로드 및 Y 채널 추출
    Y_A, U_A, V_A, shape_A = extract_y_channel(image_A_path)
    Y_B, U_B, V_B, shape_B = extract_y_channel(image_B_path)
    
    # 이미지 B(손바닥)를 기준으로 크기 맞추기
    # 이미지 B는 전체가 보이도록 유지
    h_B, w_B = shape_B
    h_A, w_A = shape_A
    
    # 이미지 A를 이미지 B 크기에 맞춰서 리사이즈 (비율 유지, B 크기 내에 맞춤)
    # PIL을 사용하여 리사이즈
    img_A_pil = Image.fromarray(Y_A.astype(np.uint8))
    
    # 비율 유지하면서 이미지 B 크기 내에 맞춤
    aspect_ratio_A = w_A / h_A
    aspect_ratio_B = w_B / h_B
    
    # 이미지 B 크기를 넘지 않도록 리사이즈
    if aspect_ratio_A > aspect_ratio_B:
        # A가 더 넓음 - 너비를 B에 맞춤
        new_w_A = w_B
        new_h_A = int(w_B / aspect_ratio_A)
    else:
        # A가 더 높음 - 높이를 B에 맞춤
        new_h_A = h_B
        new_w_A = int(h_B * aspect_ratio_A)
    
    # 리사이즈
    img_A_resized = img_A_pil.resize((new_w_A, new_h_A), Image.LANCZOS)
    Y_A_resized = np.array(img_A_resized, dtype=np.float64)
    
    # U, V 채널도 리사이즈 (음수값 처리)
    # U, V는 음수값을 가질 수 있으므로 정규화하여 리사이즈
    U_A_min, U_A_max = U_A.min(), U_A.max()
    V_A_min, V_A_max = V_A.min(), V_A.max()
    
    # 정규화 (0~1 범위)
    U_A_norm = (U_A - U_A_min) / (U_A_max - U_A_min + 1e-10)
    V_A_norm = (V_A - V_A_min) / (V_A_max - V_A_min + 1e-10)
    
    # 리사이즈 (0~1 범위 유지)
    U_A_resized_norm = np.array(Image.fromarray((U_A_norm * 255).astype(np.uint8))
                                .resize((new_w_A, new_h_A), Image.LANCZOS), dtype=np.float64) / 255.0
    V_A_resized_norm = np.array(Image.fromarray((V_A_norm * 255).astype(np.uint8))
                                .resize((new_w_A, new_h_A), Image.LANCZOS), dtype=np.float64) / 255.0
    
    # 원래 범위로 복원
    U_A_resized = U_A_resized_norm * (U_A_max - U_A_min) + U_A_min
    V_A_resized = V_A_resized_norm * (V_A_max - V_A_min) + V_A_min
    
    # 이미지 A를 이미지 B 크기의 캔버스에 중앙 배치
    Y_A_padded = np.zeros((h_B, w_B), dtype=Y_A_resized.dtype)
    # U, V는 손바닥의 값으로 채우기 (0이 아닌) - 색상 왜곡 방지
    U_A_padded = U_B.copy()  # 손바닥 U 채널로 시작
    V_A_padded = V_B.copy()  # 손바닥 V 채널로 시작
    
    # 중앙에 배치
    h_start = (h_B - new_h_A) // 2
    w_start = (w_B - new_w_A) // 2
    h_end = h_start + new_h_A
    w_end = w_start + new_w_A
    
    # 경계 체크
    h_start = max(0, h_start)
    w_start = max(0, w_start)
    h_end = min(h_B, h_end)
    w_end = min(w_B, w_end)
    
    # 실제 복사할 크기
    h_copy = h_end - h_start
    w_copy = w_end - w_start
    
    # 리사이즈된 이미지에서 복사할 부분
    h_A_start = max(0, -h_start)
    w_A_start = max(0, -w_start)
    h_A_end = h_A_start + h_copy
    w_A_end = w_A_start + w_copy
    
    Y_A_padded[h_start:h_end, w_start:w_end] = Y_A_resized[h_A_start:h_A_end, w_A_start:w_A_end]
    # U, V는 눈 영역에만 복사 (나머지는 이미 손바닥 값)
    U_A_padded[h_start:h_end, w_start:w_end] = U_A_resized[h_A_start:h_A_end, w_A_start:w_A_end]
    V_A_padded[h_start:h_end, w_start:w_end] = V_A_resized[h_A_start:h_A_end, w_A_start:w_A_end]
    
    # 최종 이미지
    Y_A = Y_A_padded
    Y_B = Y_B  # 이미지 B는 그대로
    U_A = U_A_padded
    V_A = V_A_padded
    U_B = U_B
    V_B = V_B
    
    # 최종 크기
    h_min, w_min = h_B, w_B
    
    # Step 2 & 3: Gaussian Pyramid 생성 (Y, U, V 모두)
    gaussian_A_Y = build_gaussian_pyramid(Y_A, levels, sigma, kernel_size)
    gaussian_B_Y = build_gaussian_pyramid(Y_B, levels, sigma, kernel_size)
    gaussian_A_U = build_gaussian_pyramid(U_A, levels, sigma, kernel_size)
    gaussian_B_U = build_gaussian_pyramid(U_B, levels, sigma, kernel_size)
    gaussian_A_V = build_gaussian_pyramid(V_A, levels, sigma, kernel_size)
    gaussian_B_V = build_gaussian_pyramid(V_B, levels, sigma, kernel_size)
    
    # Step 4: Laplacian Pyramid 생성 (Y, U, V 모두)
    laplacian_A_Y = build_laplacian_pyramid(gaussian_A_Y, sigma, kernel_size)
    laplacian_B_Y = build_laplacian_pyramid(gaussian_B_Y, sigma, kernel_size)
    laplacian_A_U = build_laplacian_pyramid(gaussian_A_U, sigma, kernel_size)
    laplacian_B_U = build_laplacian_pyramid(gaussian_B_U, sigma, kernel_size)
    laplacian_A_V = build_laplacian_pyramid(gaussian_A_V, sigma, kernel_size)
    laplacian_B_V = build_laplacian_pyramid(gaussian_B_V, sigma, kernel_size)
    
    # Step 5: Mask 생성 및 블렌딩
    # 이미지 A(눈)가 중앙에 배치되어 있으므로 타원형 마스크 생성
    mask = np.zeros((h_min, w_min), dtype=np.float64)
    
    # 이미지 A가 배치된 영역 계산
    eye_center_h = h_start + new_h_A // 2
    eye_center_w = w_start + new_w_A // 2
    
    # 타원형 마스크 생성 (이미지 A 영역 중심)
    y, x = np.ogrid[:h_min, :w_min]
    
    # 타원형 거리 계산
    eye_radius_h = new_h_A // 2 + gradient_width
    eye_radius_w = new_w_A // 2 + gradient_width
    
    # 타원형 거리
    dist_h = np.abs(y - eye_center_h)
    dist_w = np.abs(x - eye_center_w)
    dist_normalized = np.sqrt((dist_h / eye_radius_h)**2 + (dist_w / eye_radius_w)**2)
    
    # 그라데이션 마스크 (중앙이 1, 주변이 0)
    mask = np.where(dist_normalized < 1.0,
                   0.5 * (1 + np.cos(np.pi * dist_normalized)),
                   0.0)
    
    mask_pyramid = create_mask_pyramid(mask, levels, sigma, kernel_size)
    
    # Y, U, V 채널 모두 블렌딩
    blended_laplacian_Y = blend_laplacian_pyramids(laplacian_A_Y, laplacian_B_Y, mask_pyramid)
    blended_laplacian_U = blend_laplacian_pyramids(laplacian_A_U, laplacian_B_U, mask_pyramid)
    blended_laplacian_V = blend_laplacian_pyramids(laplacian_A_V, laplacian_B_V, mask_pyramid)
    
    # Step 6: Pyramid Collapse (Y, U, V 모두)
    Y_blended = collapse_laplacian_pyramid(blended_laplacian_Y, sigma, kernel_size)
    U_blended = collapse_laplacian_pyramid(blended_laplacian_U, sigma, kernel_size)
    V_blended = collapse_laplacian_pyramid(blended_laplacian_V, sigma, kernel_size)
    
    # Step 7: YUV 재결합 및 RGB 복원
    # U, V 채널은 이미 pyramid collapse로 블렌딩 완료
    rgb_result = combine_yuv_channels(Y_blended, U_blended, V_blended)
    
    # 중간 결과 저장
    if save_intermediate and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'step2_gaussian_pyramid_A'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'step2_gaussian_pyramid_B'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'step4_laplacian_pyramid_A'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'step4_laplacian_pyramid_B'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'step5_blended_laplacian'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'step5_mask_pyramid'), exist_ok=True)
        
        # Gaussian Pyramid A 저장 (색상 포함 - Y, U, V 모두 다운샘플링됨)
        fig, axes = plt.subplots(1, len(gaussian_A_Y), figsize=(4*len(gaussian_A_Y), 4))
        for i, level_Y in enumerate(gaussian_A_Y):
            level_U = gaussian_A_U[i]
            level_V = gaussian_A_V[i]
            
            # YUV → RGB 변환
            rgb_level = yuv_to_rgb(level_Y, level_U, level_V)
            
            axes[i].imshow(rgb_level.astype(np.uint8))
            axes[i].set_title(f'Level {i}')
            axes[i].axis('off')
            Image.fromarray(rgb_level.astype(np.uint8)).save(
                os.path.join(output_dir, 'step2_gaussian_pyramid_A', f'level_{i}.png')
            )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'step2_gaussian_pyramid_A', 'pyramid_overview.png'), dpi=150)
        plt.close()
        
        # Gaussian Pyramid B 저장 (색상 포함 - Y, U, V 모두 다운샘플링됨)
        fig, axes = plt.subplots(1, len(gaussian_B_Y), figsize=(4*len(gaussian_B_Y), 4))
        for i, level_Y in enumerate(gaussian_B_Y):
            level_U = gaussian_B_U[i]
            level_V = gaussian_B_V[i]
            
            # YUV → RGB 변환
            rgb_level = yuv_to_rgb(level_Y, level_U, level_V)
            
            axes[i].imshow(rgb_level.astype(np.uint8))
            axes[i].set_title(f'Level {i}')
            axes[i].axis('off')
            Image.fromarray(rgb_level.astype(np.uint8)).save(
                os.path.join(output_dir, 'step2_gaussian_pyramid_B', f'level_{i}.png')
            )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'step2_gaussian_pyramid_B', 'pyramid_overview.png'), dpi=150)
        plt.close()
        
        # Laplacian Pyramid A 저장 (값 범위 조정 필요)
        fig, axes = plt.subplots(1, len(laplacian_A_Y), figsize=(4*len(laplacian_A_Y), 4))
        for i, level in enumerate(laplacian_A_Y):
            # Laplacian은 음수값도 있으므로 시각화를 위해 조정
            level_vis = (level - level.min()) / (level.max() - level.min() + 1e-10) * 255
            axes[i].imshow(level_vis, cmap='gray')
            axes[i].set_title(f'Level {i}')
            axes[i].axis('off')
            Image.fromarray(level_vis.astype(np.uint8)).save(
                os.path.join(output_dir, 'step4_laplacian_pyramid_A', f'level_{i}.png')
            )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'step4_laplacian_pyramid_A', 'pyramid_overview.png'), dpi=150)
        plt.close()
        
        # Laplacian Pyramid B 저장
        fig, axes = plt.subplots(1, len(laplacian_B_Y), figsize=(4*len(laplacian_B_Y), 4))
        for i, level in enumerate(laplacian_B_Y):
            level_vis = (level - level.min()) / (level.max() - level.min() + 1e-10) * 255
            axes[i].imshow(level_vis, cmap='gray')
            axes[i].set_title(f'Level {i}')
            axes[i].axis('off')
            Image.fromarray(level_vis.astype(np.uint8)).save(
                os.path.join(output_dir, 'step4_laplacian_pyramid_B', f'level_{i}.png')
            )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'step4_laplacian_pyramid_B', 'pyramid_overview.png'), dpi=150)
        plt.close()
        
        # Blended Laplacian Pyramid 저장
        fig, axes = plt.subplots(1, len(blended_laplacian_Y), figsize=(4*len(blended_laplacian_Y), 4))
        for i, level in enumerate(blended_laplacian_Y):
            level_vis = (level - level.min()) / (level.max() - level.min() + 1e-10) * 255
            axes[i].imshow(level_vis, cmap='gray')
            axes[i].set_title(f'Level {i}')
            axes[i].axis('off')
            Image.fromarray(level_vis.astype(np.uint8)).save(
                os.path.join(output_dir, 'step5_blended_laplacian', f'level_{i}.png')
            )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'step5_blended_laplacian', 'pyramid_overview.png'), dpi=150)
        plt.close()
        
        # Mask Pyramid 저장
        fig, axes = plt.subplots(1, len(mask_pyramid), figsize=(4*len(mask_pyramid), 4))
        for i, level_mask in enumerate(mask_pyramid):
            mask_vis = (level_mask * 255).astype(np.uint8)
            axes[i].imshow(mask_vis, cmap='gray')
            axes[i].set_title(f'Level {i}')
            axes[i].axis('off')
            Image.fromarray(mask_vis).save(
                os.path.join(output_dir, 'step5_mask_pyramid', f'level_{i}.png')
            )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'step5_mask_pyramid', 'pyramid_overview.png'), dpi=150)
        plt.close()
    
    return rgb_result.astype(np.uint8)


if __name__ == "__main__":
    print("Image blending functions are ready!")
    print("Use blend_images() function when image files are ready.")

