"""
ROI 영역에 eye 이미지를 Laplacian Pyramid 방식으로 합성
- 타원 마스크를 사용한 블렌딩
- ROI 내부에서만 블렌딩, padding 없이 crop
"""

import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import os


# ============================================================================
# Step 1: 이미지 로드 및 전처리
# ============================================================================

def load_image_rgb(image_path):
    """이미지를 RGB로 로드 (EXIF orientation 자동 적용)"""
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert('RGB')
    return np.array(img, dtype=np.float64)


def rgb_to_yuv(rgb_image):
    """RGB를 YUV로 변환"""
    R, G, B = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.14713 * R - 0.28886 * G + 0.436 * B
    V = 0.615 * R - 0.51499 * G - 0.10001 * B
    return Y, U, V


def yuv_to_rgb(Y, U, V):
    """YUV를 RGB로 변환"""
    R = Y + 1.13983 * V
    G = Y - 0.39465 * U - 0.58060 * V
    B = Y + 2.03211 * U
    rgb = np.stack([R, G, B], axis=2)
    rgb = np.clip(rgb, 0, 255)
    return rgb


def crop_roi(image, roi_coords):
    """이미지에서 ROI 영역을 crop"""
    y_start, y_end, x_start, x_end = roi_coords
    return image[y_start:y_end, x_start:x_end]


def prepare_roi_images(hand_path, eye_path, roi_coords):
    """
    hand와 eye 이미지를 준비하고 ROI 영역에 맞춤 (padding 없이 crop)
    
    Args:
        hand_path: hand 이미지 경로
        eye_path: eye 이미지 경로
        roi_coords: (y_start, y_end, x_start, x_end) ROI 좌표
    
    Returns:
        tuple: (hand_Y_roi, hand_U_roi, hand_V_roi,
                eye_Y_roi, eye_U_roi, eye_V_roi,
                hand_Y_full, hand_U_full, hand_V_full)
    """
    # 이미지 로드
    hand_rgb = load_image_rgb(hand_path)
    eye_rgb = load_image_rgb(eye_path)
    
    # YUV 변환
    hand_Y, hand_U, hand_V = rgb_to_yuv(hand_rgb)
    eye_Y, eye_U, eye_V = rgb_to_yuv(eye_rgb)
    
    # hand에서 ROI crop
    hand_Y_roi = crop_roi(hand_Y, roi_coords)
    hand_U_roi = crop_roi(hand_U, roi_coords)
    hand_V_roi = crop_roi(hand_V, roi_coords)
    
    # eye를 ROI 크기에 정확히 맞춰 리사이즈 (padding 없음, 비율 무시)
    roi_h, roi_w = hand_Y_roi.shape
    
    # PIL로 정확히 ROI 크기로 리사이즈
    eye_Y_pil = Image.fromarray(eye_Y.astype(np.uint8))
    eye_Y_roi = np.array(eye_Y_pil.resize((roi_w, roi_h), Image.LANCZOS), dtype=np.float64)
    
    # U, V 채널도 리사이즈
    eye_U_min, eye_U_max = eye_U.min(), eye_U.max()
    eye_V_min, eye_V_max = eye_V.min(), eye_V.max()
    
    eye_U_norm = (eye_U - eye_U_min) / (eye_U_max - eye_U_min + 1e-10)
    eye_V_norm = (eye_V - eye_V_min) / (eye_V_max - eye_V_min + 1e-10)
    
    eye_U_resized_norm = np.array(Image.fromarray((eye_U_norm * 255).astype(np.uint8))
                                  .resize((roi_w, roi_h), Image.LANCZOS), dtype=np.float64) / 255.0
    eye_V_resized_norm = np.array(Image.fromarray((eye_V_norm * 255).astype(np.uint8))
                                  .resize((roi_w, roi_h), Image.LANCZOS), dtype=np.float64) / 255.0
    
    eye_U_roi = eye_U_resized_norm * (eye_U_max - eye_U_min) + eye_U_min
    eye_V_roi = eye_V_resized_norm * (eye_V_max - eye_V_min) + eye_V_min
    
    return (hand_Y_roi, hand_U_roi, hand_V_roi,
            eye_Y_roi, eye_U_roi, eye_V_roi,
            hand_Y, hand_U, hand_V)


def visualize_step(title, images, labels, save_path=None, cmap=None):
    """단계별 결과 시각화"""
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
    if n_images == 1:
        axes = [axes]
    
    for i, (img, label) in enumerate(zip(images, labels)):
        if cmap and len(img.shape) == 2:
            axes[i].imshow(img, cmap=cmap)
        else:
            if len(img.shape) == 2:
                axes[i].imshow(img, cmap='gray')
            else:
                axes[i].imshow(img.astype(np.uint8))
        axes[i].set_title(label)
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장: {save_path}")
    
    plt.close()


# ============================================================================
# Step 2: Gaussian Kernel & Filtering
# ============================================================================

def create_gaussian_kernel(sigma, kernel_size=None):
    """2D Gaussian 커널 생성"""
    if kernel_size is None:
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
    
    center = kernel_size // 2
    x, y = np.meshgrid(np.arange(kernel_size) - center, 
                       np.arange(kernel_size) - center)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel


def gaussian_filter_2d(image, sigma, kernel_size=None):
    """Gaussian 필터 적용"""
    kernel = create_gaussian_kernel(sigma, kernel_size)
    filtered = convolve2d(image, kernel, mode='same', boundary='symm')
    return filtered


def downsample(image, factor=2):
    """이미지 다운샘플링"""
    return image[::factor, ::factor]


# ============================================================================
# Step 3: Gaussian Pyramid 생성
# ============================================================================

def build_gaussian_pyramid(image, levels, sigma=1.0, kernel_size=None):
    """Gaussian Pyramid 생성"""
    pyramid = [image.copy()]
    
    for i in range(levels - 1):
        # Gaussian 필터 적용
        filtered = gaussian_filter_2d(pyramid[-1], sigma, kernel_size)
        # 다운샘플링
        downsampled = downsample(filtered)
        pyramid.append(downsampled)
    
    return pyramid


def visualize_gaussian_pyramid(pyramid, title, save_path=None, U_pyramid=None, V_pyramid=None, is_mask=False):
    """Gaussian Pyramid 시각화 (Y, U, V 채널 모두 처리)"""
    n_levels = len(pyramid)
    fig, axes = plt.subplots(1, n_levels, figsize=(4*n_levels, 4))
    if n_levels == 1:
        axes = [axes]
    
    for i, level in enumerate(pyramid):
        if U_pyramid is not None and V_pyramid is not None:
            # Y, U, V를 RGB로 변환
            rgb_level = yuv_to_rgb(level, U_pyramid[i], V_pyramid[i])
            axes[i].imshow(rgb_level.astype(np.uint8))
        elif is_mask:
            # Mask는 0~1 범위이므로 0~255로 변환
            mask_vis = (level * 255).astype(np.uint8)
            axes[i].imshow(mask_vis, cmap='gray', vmin=0, vmax=255)
        else:
            # Y 채널만 grayscale로 표시
            axes[i].imshow(level.astype(np.uint8), cmap='gray')
        axes[i].set_title(f'Level {i}\n{level.shape[1]}x{level.shape[0]}')
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장: {save_path}")
    
    plt.close()


# ============================================================================
# Step 4: Laplacian Pyramid 생성
# ============================================================================

def upsample(image, target_shape):
    """
    이미지 업샘플링 (음수값을 가진 채널도 처리)
    """
    h, w = image.shape
    target_h, target_w = target_shape
    
    # 음수값을 가진 채널도 처리하기 위해 정규화
    img_min = image.min()
    img_max = image.max()
    img_range = img_max - img_min
    
    if img_range > 1e-10:
        # 정규화 (0~1 범위)
        img_norm = (image - img_min) / img_range
        
        # PIL로 리사이즈 (0~1 범위)
        img_pil = Image.fromarray((img_norm * 255).astype(np.uint8))
        upsampled_norm = np.array(img_pil.resize((target_w, target_h), Image.LANCZOS), dtype=np.float64) / 255.0
        
        # 원래 범위로 복원
        upsampled = upsampled_norm * img_range + img_min
    else:
        # 범위가 거의 없으면 그냥 리사이즈
        img_pil = Image.fromarray(image.astype(np.uint8))
        upsampled = np.array(img_pil.resize((target_w, target_h), Image.LANCZOS), dtype=np.float64)
    
    return upsampled


def build_laplacian_pyramid(gaussian_pyramid, sigma=1.0, kernel_size=None):
    """
    Laplacian Pyramid 생성 (일반 구조)
    LPI(L) = GPI(L) - upsample(GPI(L+1)) for L = 0 to levels-2
    마지막 레벨은 포함하지 않음 (levels-1개만 생성)
    """
    laplacian_pyramid = []
    
    for i in range(len(gaussian_pyramid) - 1):
        # 현재 레벨
        current = gaussian_pyramid[i]
        # 다음 레벨을 업샘플링
        next_level = gaussian_pyramid[i + 1]
        upsampled = upsample(next_level, current.shape)
        
        # Gaussian 필터 적용 (업샘플링된 이미지에)
        upsampled_filtered = gaussian_filter_2d(upsampled, sigma, kernel_size)
        
        # Laplacian = 현재 - 업샘플링된 다음 레벨
        laplacian = current - upsampled_filtered
        laplacian_pyramid.append(laplacian)
    
    # 마지막 레벨은 포함하지 않음 (일반 구조)
    return laplacian_pyramid


def visualize_laplacian_pyramid(pyramid, title, save_path=None):
    """Laplacian Pyramid 시각화"""
    n_levels = len(pyramid)
    fig, axes = plt.subplots(1, n_levels, figsize=(4*n_levels, 4))
    if n_levels == 1:
        axes = [axes]
    
    for i, level in enumerate(pyramid):
        # Laplacian은 음수값을 가질 수 있으므로 정규화
        level_normalized = (level - level.min()) / (level.max() - level.min() + 1e-10) * 255
        axes[i].imshow(level_normalized.astype(np.uint8), cmap='gray')
        axes[i].set_title(f'Level {i}\n{level.shape[1]}x{level.shape[0]}')
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장: {save_path}")
    
    plt.close()


# ============================================================================
# Step 5: 타원 마스크 생성
# ============================================================================

def create_ellipse_mask(roi_shape, center_ratio=(0.5, 0.5), size_ratio=(0.8, 0.8), sigma_blur=3.0):
    """
    ROI 영역에 타원 마스크 생성
    
    Args:
        roi_shape: (height, width) ROI 크기
        center_ratio: (y_ratio, x_ratio) 타원 중심 위치 비율 (0~1, 기본값: (0.5, 0.5) = 중심)
        size_ratio: (y_ratio, x_ratio) 타원 크기 비율 (0~1, 기본값: (0.8, 0.8) = ROI의 80%)
        sigma_blur: Gaussian blur 표준편차
    
    Returns:
        numpy array: 0~1 범위의 타원 마스크
    """
    h, w = roi_shape
    center_y = int(h * center_ratio[0])
    center_x = int(w * center_ratio[1])
    
    # 타원 반지름 (ROI 크기의 비율로)
    radius_y = int(h * size_ratio[0] / 2)
    radius_x = int(w * size_ratio[1] / 2)
    
    # 좌표 그리드 생성
    y, x = np.ogrid[:h, :w]
    
    # 타원 방정식: ((x-cx)/rx)² + ((y-cy)/ry)² <= 1
    ellipse_mask = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 <= 1
    
    # float로 변환
    ellipse_mask = ellipse_mask.astype(np.float64)
    
    # Gaussian blur로 부드럽게
    ellipse_mask = gaussian_filter(ellipse_mask, sigma=sigma_blur)
    
    # 0~1 범위로 정규화
    ellipse_mask = (ellipse_mask - ellipse_mask.min()) / (ellipse_mask.max() - ellipse_mask.min() + 1e-10)
    
    return ellipse_mask


def create_mask_pyramid(mask, levels, sigma=1.0, kernel_size=None):
    """Mask Pyramid 생성"""
    mask_pyramid = [mask.copy()]
    
    for i in range(levels - 1):
        # Gaussian 필터 적용
        filtered = gaussian_filter_2d(mask_pyramid[-1], sigma, kernel_size)
        # 다운샘플링
        downsampled = downsample(filtered)
        mask_pyramid.append(downsampled)
    
    return mask_pyramid


def visualize_mask(mask, title, save_path=None):
    """Mask 시각화"""
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar()
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장: {save_path}")
    
    plt.close()


# ============================================================================
# Step 6: Laplacian Pyramid Blending
# ============================================================================

def blend_laplacian_pyramids(laplacian_hand, laplacian_eye, mask_pyramid):
    """
    두 Laplacian Pyramid를 마스크로 블렌딩
    blended = eye * mask + hand * (1 - mask)
    """
    blended_pyramid = []
    
    for i in range(len(laplacian_hand)):
        lap_hand = laplacian_hand[i]
        lap_eye = laplacian_eye[i]
        mask = mask_pyramid[i]
        
        # 크기 맞추기
        h_min = min(lap_hand.shape[0], lap_eye.shape[0], mask.shape[0])
        w_min = min(lap_hand.shape[1], lap_eye.shape[1], mask.shape[1])
        
        lap_hand = lap_hand[:h_min, :w_min]
        lap_eye = lap_eye[:h_min, :w_min]
        mask = mask[:h_min, :w_min]
        
        # 블렌딩: eye * mask + hand * (1 - mask)
        blended = lap_eye * mask + lap_hand * (1 - mask)
        blended_pyramid.append(blended)
    
    return blended_pyramid


def visualize_blended_pyramid(pyramid, title, save_path=None):
    """Blended Laplacian Pyramid 시각화"""
    n_levels = len(pyramid)
    fig, axes = plt.subplots(1, n_levels, figsize=(4*n_levels, 4))
    if n_levels == 1:
        axes = [axes]
    
    for i, level in enumerate(pyramid):
        # 정규화
        level_normalized = (level - level.min()) / (level.max() - level.min() + 1e-10) * 255
        axes[i].imshow(level_normalized.astype(np.uint8), cmap='gray')
        axes[i].set_title(f'Level {i}\n{level.shape[1]}x{level.shape[0]}')
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장: {save_path}")
    
    plt.close()


# ============================================================================
# Step 7: Pyramid Collapse (복원)
# ============================================================================

def collapse_laplacian_pyramid(laplacian_pyramid, gaussian_base, sigma=1.0, kernel_size=None, return_steps=False):
    """
    Laplacian Pyramid를 복원하여 원본 이미지 재구성 (일반 구조)
    Gaussian의 마지막 레벨에서 시작하여 Laplacian을 더해감
    
    Args:
        laplacian_pyramid: Laplacian Pyramid (levels-1개)
        gaussian_base: Gaussian Pyramid의 마지막 레벨 (복원의 시작점)
        return_steps: True이면 복원 과정의 각 단계를 리스트로 반환
    
    Returns:
        reconstructed: 복원된 이미지
        reconstruction_steps: return_steps=True일 때 각 단계의 이미지 리스트
    """
    # Gaussian의 마지막 레벨에서 시작 (일반 구조)
    reconstructed = gaussian_base.copy()
    reconstruction_steps = [reconstructed.copy()] if return_steps else None
    
    # 역순으로 업샘플링하고 Laplacian 더하기
    for i in range(len(laplacian_pyramid) - 1, -1, -1):
        # 목표 크기
        target_shape = laplacian_pyramid[i].shape
        
        # 업샘플링
        upsampled = upsample(reconstructed, target_shape)
        
        # Gaussian 필터 적용
        upsampled_filtered = gaussian_filter_2d(upsampled, sigma, kernel_size)
        
        # Laplacian 더하기
        reconstructed = upsampled_filtered + laplacian_pyramid[i]
        
        if return_steps:
            reconstruction_steps.append(reconstructed.copy())
    
    if return_steps:
        return reconstructed, reconstruction_steps
    return reconstructed


# ============================================================================
# 메인 블렌딩 파이프라인
# ============================================================================

def roi_eye_blend(hand_path, eye_path, roi_coords, output_dir='test3', 
                  levels=5, sigma=1.0, sigma_blur=3.0,
                  ellipse_center=(0.5, 0.5), ellipse_size=(0.8, 0.8)):
    """
    hand 이미지의 ROI 영역에 eye 이미지를 Laplacian Pyramid로 블렌딩 (타원 마스크 사용)
    
    Args:
        hand_path: hand 이미지 경로
        eye_path: eye 이미지 경로
        roi_coords: (y_start, y_end, x_start, x_end) ROI 좌표
        output_dir: 결과 저장 디렉토리
        levels: 피라미드 레벨 수
        sigma: Gaussian 표준편차
        sigma_blur: 타원 마스크 blur 표준편차
        ellipse_center: 타원 중심 위치 비율 (y_ratio, x_ratio)
        ellipse_size: 타원 크기 비율 (y_ratio, x_ratio)
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'step1_preparation'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'step2_gaussian_pyramid_hand'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'step2_gaussian_pyramid_eye'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'step3_laplacian_pyramid_hand'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'step3_laplacian_pyramid_eye'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'step4_ellipse_mask'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'step5_blended_laplacian'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'step6_final'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'step7_pyramid_visualization'), exist_ok=True)
    
    print("=" * 60)
    print("ROI 기반 Eye Blending 시작 (타원 마스크 사용)")
    print("=" * 60)
    
    # Step 1: 이미지 준비 및 ROI crop
    print("\n[Step 1] 이미지 준비 및 ROI crop")
    (hand_Y_roi, hand_U_roi, hand_V_roi,
     eye_Y_roi, eye_U_roi, eye_V_roi,
     hand_Y_full, hand_U_full, hand_V_full) = prepare_roi_images(
        hand_path, eye_path, roi_coords
    )
    
    # Step 1 시각화
    hand_roi_rgb = yuv_to_rgb(hand_Y_roi, hand_U_roi, hand_V_roi)
    eye_roi_rgb = yuv_to_rgb(eye_Y_roi, eye_U_roi, eye_V_roi)
    visualize_step("Step 1: ROI 영역 crop", 
                   [hand_roi_rgb, eye_roi_rgb],
                   ["Hand ROI", "Eye ROI"],
                   os.path.join(output_dir, 'step1_preparation', 'roi_crop.png'))
    
    # Step 2: Gaussian Pyramid 생성 (Y, U, V 모두)
    print("\n[Step 2] Gaussian Pyramid 생성 (Y, U, V 채널 모두)")
    gaussian_hand_Y = build_gaussian_pyramid(hand_Y_roi, levels, sigma)
    gaussian_hand_U = build_gaussian_pyramid(hand_U_roi, levels, sigma)
    gaussian_hand_V = build_gaussian_pyramid(hand_V_roi, levels, sigma)
    
    gaussian_eye_Y = build_gaussian_pyramid(eye_Y_roi, levels, sigma)
    gaussian_eye_U = build_gaussian_pyramid(eye_U_roi, levels, sigma)
    gaussian_eye_V = build_gaussian_pyramid(eye_V_roi, levels, sigma)
    
    # Y, U, V를 RGB로 변환하여 시각화
    visualize_gaussian_pyramid(gaussian_hand_Y, "Hand ROI Gaussian Pyramid",
                               os.path.join(output_dir, 'step2_gaussian_pyramid_hand', 'pyramid.png'),
                               gaussian_hand_U, gaussian_hand_V)
    visualize_gaussian_pyramid(gaussian_eye_Y, "Eye ROI Gaussian Pyramid",
                               os.path.join(output_dir, 'step2_gaussian_pyramid_eye', 'pyramid.png'),
                               gaussian_eye_U, gaussian_eye_V)
    
    # 각 레벨 저장 (RGB로 변환하여 저장)
    for i, level_Y in enumerate(gaussian_hand_Y):
        level_U = gaussian_hand_U[i]
        level_V = gaussian_hand_V[i]
        rgb_level = yuv_to_rgb(level_Y, level_U, level_V)
        Image.fromarray(rgb_level.astype(np.uint8)).save(
            os.path.join(output_dir, 'step2_gaussian_pyramid_hand', f'level_{i}.png')
        )
    for i, level_Y in enumerate(gaussian_eye_Y):
        level_U = gaussian_eye_U[i]
        level_V = gaussian_eye_V[i]
        rgb_level = yuv_to_rgb(level_Y, level_U, level_V)
        Image.fromarray(rgb_level.astype(np.uint8)).save(
            os.path.join(output_dir, 'step2_gaussian_pyramid_eye', f'level_{i}.png')
        )
    
    # Step 3: Laplacian Pyramid 생성
    print("\n[Step 3] Laplacian Pyramid 생성")
    laplacian_hand_Y = build_laplacian_pyramid(gaussian_hand_Y, sigma)
    laplacian_eye_Y = build_laplacian_pyramid(gaussian_eye_Y, sigma)
    
    visualize_laplacian_pyramid(laplacian_hand_Y, "Hand ROI Laplacian Pyramid",
                               os.path.join(output_dir, 'step3_laplacian_pyramid_hand', 'pyramid.png'))
    visualize_laplacian_pyramid(laplacian_eye_Y, "Eye ROI Laplacian Pyramid",
                               os.path.join(output_dir, 'step3_laplacian_pyramid_eye', 'pyramid.png'))
    
    # 각 레벨 저장
    for i, level in enumerate(laplacian_hand_Y):
        level_normalized = (level - level.min()) / (level.max() - level.min() + 1e-10) * 255
        Image.fromarray(level_normalized.astype(np.uint8)).save(
            os.path.join(output_dir, 'step3_laplacian_pyramid_hand', f'level_{i}.png')
        )
    for i, level in enumerate(laplacian_eye_Y):
        level_normalized = (level - level.min()) / (level.max() - level.min() + 1e-10) * 255
        Image.fromarray(level_normalized.astype(np.uint8)).save(
            os.path.join(output_dir, 'step3_laplacian_pyramid_eye', f'level_{i}.png')
        )
    
    # Step 4: 타원 마스크 생성
    print("\n[Step 4] 타원 마스크 생성")
    roi_h, roi_w = hand_Y_roi.shape
    ellipse_mask = create_ellipse_mask((roi_h, roi_w), ellipse_center, ellipse_size, sigma_blur)
    mask_pyramid = create_mask_pyramid(ellipse_mask, levels, sigma)
    
    visualize_mask(ellipse_mask, "Ellipse Mask", 
                   os.path.join(output_dir, 'step4_ellipse_mask', 'ellipse_mask.png'))
    
    # Mask pyramid 시각화
    visualize_gaussian_pyramid(mask_pyramid, "Ellipse Mask Pyramid",
                              os.path.join(output_dir, 'step4_ellipse_mask', 'mask_pyramid.png'),
                              is_mask=True)
    
    # 각 레벨 저장
    for i, level_mask in enumerate(mask_pyramid):
        mask_vis = (level_mask * 255).astype(np.uint8)
        Image.fromarray(mask_vis).save(
            os.path.join(output_dir, 'step4_ellipse_mask', f'mask_level_{i}.png')
        )
    
    # Step 5: Laplacian Pyramid Blending
    print("\n[Step 5] Laplacian Pyramid Blending")
    blended_laplacian_Y = blend_laplacian_pyramids(laplacian_hand_Y, laplacian_eye_Y, mask_pyramid)
    
    visualize_blended_pyramid(blended_laplacian_Y, "Blended Laplacian Pyramid (Y)",
                              os.path.join(output_dir, 'step5_blended_laplacian', 'pyramid_Y.png'))
    
    # 각 레벨 저장
    for i, level in enumerate(blended_laplacian_Y):
        level_normalized = (level - level.min()) / (level.max() - level.min() + 1e-10) * 255
        Image.fromarray(level_normalized.astype(np.uint8)).save(
            os.path.join(output_dir, 'step5_blended_laplacian', f'level_{i}.png')
        )
    
    # U, V 채널도 같은 방식으로 블렌딩
    laplacian_hand_U = build_laplacian_pyramid(gaussian_hand_U, sigma)
    laplacian_eye_U = build_laplacian_pyramid(gaussian_eye_U, sigma)
    blended_laplacian_U = blend_laplacian_pyramids(laplacian_hand_U, laplacian_eye_U, mask_pyramid)
    
    laplacian_hand_V = build_laplacian_pyramid(gaussian_hand_V, sigma)
    laplacian_eye_V = build_laplacian_pyramid(gaussian_eye_V, sigma)
    blended_laplacian_V = blend_laplacian_pyramids(laplacian_hand_V, laplacian_eye_V, mask_pyramid)
    
    # Step 6: Pyramid Collapse (복원)
    print("\n[Step 6] Pyramid Collapse (복원)")
    # Gaussian의 마지막 레벨을 블렌딩 (일반 구조)
    gaussian_hand_Y_last = gaussian_hand_Y[-1]
    gaussian_eye_Y_last = gaussian_eye_Y[-1]
    mask_last = mask_pyramid[-1]
    
    # 크기 맞추기
    h_min = min(gaussian_hand_Y_last.shape[0], gaussian_eye_Y_last.shape[0], mask_last.shape[0])
    w_min = min(gaussian_hand_Y_last.shape[1], gaussian_eye_Y_last.shape[1], mask_last.shape[1])
    gaussian_hand_Y_last = gaussian_hand_Y_last[:h_min, :w_min]
    gaussian_eye_Y_last = gaussian_eye_Y_last[:h_min, :w_min]
    mask_last = mask_last[:h_min, :w_min]
    
    # 마지막 Gaussian 레벨 블렌딩
    Y_gaussian_base = gaussian_eye_Y_last * mask_last + gaussian_hand_Y_last * (1 - mask_last)
    
    # U, V도 동일하게
    gaussian_hand_U_last = gaussian_hand_U[-1][:h_min, :w_min]
    gaussian_eye_U_last = gaussian_eye_U[-1][:h_min, :w_min]
    U_gaussian_base = gaussian_eye_U_last * mask_last + gaussian_hand_U_last * (1 - mask_last)
    
    gaussian_hand_V_last = gaussian_hand_V[-1][:h_min, :w_min]
    gaussian_eye_V_last = gaussian_eye_V[-1][:h_min, :w_min]
    V_gaussian_base = gaussian_eye_V_last * mask_last + gaussian_hand_V_last * (1 - mask_last)
    
    # Laplacian Pyramid 복원 (일반 구조)
    Y_blended_roi, Y_reconstruction_steps = collapse_laplacian_pyramid(blended_laplacian_Y, Y_gaussian_base, sigma, return_steps=True)
    U_blended_roi = collapse_laplacian_pyramid(blended_laplacian_U, U_gaussian_base, sigma)
    V_blended_roi = collapse_laplacian_pyramid(blended_laplacian_V, V_gaussian_base, sigma)
    
    # YUV → RGB 변환
    result_roi_rgb = yuv_to_rgb(Y_blended_roi, U_blended_roi, V_blended_roi)
    
    # Step 7: 종합 Pyramid 시각화 (Gaussian, Laplacian, Reconstruction)
    print("\n[Step 7] 종합 Pyramid 시각화")
    
    # 블렌딩된 결과의 Gaussian Pyramid 생성
    blended_gaussian_Y = build_gaussian_pyramid(Y_blended_roi, levels, sigma)
    blended_gaussian_U = build_gaussian_pyramid(U_blended_roi, levels, sigma)
    blended_gaussian_V = build_gaussian_pyramid(V_blended_roi, levels, sigma)
    
    # 블렌딩된 결과의 Laplacian Pyramid 생성 (일반 구조: levels-1개)
    blended_laplacian_result_Y = build_laplacian_pyramid(blended_gaussian_Y, sigma)
    
    # 복원 과정을 RGB로 변환
    reconstruction_steps_rgb = []
    for i, step_Y in enumerate(Y_reconstruction_steps):
        # U, V도 같은 레벨로 맞춤
        step_U = blended_gaussian_U[min(i, len(blended_gaussian_U)-1)]
        step_V = blended_gaussian_V[min(i, len(blended_gaussian_V)-1)]
        
        # 크기 맞추기
        h_min = min(step_Y.shape[0], step_U.shape[0], step_V.shape[0])
        w_min = min(step_Y.shape[1], step_U.shape[1], step_V.shape[1])
        step_Y = step_Y[:h_min, :w_min]
        step_U = step_U[:h_min, :w_min]
        step_V = step_V[:h_min, :w_min]
        
        step_rgb = yuv_to_rgb(step_Y, step_U, step_V)
        reconstruction_steps_rgb.append(step_rgb)
    
    # Gaussian Pyramid를 RGB로 변환
    gaussian_pyramid_rgb = []
    for i in range(len(blended_gaussian_Y)):
        rgb_level = yuv_to_rgb(blended_gaussian_Y[i], blended_gaussian_U[i], blended_gaussian_V[i])
        gaussian_pyramid_rgb.append(rgb_level)
    
    # Laplacian Pyramid 정규화 (시각화용)
    laplacian_pyramid_vis = []
    for i, level in enumerate(blended_laplacian_result_Y):
        level_normalized = (level - level.min()) / (level.max() - level.min() + 1e-10) * 255
        laplacian_pyramid_vis.append(level_normalized.astype(np.uint8))
    
    # 종합 시각화 생성 (3행: Gaussian, Laplacian, Reconstruction)
    n_gaussian = len(blended_gaussian_Y)  # levels개
    n_laplacian = len(blended_laplacian_result_Y)  # levels-1개
    n_reconstruction = len(reconstruction_steps_rgb)  # levels개 (마지막 Gaussian + Laplacian들)
    
    # 최대 열 수는 Gaussian의 개수
    n_cols = n_gaussian
    fig = plt.figure(figsize=(4*n_cols, 12))
    
    # 상단 행: Gaussian Pyramid (블렌딩된 이미지) - levels개
    for i in range(n_gaussian):
        ax = plt.subplot(3, n_cols, i + 1)
        ax.imshow(gaussian_pyramid_rgb[i].astype(np.uint8))
        ax.set_title(f'Level {i}\n{gaussian_pyramid_rgb[i].shape[1]}x{gaussian_pyramid_rgb[i].shape[0]}', fontsize=10)
        ax.axis('off')
    
    # 중간 행: Laplacian Pyramid (디테일 레이어) - levels-1개
    for i in range(n_cols):
        ax = plt.subplot(3, n_cols, n_cols + i + 1)
        if i < n_laplacian:
            ax.imshow(laplacian_pyramid_vis[i], cmap='gray')
            ax.set_title(f'Level {i}\n{laplacian_pyramid_vis[i].shape[1]}x{laplacian_pyramid_vis[i].shape[0]}', fontsize=10)
        else:
            # 마지막 열은 비워둠 (Laplacian은 levels-1개만 있음)
            ax.axis('off')
            ax.set_title(f'Level {i}\n(no Laplacian)', fontsize=10, style='italic')
        ax.axis('off')
    
    # 하단 행: Reconstruction (복원 과정) - levels개
    # reconstruction_steps는 마지막 Gaussian에서 시작하여 Laplacian을 더해감
    for i in range(n_cols):
        ax = plt.subplot(3, n_cols, 2*n_cols + i + 1)
        if i < n_reconstruction:
            idx = n_reconstruction - 1 - i  # 역순 인덱스
            ax.imshow(reconstruction_steps_rgb[idx].astype(np.uint8))
            level_num = n_gaussian - 1 - i
            if i == n_reconstruction - 1:
                ax.set_title(f'Level {level_num}\n(final result!)', fontsize=10, fontweight='bold')
            else:
                ax.set_title(f'Level {level_num}\n{reconstruction_steps_rgb[idx].shape[1]}x{reconstruction_steps_rgb[idx].shape[0]}', fontsize=10)
        else:
            ax.axis('off')
        ax.axis('off')
    
    # 행 제목
    fig.text(0.5, 0.95, 'Gaussian Pyramid (Blended Image)', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.64, 'Laplacian Pyramid (Detail Layers)', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.32, 'Reconstructed Blended Image', ha='center', fontsize=14, fontweight='bold')
    
    plt.suptitle('Image Blending using Image Pyramids (General Structure)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_dir, 'step7_pyramid_visualization', 'pyramid_complete.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"저장: {save_path}")
    plt.close()
    
    # Step 6: ROI 영역을 원본 hand에 삽입 (여러 방법 비교)
    print("\n[Step 6] ROI 영역을 원본 hand에 삽입 (여러 방법 비교)")
    y_start, y_end, x_start, x_end = roi_coords
    roi_h, roi_w = y_end - y_start, x_end - x_start
    
    # ===== 방법 4: 경계 부분에 추가 Gaussian Blur =====
    print("\n[방법 4] 경계 부분에 추가 Gaussian Blur 적용")
    result_full_Y_method4 = hand_Y_full.copy()
    result_full_U_method4 = hand_U_full.copy()
    result_full_V_method4 = hand_V_full.copy()
    
    # ROI 영역에 블렌딩된 결과 삽입
    result_full_Y_method4[y_start:y_end, x_start:x_end] = Y_blended_roi
    result_full_U_method4[y_start:y_end, x_start:x_end] = U_blended_roi
    result_full_V_method4[y_start:y_end, x_start:x_end] = V_blended_roi
    
    # 경계 영역만 선택적으로 blur
    border_region = np.zeros_like(result_full_Y_method4)
    border_region[y_start:y_end, x_start:x_end] = 1.0
    border_region = gaussian_filter(border_region, sigma=5.0)
    
    # 경계 부분만 blur 적용
    blurred_Y = gaussian_filter(result_full_Y_method4, sigma=1.0)
    blurred_U = gaussian_filter(result_full_U_method4, sigma=1.0)
    blurred_V = gaussian_filter(result_full_V_method4, sigma=1.0)
    
    result_full_Y_method4 = result_full_Y_method4 * (1 - border_region) + blurred_Y * border_region
    result_full_U_method4 = result_full_U_method4 * (1 - border_region) + blurred_U * border_region
    result_full_V_method4 = result_full_V_method4 * (1 - border_region) + blurred_V * border_region
    
    # RGB 변환
    result_full_rgb_method4 = yuv_to_rgb(result_full_Y_method4, result_full_U_method4, result_full_V_method4)
    
    # ===== 방법 1: Feathering Mask 적용 =====
    print("\n[방법 1] Feathering Mask 적용")
    result_full_Y_method1 = hand_Y_full.copy()
    result_full_U_method1 = hand_U_full.copy()
    result_full_V_method1 = hand_V_full.copy()
    
    # ROI 경계에 feathering mask 생성
    feather_width = 30  # 경계 확장 폭
    feather_mask = np.ones((roi_h, roi_w), dtype=np.float64)
    
    # 경계 부분에 그라데이션 적용
    y, x = np.ogrid[:roi_h, :roi_w]
    dist_to_edge = np.minimum(
        np.minimum(y, roi_h - 1 - y),  # 상하 경계까지 거리
        np.minimum(x, roi_w - 1 - x)   # 좌우 경계까지 거리
    )
    
    # feather_width 내에서 그라데이션
    feather_mask = np.clip(dist_to_edge / feather_width, 0, 1)
    
    # Gaussian blur로 더 부드럽게
    feather_mask = gaussian_filter(feather_mask, sigma=3.0)
    
    # ROI 영역에 feathering 적용하여 삽입
    hand_Y_roi_original = hand_Y_full[y_start:y_end, x_start:x_end]
    hand_U_roi_original = hand_U_full[y_start:y_end, x_start:x_end]
    hand_V_roi_original = hand_V_full[y_start:y_end, x_start:x_end]
    
    result_full_Y_method1[y_start:y_end, x_start:x_end] = (
        Y_blended_roi * feather_mask + 
        hand_Y_roi_original * (1 - feather_mask)
    )
    result_full_U_method1[y_start:y_end, x_start:x_end] = (
        U_blended_roi * feather_mask + 
        hand_U_roi_original * (1 - feather_mask)
    )
    result_full_V_method1[y_start:y_end, x_start:x_end] = (
        V_blended_roi * feather_mask + 
        hand_V_roi_original * (1 - feather_mask)
    )
    
    # RGB 변환
    result_full_rgb_method1 = yuv_to_rgb(result_full_Y_method1, result_full_U_method1, result_full_V_method1)
    
    # ===== 원본 (경계 처리 없음) =====
    result_full_Y_original = hand_Y_full.copy()
    result_full_U_original = hand_U_full.copy()
    result_full_V_original = hand_V_full.copy()
    
    result_full_Y_original[y_start:y_end, x_start:x_end] = Y_blended_roi
    result_full_U_original[y_start:y_end, x_start:x_end] = U_blended_roi
    result_full_V_original[y_start:y_end, x_start:x_end] = V_blended_roi
    
    result_full_rgb_original = yuv_to_rgb(result_full_Y_original, result_full_U_original, result_full_V_original)
    
    # Step 6 시각화 (비교)
    visualize_step("Step 6: 경계 처리 방법 비교", 
                   [result_full_rgb_original, result_full_rgb_method4, result_full_rgb_method1],
                   ["Original (경계 없음)", "Method 4 (Gaussian Blur)", "Method 1 (Feathering)"],
                   os.path.join(output_dir, 'step6_final', 'comparison.png'))
    
    # 각 방법별 결과 저장
    result_img_original = Image.fromarray(result_full_rgb_original.astype(np.uint8))
    result_img_original.save(os.path.join(output_dir, 'step6_final', 'blended_result_original.jpg'), quality=95)
    
    result_img_method4 = Image.fromarray(result_full_rgb_method4.astype(np.uint8))
    result_img_method4.save(os.path.join(output_dir, 'step6_final', 'blended_result_method4_blur.jpg'), quality=95)
    
    result_img_method1 = Image.fromarray(result_full_rgb_method1.astype(np.uint8))
    result_img_method1.save(os.path.join(output_dir, 'step6_final', 'blended_result_method1_feathering.jpg'), quality=95)
    
    # 기본 결과는 feathering 방법 사용
    result_full_rgb = result_full_rgb_method1
    
    print("\n" + "=" * 60)
    print("블렌딩 완료!")
    print(f"결과가 '{output_dir}' 폴더에 저장되었습니다.")
    print("=" * 60)
    
    return result_full_rgb


if __name__ == "__main__":
    # 이미지 경로
    hand_path = 'res/hand.jpg'
    eye_path = 'res/eye2.jpg'
    
    # ROI 좌표: (y_start, y_end, x_start, x_end)
    # hand에서 가로(224~393), 세로(318~416) 영역
    roi_coords = (318, 416, 224, 393)
    
    # 블렌딩 실행
    result = roi_eye_blend(
        hand_path=hand_path,
        eye_path=eye_path,
        roi_coords=roi_coords,
        output_dir='test3',
        levels=5,
        sigma=1.0,
        sigma_blur=3.0,
        ellipse_center=(0.5, 0.5),  # ROI 중심
        ellipse_size=(0.8, 0.8)      # ROI 크기의 80%
    )

