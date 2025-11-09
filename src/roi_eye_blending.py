"""
ROI 영역에 eye 이미지를 Laplacian Pyramid 방식으로 합성
- eye의 눈 윤곽만 blending 대상
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
    """Laplacian Pyramid 생성"""
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
    
    # 마지막 레벨은 Gaussian의 마지막 레벨
    laplacian_pyramid.append(gaussian_pyramid[-1].copy())
    
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
# Step 5: Edge Mask 생성 (eye의 Laplacian 절댓값 기반)
# ============================================================================

def create_edge_mask_from_laplacian(laplacian_pyramid, sigma_blur=3.0):
    """
    eye의 Laplacian 절댓값으로 edge mask 생성
    
    Args:
        laplacian_pyramid: eye의 Laplacian pyramid
        sigma_blur: Gaussian blur 표준편차
    
    Returns:
        numpy array: 0~1 범위의 edge mask
    """
    # 각 레벨의 Laplacian 절댓값을 합산
    edge_map = np.zeros_like(laplacian_pyramid[0])
    
    for i, lap in enumerate(laplacian_pyramid[:-1]):  # 마지막 레벨 제외
        # 절댓값
        abs_lap = np.abs(lap)
        # 원본 크기로 업샘플링
        if i > 0:
            abs_lap = upsample(abs_lap, edge_map.shape)
        edge_map += abs_lap
    
    # 정규화
    edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-10)
    
    # Gaussian blur로 부드럽게
    edge_mask = gaussian_filter(edge_map, sigma=sigma_blur)
    
    # 0~1 범위로 정규화
    edge_mask = (edge_mask - edge_mask.min()) / (edge_mask.max() - edge_mask.min() + 1e-10)
    
    return edge_mask


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

def collapse_laplacian_pyramid(laplacian_pyramid, sigma=1.0, kernel_size=None):
    """Laplacian Pyramid를 복원하여 원본 이미지 재구성"""
    # 마지막 레벨부터 시작
    reconstructed = laplacian_pyramid[-1].copy()
    
    # 역순으로 업샘플링하고 더하기
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        # 목표 크기
        target_shape = laplacian_pyramid[i].shape
        
        # 업샘플링
        upsampled = upsample(reconstructed, target_shape)
        
        # Gaussian 필터 적용
        upsampled_filtered = gaussian_filter_2d(upsampled, sigma, kernel_size)
        
        # Laplacian 더하기
        reconstructed = upsampled_filtered + laplacian_pyramid[i]
    
    return reconstructed


# ============================================================================
# 메인 블렌딩 파이프라인
# ============================================================================

def roi_eye_blend(hand_path, eye_path, roi_coords, output_dir='test2', 
                  levels=5, sigma=1.0, sigma_blur=3.0):
    """
    hand 이미지의 ROI 영역에 eye 이미지를 Laplacian Pyramid로 블렌딩
    
    Args:
        hand_path: hand 이미지 경로
        eye_path: eye 이미지 경로
        roi_coords: (y_start, y_end, x_start, x_end) ROI 좌표
        output_dir: 결과 저장 디렉토리
        levels: 피라미드 레벨 수
        sigma: Gaussian 표준편차
        sigma_blur: Edge mask blur 표준편차
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'step1_preparation'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'step2_gaussian_pyramid_hand'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'step2_gaussian_pyramid_eye'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'step3_laplacian_pyramid_hand'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'step3_laplacian_pyramid_eye'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'step4_edge_mask'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'step5_blended_laplacian'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'step6_final'), exist_ok=True)
    
    print("=" * 60)
    print("ROI 기반 Eye Blending 시작")
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
    
    # Step 4: Edge Mask 생성 (eye의 Laplacian 절댓값 기반)
    print("\n[Step 4] Edge Mask 생성 (eye Laplacian 절댓값 기반)")
    edge_mask = create_edge_mask_from_laplacian(laplacian_eye_Y, sigma_blur)
    mask_pyramid = create_mask_pyramid(edge_mask, levels, sigma)
    
    visualize_mask(edge_mask, "Edge Mask (from Eye Laplacian)", 
                   os.path.join(output_dir, 'step4_edge_mask', 'edge_mask.png'))
    
    # Mask pyramid 시각화
    visualize_gaussian_pyramid(mask_pyramid, "Edge Mask Pyramid",
                              os.path.join(output_dir, 'step4_edge_mask', 'mask_pyramid.png'),
                              is_mask=True)
    
    # 각 레벨 저장
    for i, level_mask in enumerate(mask_pyramid):
        mask_vis = (level_mask * 255).astype(np.uint8)
        Image.fromarray(mask_vis).save(
            os.path.join(output_dir, 'step4_edge_mask', f'mask_level_{i}.png')
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
    Y_blended_roi = collapse_laplacian_pyramid(blended_laplacian_Y, sigma)
    U_blended_roi = collapse_laplacian_pyramid(blended_laplacian_U, sigma)
    V_blended_roi = collapse_laplacian_pyramid(blended_laplacian_V, sigma)
    
    # YUV → RGB 변환
    result_roi_rgb = yuv_to_rgb(Y_blended_roi, U_blended_roi, V_blended_roi)
    
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
        output_dir='test2',
        levels=5,
        sigma=1.0,
        sigma_blur=3.0
    )

