"""
eye2와 hand 이미지 블렌딩 실행 스크립트
"""

from image_blending import blend_images_with_steps
import os

# 현재 스크립트의 디렉토리 기준으로 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 이미지 경로
image_A_path = os.path.join('res', 'eye2.jpg')
image_B_path = os.path.join('res', 'hand.jpg')
output_dir = 'test'

print("=" * 60)
print("Laplacian Pyramid Image Blending 실행")
print("=" * 60)
print(f"이미지 A: {image_A_path}")
print(f"이미지 B: {image_B_path}")
print(f"결과 저장 위치: {output_dir}")
print("=" * 60)
print()

# Step별 결과를 저장하며 블렌딩 실행
results = blend_images_with_steps(
    image_A_path=image_A_path,
    image_B_path=image_B_path,
    output_dir=output_dir,
    levels=5,
    sigma=1.0,
    kernel_size=None,
    blend_line=None,  # 중간선
    gradient_width=50,
    save_intermediate=True
)

print()
print("=" * 60)
print("블렌딩 완료!")
print(f"모든 결과는 '{output_dir}' 폴더에 저장되었습니다.")
print("=" * 60)

