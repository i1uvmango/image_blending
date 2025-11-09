"""
eye2와 hand 이미지 블렌딩 실행
"""

import sys
import os

# 현재 파일의 디렉토리를 작업 디렉토리로 설정
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__이 없는 경우 (exec로 실행된 경우)
    import glob
    workspace = r'C:\Users\xcrui\OneDrive'
    files = glob.glob(os.path.join(workspace, '**', 'execute_blending.py'), recursive=True)
    if files:
        current_dir = os.path.dirname(os.path.abspath(files[0]))
    else:
        current_dir = os.getcwd()
os.chdir(current_dir)
sys.path.insert(0, current_dir)

from image_blending import blend_images
from PIL import Image

# 이미지 경로
image_A_path = os.path.join('res', 'eye2.jpg')
image_B_path = os.path.join('res', 'hand2.jpg')
output_dir = 'test'

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("Laplacian Pyramid Image Blending 실행")
print("=" * 60)
print(f"작업 디렉토리: {os.getcwd()}")
print(f"이미지 A: {image_A_path} (존재: {os.path.exists(image_A_path)})")
print(f"이미지 B: {image_B_path} (존재: {os.path.exists(image_B_path)})")
print(f"결과 저장 위치: {output_dir}")
print("=" * 60)
print()

# 블렌딩 실행
try:
    result = blend_images(
        image_A_path=image_A_path,
        image_B_path=image_B_path,
        levels=5,
        sigma=1.0,
        kernel_size=None,
        blend_line=None,  # None이면 중간
        gradient_width=50,
        output_dir=output_dir,
        save_intermediate=True  # 중간 결과 저장
    )
    
    # 결과 저장
    result_img = Image.fromarray(result)
    result_path = os.path.join(output_dir, 'blended_result.jpg')
    result_img.save(result_path)
    
    print()
    print("=" * 60)
    print("블렌딩 완료!")
    print(f"결과가 '{result_path}'에 저장되었습니다.")
    print("=" * 60)
except Exception as e:
    print(f"오류 발생: {e}")
    import traceback
    traceback.print_exc()

