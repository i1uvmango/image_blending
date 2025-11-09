"""
이미지 블렌딩 사용 예시
"""

from image_blending import blend_images

# 이미지 경로
image_A_path = 'res/eye2.jpg'
image_B_path = 'res/hand.jpg'

# 블렌딩 실행
result = blend_images(
    image_A_path=image_A_path,
    image_B_path=image_B_path,
    levels=5,
    sigma=1.0,
    kernel_size=None,
    blend_line=None,  # None이면 중간
    gradient_width=50
)

# 결과 저장
from PIL import Image
result_img = Image.fromarray(result)
result_img.save('blended_result.jpg')
print("블렌딩 완료! 결과가 'blended_result.jpg'에 저장되었습니다.")

