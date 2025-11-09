Step 1. 이미지 로드, YUV 변환, Y 채널 분리

두 이미지를 RGB로 로드, numpy로 YUV 변환, Y 채널만 추출하는 함수를 작성하라.

Step 2. Custom Gaussian Filtering & Downsampling

시그마와 커널크기를 파라미터로 받아 직접 Gaussian 커널(2D) 생성, 컨볼루션 후 (stride=2)로 downsampling하는 함수를 구현하라.

Step 3. Gaussian Pyramid 생성

한 장의 Y채널 이미지를 입력받아, 지정한 level 수만큼 (이전 함수 재사용) 리스트 형태 Gaussian pyramid 생성코드(재귀가 아닌 반복문 이용) 작성하라.

Step 4. Laplacian Pyramid 생성

Gaussian pyramid를 입력받아, 각 레벨마다 낮은 해상도(up-sample) 이미지를 올리고(보간), 해당 레벨의 Gaussian에서 다운업된 이미지를 뺀 Laplacian pyramid 계산함수를 작성하라.

Step 5. Mask 정의 및 Pyramid-level Blending

이미지 shape와 결합구역(예: 중간 선) 기준으로 0~1의 부드러운 그라데이션 mask를 만든다.

두 이미지 Laplacian pyramid, mask(혹은 mask pyramid)를 받아 단계별로 blending하는 함수를 구현하라.

Step 6. Pyramid Collapse (Blended Image 복원)

Blended Laplacian pyramid를 받아 upsample+add 방식으로 최종 blended Y채널 이미지를 복원하는 코드를 작성하라.

Step 7. YUV + UV 재결합, 최종 RGB 복원

blended Y, 원래의 U/V 채널을 결합, numpy로 YUV → RGB 복원함수 작성.

