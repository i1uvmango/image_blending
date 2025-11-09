Raw-Level Image Blending(Without OpenCV)
단계별 설계 계획
이미지(YUV 변환 및 Y(밝기) 채널 분리)

입력 이미지를 RGB → YUV 컬러공간으로 변환, 밝기(Y)만 가져오기

색 왜곡 방지를 위해 전체 블렌딩 연산은 Y 채널에서만 진행

Gaussian Pyramid 생성 (직접 커널로 구현)

지정한 시그마, 커널 크기로 Gaussian 블러를 적용한 후 downsampling(2배, 혹은 stride 지정하여)

각 단에서 pillow/ numpy 등으로 해당 연산의 미세 구현

Laplacian Pyramid 생성

각 레벨을 upsample(직접 구현, interpolation), 바로 윗 레벨의 가우시안에서 빼서 세부 정보 추출

Laplacian Pyramid는 디테일(에지‧텍스처) 정보

Blending(마스크 이용, 각 피라미드끼리 합성)

두 이미지를 합치는 mask(예: 손과 눈의 경계선)을 직접 정의(0~1),

Blend: 각 레벨의 Laplacian을 mask에 따라 선형 조합

각 레벨별로:
L
S
(
i
,
j
)
=
G
m
a
s
k
(
i
,
j
)
⋅
L
A
(
i
,
j
)
+
(
1
−
G
m
a
s
k
(
i
,
j
)
)
⋅
L
B
(
i
,
j
)
LS(i,j)=G 
mask
 (i,j)⋅LA(i,j)+(1−G 
mask
 (i,j))⋅LB(i,j)

피라미드 Collapse 및 Reconstruction

Master blended Laplacian pyramid를 상위 → 하위로 upsampling + 더하기 반복, 원래 이미지 크기로 복원

Upsample: 0 사이 넣고, Gaussian blur 등으로 매끄럽게

YUV → RGB 재조합 및 결과 출력

완성된 Y 채널과 원래의 U, V를 합쳐 RGB로 최종 변환

(추가분석) 커널 변화 실험/stride변화 예시

한 레벨에 대해 kernel size/sigma 등 파라미터 바꿔 결과 비교/분석

부가적 인사이트·주의점
Edges(에지) 혹은 관심영역이 뚜렷한 경우, 해당 부분 pyramid level 더 깊게 길게 설정(adaptive)

Laplacian 추출시 upsample 연산/필터링(jaggy/noise 방지) 꼼꼼히 구현필요

Mask가 부드럽게(Feathering) gradient 형태일 때 경계 blending이 훨씬 자연스러움

Color channel(Y만)로 분리하는 이유와, RGB에서 blending할 때 발생할 수 있는 색왜곡 사례 시각적으로 비교

커널 사이즈, sigma, stride 등 blending 품질에 미치는 영향 실험 및 종합분석 권장