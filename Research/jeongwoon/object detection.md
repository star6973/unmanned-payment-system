## Object detection이란?

### - Image Classification task에 사물의 위치를 Bounding Box로 예측하는 Regression task가 추가된 문제dlek.
### - "Object Detection = Multi-labeled Classification + Bounding Box Regression"
### - Object detection 방법론 종류

#### 1) sliding window 
#### - 다양한 scale의 window를 이미지의 왼쪽 위부터 오른쪽 아래까지 sliding하며 score를 계산하는 방법
#### - 하나의 이미지에서 여러 번 score를 계산하여야 하므로 속도 측면에서 비효율적.
#### - 문제를 해결하기 위해 Deformable Part Model(DPM)이 쓰임.
##### * DPM : 정적 feature 추출, region 분류, 고득점 region에 대한 bbox 예측 등을 하기 위해 분리된 파이프라인을 사용하는 기법

#### 2) selective search
#### - 영상의 계층적 구조를 활용하여 영역을 탐색하고 그룹화하는 과정을 반복
#### - 객체의 위치를 proposal 해주는 기능을 수행

## First Object detection (2)
### < OverFeat >
#### - 2013년 열린 ImageNet 대회인 ILSVRC2013에서 Object Localization 부문에서 1위를 차지하였고, Object Detection 부문에서는 3위를 차지한 방법론
##### * Object Localization은 Object Detection에 비해 쉬운 task이며 이미지 당 하나의 object에 대해 bounding box를 얻는 문제를 의미.
#### -  OverFeat은 arXiv 기준 2013년 12월 21일에 처음 업로드 되었으며, ImageNet 대회에서 Classification이 아닌 다른 분야에 처음으로 CNN을 적용한 방법론
#### (1) input image가 있으면 Selective Search를 통해 Region Proposal을 수행
#### (2) 그 뒤, Proposal된 영역들을 CNN의 고정 사이즈를 가지는 입력으로 변환
#### (3) 각 영역마다 Classification을 수행하는 방식