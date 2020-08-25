# Rich feature hierarchies for accurate object detection and semantic segmentation Tech Report
## Tech Report Review
- Abstract
    - object detection 은 지난 몇년간 정체기
    - 주로 low level과 high level의 앙상블을 하는것이 가장 좋은 효과를 보였지만, 다소 복잡한 방식
    - 따라서 본 paper에서는 간단하고 scalable 가능한 detection 알고리즘을 제안하여 이전 VOC2012의 최우수 모델보다 mAP를 30% 이상 끌어올림<br>그 방법으로는
        1. 이미지를 분리하고, localize 하기 위해서 아래에서부터 윗방향으로 region proposal 에 대해서 high capacity CNN 적용
        1. 만약 라벨 된 훈련 데이터가 부족할 경우 중요한 performance boost를 하도록 함
    - Proposal region 에 CNN을 적용했기 때문에 R-CNN이라고 함
    