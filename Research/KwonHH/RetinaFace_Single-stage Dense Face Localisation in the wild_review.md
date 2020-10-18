# RetinaFace : Single-stage Dense Face Localisation in the wild
## 논문 리뷰
- Abstract
    - Retina Face 특징
         - 강직한 single-stage 기법의 face detector
         - pixel-wise 연산을 통해서 localization 수행
         - 다양한 크기의 face에 대해서 joint extra-supervised & self-supervised multi task 학습을 통해 이점을 얻음<br>
    - Retina Face contribution
        - 5개의 landmark를 WIDER FACE dataset에 찍고, extra supervision을 통해서 hard face detection의 성능 향상
        - pixel-wise 3D face shape를 예측하기 위해서 supervised mesh decoder branch 를 추가
        - WIDER FACE dataset의 hard test에 대해서 AP를 향상 시킴
        - ArcFAce를 적용하여 더욱 성능을 향상시킴
        - weight가 작은 backbone을 사용하여 CPU에서 구현할 수 있는 모델을 설계
1. Introduction
    - 자동으로 얼굴의 위치를 찾는 것은 face 이미지 분석을 위한 여러 application 에서 필수 과정
    - 좁은 의미의 face localization(=전통적인 face localization) : face bounding box를 scale 과 position을 고려하지 않고 계산함
    - 넓은 의미의 face localization : face detection, face alignment, pixel-wise face parsing, 3D dense correspondence regression 포함
        - 이러한 깊은 face localization은 서로 다른 크기의 face에 대해서도 정확한 position을 찾을 수 있음
    - 기존의 Object Detection에 비해 face detection은 더 작은 변화율(약 1:1 ~ 1:1.5)를 갖지만, 훨씬 큰 scale 변화율을 가짐(거의 수 pixels ~ 몇 천 pixel)
    - 가장 최근 sota 방식은 "깊은 face localization", "feature 피라미드의 scale", "약속된 동작에 대한 설명", "two-stage 방식에 비해 빠른 속도"의 특징을 갖는 single-stage 방식에 더욱 집중되어있음
    - 이러한 경향에 따라서 본 논문도 single-stage 방식을 진화시켰으며, supervised 및 self-supervised 에 의한 multi-task loss를 구하고 이를 토대로 dense face localization하는 방식을 적용
    - 