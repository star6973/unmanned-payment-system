# Joint Face Detection and Alignment using Multi-task Cascaded Convolution Networks
## 논문 리뷰
- Abstract
    - Face Detection and Alignment 는 1) 다양한 Poses 를 고려, 2) Illumination & Occlusion 문제에 대한 제약을 해결해야 하는 도전과제가 존재하는 분야
    - 최근 Deep learning 의 활용으로 Face Detection 과 Alignment 성능이 향상됨<br><br>
    - 본 논문에서는, "Deep cascaded multi-task"라는 Framework를 제안하고, 각각의 perfomance를 끌어올리기 위해서 내재된 correlation을 활용함
    - 특히, Framework는 3개의 신중하게 설계된 deep conv.network 와 cascaded structure를 적용 => "coarse-to-fine manner" 방식으로 face 예측 및 landmark 의 location 예측하도록 함<br><br> < Coarse-to-fine manner > <br> : 낮은 해상도에서 높은 해상도로 점진적으로 증가시켜가며 적용하는 방식. 이후 더 좋은 결과를 보인 image를 propagate 함<br><br><br>
1. Introduction
    - Face detection & alignment 는 face application 에 필수적인 필수적이지만, 광범위의 variation of face (ex 1) occlusions 2) 다양한 pose variation 3) extreme lighting) 로 인해서 굉장히 어려운 문제
    - 이에 "P.Viola et at. <Robust real-time face detection>" 논문에서는 cascaded classifier를 사용하여 real time에서 좋은 성능을 보임<br>그러나, 몇몇 논문에서 이러한 detector는 현실 속 사람 얼굴에서의 다양한 variation에 대해서는 성능이 낮다는 것을 보여줌
    - cascade structure 외에도 는 몇몇 논문에서 DPM(Deformable Part Model)을 소개하고, 좋은 성능을 보이는데 성공함
    - 그러나, DPM 방식도 단점을 갖고 있었는데, 그것은 바로 연산량이 많다는 점<br><br>
    - CNNs 등장 이후 많은 연구에서 CNN 기반으로 Face detection 접근을 시도
        - S. Yang et al. “From facial parts responses to face detection: A deep learning approach”
            - deep CNN 을 훈련시켜, face 예측지점에 window를 생성하는 방식으로, 얼굴에 대한 높은 응답을 얻으려고 시도
            - 그러나, CNN의 복잡성으로 인해서 이 과정에는 많은 시간이 소요됨
        - H. Li et al. “A convolutional neural network cascade for face detection”
            - cascaded CNN 방식을 적용하여 face detection
            - 그러나, 이 방식은 bounding box에 대한 calibration 이 필요하며, 이것은 많은 연산량을 요구하고 facial landmarks의 위치 간 내재된 correlation 를 무시함
    - 또한, Face alignment 분야에서도 Deep learning 을 활용한 연구들이 많이 이뤄짐
    - 그러나, Face Detection 과 Alignment 의 많은 방법들은 Detection 과 Alignment 사이의 내재된 correlation을 무시함
    - 따라서 이러한 문제를 해결하기 위한 여러 연구 역시 있음
        - Chen et al. “Joint cascade face detection and alignment,”
            - features 의 픽셀 값 차이를 사용한 Random forest 방식
        - zhang et al. “Improving multiview face detection with multi-task deep convolutional neural networks,”
            - multi-task CNN 을 사용하여 multi-view face detection의 정확도를 향상시키려고 함
            - 그러나, detection 정확도는 weak face deector로부터 생성된 초기 detection windows로 인해서 제한된 값을 가짐<br><br>
    - 한편, training 과정에서 어려운 sample을 사용하는 것은 detection을 강하게 할 수 있기 때문에 중요함
    - 그러나, 정통적인 방식의 sample mining 은 offline 방식으로 진행되는 경우가 일반적임
    - online 방식으로 어려운 sample mining 을 한다면, Face detection & alignment 에서 자동으로 training 과정에 대한 적응이 이뤄지기 때문에 더욱 바람직하다<br><br>
    - 본 논문에서, Face detectio 과 alignment 통합을 위해서 정의되지 않은 cascaded CNNs 가 multi-task learning 방식으로 사용된 Framework를 제안
        - 먼저, 얕은 CNN을 통해서 빠르게 후보 windows를 생산
        - 그 다음, 깊은 CNN을 통해서 얼굴이 아닌 곳에 대한 window를 생산하고, 첫 번째 단계에서 생성된 window를 제거
        - 최종적으로, 더욱 powerful CNN을 사용하여 결과 및 Face landmark 위치에 대한 정제 과정을 거침<br><br>
    - Contribution
        1. 새로운 cascaded CNNs 기반의 framework 를 제안하여 face detecion 및 alignment 수행
            - 또한 real-time 을 위한 light CNN 구조를 설계
        1.  성능 향상을 위해서 online 방식의 hard sample mining 제안
        1. 확장된 실험을 통해서 sota의 detection 과 alignment 방식과는 비교되는 방식으로 성능 향상을 이룸(?) <br><br><br>
1. Approach
1. Experiment
1. Conclusion