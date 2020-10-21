# 논문 내용 정리(DSFD: Dual Shot Face Detector)
## 1. Abstract
- 3가지 측면으로 공헌  
    1) better feature learning - FEM(Feature Enhance Module): single shot detector에서 dual shot detector로 확장할 때 feature map을 강화  
    2) progressive loss design - PAL(Progressive Anchor Loss): 2개의 다른 anchors 세트를 계산해서 feature를 효과적으로 촉진  
    3) anchor assign based data augmentation - IAM(Improved Anchor Matching): data augmentation 기법에 새로운 anchor 할당 전략을 통합시켜 regressor를 초기화하는데 기여  

<br><br>

## 2. Introduction
- Face detection은 alignment, parsing, recognition, verification과 같은 다양한 얼굴의 어플리케이션을 위한 기본 단계이다.
- CNN 이전에는 Viola-Jones의 수작업으로 만든 기능의 AdaBoost 알고리즘이 사용되었다.
- 하지만 아직 얼굴의 다양한 표정과 포즈, 폐색된 얼굴 등의 실제 시나리오는 여전히 도전과제에 놓여있다.
- 이전까지의 SOTA의 face detector는 크게 2가지 카테고리로 나뉜다.
    + two stage 방법에 기반되며, Faster RCNN에 적용되는 RPN(Region Proposal Network)
    + one stage 방법에 기반되며, RPN이 사라진 SSD(Single Shot Detector)

- 최근에는 one-stage face detection 프레임워크들이 더 높은 추론 효율과 직관적인 시스템 전개로 더욱 각광받고 있다.
- 위의 방법들이 눈부신 발전을 이뤘음에도 불구하고, 아직 3가지 측면의 문제가 남아있다.  
    1) Feature learning  
        + feature 추출은 face detector에서 필수불가결한 부분이다.
        + 최근에 FPN은 SOTA의 detector에서 가장 많이 사용되었다.
        + 하지만 FPN은 아직 높은 수준과 낮은 수준의 출력 레이어 간의 계층적 feature maps를 집계하기만 한다. 현재 레이어의 정보를 고려하지 않고 anchor간의 컨텍스트 관계는 무시된다.
        + 여기서 잠깐, anchor란?
            - 객체의 bounding box를 예측하기 위해서 여러개의 anchor box를 그려서 가장 많이 겹치는 것을 선택

    2) Loss design  
        + 기존의 loss 함수는 object detection의 regression loss(얼굴 영역에 대한)와 classification loss(얼굴이 감지되었는지에 대한)를 포함한다.
        + 추가로 클래스 불균형 문제를 해결하기 위해, 어려운 예제의 sparse set에 대한 훈련에 초점을 맞춘 Focal Loss를 제안했다.
        + 또, 모든 원본 및 향상된 기능의 특징들을 사용하기 위해, 효과적으로 학습하기 위한 Hierarchical Loss를 제안했다.
        + 그러나 위의 손실 함수들은 다른 level과 shot 둘 모두에 있는 feature map의 점진적인 학습 능력을 고려하지 않는다.

    3) Anchor matching  
        + 기본적으로 각 feature map에 대해 미리 설정된 anchor는 정기적으로 컬렉션을 이미지에 따라 다른 배율과 종횡비를 가진 상자로 타일링하여 생성된다.
        + 어떤 논문에서는 positive anchor를 증가시키고자 합당한 anchor 스케일과 anchor 보상 전략으로 분석한다.
        + 하지만 이러한 전략은 아직까지 positive와 negative anchor들 사이의 불균형을 일으키는 data augmentation의 랜덤 샘플링을 무시한다.

- 본 논문에서는 위와 같은 3가지 측면의 문제들을 3가지 novel 기술을 가지고 해결하고자 한다.  
    1) feature의 discriminability와 robustness를 강화하기 위해, FEM(Feature Enhance Module)[PyramidBox의 FPN과 RFBNet의 RFB의 장점을 합친]로 사용한다.  
    2) 서로 다른 level뿐만 아니라 서로 다른 shot들도 위한 점진적인 anchor 사이즈를 사용하기 위해, PyramidBox의 계층적 loss와 pyramid anchor에서 모티브한 PAL(Progressive Anchor Loss)를 디자인했다. 구체적으로, first shot에서는 작은 anchor 사이즈를 할당했고, second shot에서는 큰 anchor 사이즈를 할당했다.  
    3) anchor와 ground truth face를 더 잘 매치시키기 위해서와 regressor에 더 나은 초기화를 제공해주기 위해서, anchor 파티션 전략과 anchor 기반 data augmentation을 통합한 IAM(Improved Anchor Matching)을 제안한다.  

- 위의 세 가지 기능들은 서로 상호보완적으로 작동하면서 더 나은 성능을 이끈다.

<br><br>

## 3. Related Works
- Feature Learning
    + 초기의 face detection은 control point set, edge orientation histogram 등과 같이 수작업으로 특징을 추출하였다. 
    + CNN이 등장한 이후, Overfeat, Cascade-CNN, MTCNN, adopt CNN은 feature pyramid를 구축하기 위해 image pyramid의 sliding window detector를 만든다. 하지만 image pyramid를 사용하는 것은 실시간으로 느리고 메모리가 비효율적이다. 결과적으로 대부분의 two-stage detector는 single scale에서 feature를 추출하는 방식이다.
    + R-CNN은 selective search를 통해 예측 영역을 획득한 다음, 각 정규화된 이미지 영역을 CNN을 통해 분류한다. Faster R-CNN, R-FCN은 RPN을 사용하여 초기화된 예측 영역을 생성한다. 게다가 RoI-pooling과 position-sensitive RoI pooling은 각각의 영역으로부터 feature를 추출할 때 적용한다.
    + 더욱 최근에는, 어떤 연구들은 작은 물체들을 더 잘 탐지하기 위해 multi-scale 방식을 제시한다. 구체적으로, SSD, MS-CNN, SSH, S3FD는 feature 계층의 multiple layer에 있는 박스를 예측한다
    + FCN, Hypercolumns, Parsenet은 segmentation에서 multiple layer features를 융합한다.
    + FPN은 top-down 구조로, high-level의 모든 스케일마다의 의미있는 정보들을 통합한다. FPN을 기반으로 하는 방법들, FAN, PyramidBox는 detection의 중요한 향상을 성취했다. 하지만 이러한 방법들은 현재 layer의 정보들을 고려하지 않는다.
    + anchors 사이에 context 관계를 무시한 위의 제시된 기법들과 다르게, 본 논문에서는 feature의 sementic을 강화하기 위한 multi-level의 확장된 convolutional layer를 통합시킨 FEM을 제안한다.

- Loss Design
    + 일반적으로 objective loss는 classification loss와 box regression loss의 합이다.
    + Girshick는 smooth L1 loss를 제안하여 gradient가 튀는 것을 방지했다.
    + Lin 외 다수 연구진들은 one stage detector에 더 나은 향상을 위해 하나의 장애물이 클래스 불균형이라는 것을 발견했다. 그러므로 그들은 동적으로 스케일된 cross entropy loss인 focal loss를 제안했다.
    + FANet은 계층적인 feature pyramid를 만들어 계층적 손실을 방지했다. 하지만 FANet에서 사용되는 anchor는 서로 다른 stage에서 같은 사이즈를 유지해야만 한다.
    + 본 논문에서는 feature가 촉진할 수 있도록 서로 다른 stage에서 서로 다른 anchor 사이즈를 적절하게 고르는 방법을 제시한다.

- Anchor Matching
    + 모델을 더욱 robust(이상값에 통계의 변동폭이 적은, 강건한)하게 만들기 위해 대다수 detection 모델은 data augmentation을 사용한다.
    + Zhang 외 다수 연구진들은 학습할 동안 anchor들로 작은 얼굴을 충분히 맞추기 위해서 anchor 보상 전략을 제안한다.
    + Wang 외 다수 연구진들은 학습할 동안 다수의 폐색된 얼굴의 수를 만들어 랜덤 crop하는 방법을 제안한다.
    + 하지만 이러한 방법들은 random sampling을 무시하는 반면, 본 논문에서는 anchor 매칭을 위한 더 나은 데이터 초기화를 제공하기 위해 anchor 할당을 합친다.

<br><br>

## 4. Dual Shot Face Detector
### 4.1. Pipeline of DSFD

<center><img src="/reference_image/MH.Ji/DSFD/1.PNG" width="60%"></center><br>

- 위의 그림에서 of1, of2, of3, of4, of5, of6를 ef1, ef2, ef3, ef4, ef5, ef6로 변경한다. 이는 원래 크기와 동일한 크기이며, second shot detection layer를 구성하기 위해 SSD 스타일의 head로 공급된다.
- S3FD, PyramidBox와 달리 FEM에서 receptive field 확대를 하고 난 후, 동일 비율 간격의 원칙을 만족하는 stride, anchor 그리고 receptive field의 세 가지 사이즈가 불필요한 새로운 anchor 디자인 전략을 활용한다.
- 그러므로 DSFD는 더욱 유연하고 강건하다. 게다가, 원본과 강화된 shot들은 2개의 서로 다른 loss를 가지며, 각각 First Shot progressive anchor Loss(FSL)과 Second Shot progressive anchor Loss(SSL)이라고 명친한다.

<br>

### 4.2. Feature Enhance Moduel
- FEM은 original feature를 더욱 차별화되고 강건하게 만들어줄 수 있다. 이를 위해 서로 다른 차원의 정보(original neuron cell의 상위 layer와 non-local neuron cell의 현재 layer의 정보)를 포함한다.
- 수학적인 공식은 다음과 같다.

    <center><img src="/reference_image/MH.Ji/DSFD/3.PNG" width="60%"></center><br>

<center><img src="/reference_image/MH.Ji/DSFD/2.PNG" width="60%"></center><br>

- 위의 그림은 FEM을 그린 것으로, FPN과 RFB에서 모티브된 것이다.
- 처음 1x1 convolutional kernel로 feature map을 정규화시킨다. 그리고나서 상위 feature map을 up-sample하여 현재 요소로 element-wise 제품을 수행한다. 마지막으로, 서로 다른 확장 convolutional layer를 포함한 3개의 하위 네트워크에 따라 feature map을 3가지로 나눈다.

<br>

### 4.3. 



## 5. Experiments