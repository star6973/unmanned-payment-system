# 논문 내용 정리(SSD:Single Shot Multibox Detector)
## 1. Abstract
- YOLO의 정확도의 한계를 극복하기 위해 출발.
- YOLO는 입력 이미지를 7x7 그리드로 나누고, 각 그리드별로 bounding box prediction을 진행하기 때문에 그리드 크기보다 작은 물체를 잡아내지 못하는 문제가 있다. 그리고 신경망을 모두 통과하면서 convolution과 pooling을 거쳐 coarse(조잡)한 정보만 남은 마지막 layer의 feature map만 사용하기 때문에 정확도가 하락하는 한계가 있다.

## 2. Introduction
- 당시 object detection의 SOTA에서는 다음과 같은 접근 방식이었다.
    + hypothesize bounding boxes
    + resample pixels or features for each box
    + apply a highquality classifier

- 이러한 방식은 정확하지만, 임베디드 시스템에서는 too computationally intensive하고, high end hardware에서도 real time application에 적용하기에는 너무 느리다.

- 이 논문에서는 bounding box 가설의 접근 방식만큼의 정확도를 가지면서, pixel과 feature를 resampling하지 않는 최초의 object detector 방식을 소개할 것이다. 모델의 결과, 높은 정확도를 가지며 속도가 크게 향상된다(VOC2007 테스트에서 R-CNN은 mAP 73.2% & 7FPS를, YOLO는 mAP 63.4% & 45FPS를, SSD는 mAP 74.3% & 59FPS를 기록했다).

- 기존에 비해 개선한 사항은 작은 convolution filter를 사용하여 object categories를 예측하고, different aspect ratio detections를 위해 bounding box의 위치의 offset을 별도의 예측 변수(필터)를 사용하였다. 그리고 이러한 필터를 여러 규모로 탐지하기 위한 네트워크의 후반 단계로부터 multiple feature maps에 적용하였다.

- 특히 각기 다른 scale의 prediction을 가지는 multiple layers를 사용하여 높은 정확도를 얻을 수 있었다.

## 3. The Single Shot Detector(SSD)
### 3.1. Model
- SSD 접근방식은 최종 탐지를 위해 non-maximum suppression을 따라 object class instances를 나타내며, 고정된 크기의 bouding boxes와 scores의 집합을 생산하는 feed-forward convolutional network이다.

- 초기 네트워크 layer는 높은 퀄리티의 이미지 분류기(base network라고 불리는)를 기반으로, auxiliary(보조) structure를 추가하여 detector를 만든다.

<center><img src="/reference_image/MH.Ji/SSD(Single Shot MultiBox Detector)/2.PNG" width="70%"></center><br>

- Multi Scale Feature Maps for Detection

    + SSD는 300x300 크기의 이미지를 입력받아서 ImageNet으로 pretrained된 VGG의 Conv5_3층까지 통과하며 feature를 추출한다.
    
    + 추출된 feature map을 convolution을 거쳐 다음 층에 넘겨주는 동시에 object detection을 수행한다. 이전 fully convolution network에서 convolution을 거치면서 디테일한 정보들이 사라지는 문제점을, 앞 layer의 feature map들을 가져와서 사용하는 방식으로 해결하였다. 즉, 여러 개의 feature map을 뽑으면서(보통 앞 layer로 갈수록 구체적인 공간정보를, 뒷 layer로 갈수록 추상적인 공간정보를 가지고 있음), 모든 정보를 활용하겠다는 것으로 생각할 수 있다.
        
    <center><img src="/reference_image/MH.Ji/SSD(Single Shot MultiBox Detector)/3.PNG" width="70%"></center><br>

    + VGG(based network)를 통과하여 얻은 feature map 대상으로 convolution을 진행하여 최종적으로 1x1 크기의 feature map을 추출한다. 그리고 각 단계별로 추출된 feature map은 detector & classifier를 통과시켜 object detection을 수행한다.

    <center><img src="/reference_image/MH.Ji/SSD(Single Shot MultiBox Detector)/4.PNG" width="70%"></center><br>

    + 하나의 그리드마다 크기가 각기 다른 default box들을 먼저 계산한다. defualt box란 Faster R-CNN에서 anchor 개념으로, 비율과 크기가 각기 다른 기본 박스를 먼저 설정해놓아 bounding box를 추론하는데 도움을 주는 장치이다.

        <center><img src="/reference_image/MH.Ji/SSD(Single Shot MultiBox Detector)/1.PNG" width="70%"></center><br>

        - 위 그림에서 고양이는 작은 물체이고, 강아지는 상대적으로 더 크다. 높은 resolution의 feature map에서는 작은 물체를 잘 잡아낼 수 있고, 낮은 resolution에서는 큰 물체를 잘 잡아낼 것이라고 추측할 수 있다.

        - SSD는 각각의 feature map을 가져와서 비율과 크기가 각기 다른 default box를 projection한다. 그리고 이렇게 찾아낸 박스들에 bounding box regression을 적용하고 confidence level을 계산한다. YOLO는 아무런 기본값 없이 2개의 box를 예측한 것과 대조적이다.

        - 즉, 하나의 그리드에서 생성하는 anchor box의 크기가 동일하다면, output feature map의 크기가 작은 4x4 filter에서는 큰 물체를 검출할 가능성이 높다. 반대로 output feature map의 크기가 큰 8x8 filter에서는 작은 물체를 검출할 가능성이 높다. 이러한 형태는 다양한 크기의 feature map을 detection하기 위해 사용한다면 더 좋은 성능을 기대할 수 있기 때문이다.

- Convolutional predictors for detection

    + 다음으로 feature map에 3x3 convolution을 적용(padding=1)하여, score for a category(classification)과 shape offset relative to the default box coordinates(bounding box regression) 값을 계산한다. 이는 각각의 default box들의 x, y, w, h의 조절값을 나타내므로 4차원 벡터에 해당한다. 위의 그림에서는 인덱스 하나에 3개의 defualt box를 적용하였기에, 결과 feature map의 크기가 5x5x12가 된다.

- Default boxes and aspect ratios

    + 마지막으로 각 feature map에서 bounding box를 몇 개씩 추출하는 것이 좋을까? 기본적으로 bounding box 1개를 뽑으면, 숫자 4개(x, y, w, h)가 나오고 추가로 classification을 위한 class 개수만큼의 정보가 추출된다. 

    + 각각의 default box마다 모든 클래스에 대하여 classification을 진행하는데, 총 20개의 클래스 + 1(배경) x default box의 수이므로 최종 feature map의 크기는 5x5x63이 된다. 이렇게 각 layer별로 feature map을 가져와 object detection을 수행한 결과들을 모두 합쳐서 loss를 구한 다음, 전체 네트워크를 학습시키는 방식으로 1 step end-to-end object detection 모델을 구성한다.

    <center><img src="/reference_image/MH.Ji/SSD(Single Shot MultiBox Detector)/5.PNG" width="70%"></center><br>

    + 위의 그림을 다시 예시로 들면, 하나의 feature map에 있는 여러 개의 location 중에서 하나의 location에서 k개의 bounding box를 뽑을 것이다. 이때, k는 4개 혹은 6개이다. 추출된 값은 총 3x3x(6x(classes + 4))가 되며, 이때 classes는 COCO DataSet으로 21개(20개 + 1개(배경 없음))이기 떄문에 150 채널 x (3x3)이 생긴다.

    + 이 150개의 채널은 위의 그림과 같이 [1번째 bounding box의 (x, y, h, w)의 4개 채널 + 1번째 bounding box의 class score에 해당하는 21개의 채널] ... [6번째 bounding box의 (x, y, h, w)의 4개 채널 + 6번째 bounding box의 class score에 해당하는 21개의 채널]로 구성된다.

### 3.2. Training
- Matching strategy

    + bounding box를 뽑아서 train을 할 때 정답인 bounding box와 추출한 bounding box와 비교를 해야 한다. 즉, classification의 정답이 있어야만 하는데, 이러한 판단 기준을 ground truth box를 가져와서 가장 많이 jaccard overlap되는 default box를 찾는다. 여기서 jaccard overlap은 IoU라고 하며, non-maximum-suppression 기법을 이용하여 threshold가 0.5 이상이면 선택하고, 미만이면 버린다.

- Training objective

    + Loss function은 Faster R-CNN과 거의 똑같다. 다음 식에서 N은 bounding box 중에 매치된 개수이고, 왼쪽 항은 classification loss, 오른쪽 항은 bounding box regression loss를 나타낸다. bounding box regression loss는 smooth L1 함수를 사용한다.

    <center><img src="/reference_image/MH.Ji/SSD(Single Shot MultiBox Detector)/6.PNG" width="70%"></center><br>

    <center><img src="/reference_image/MH.Ji/SSD(Single Shot MultiBox Detector)/7.PNG" width="70%"></center><br>

    + w와 h에 대한 함수만 log인 이유는, weight과 height의 크기가 바뀔 때마다 그 파장이 크기 때문이다.

    + SSD가 학습하는 것은 결국, 미리 정의된 bounding box를 얼마만큼 움직여야 ground truth와 비슷해질지를 결정하는 것이다. 하지만 이와 다르게, YOLO는 정해진 default box가 없고, 단순히 어디에 있는지를 예측하기 떄문에 정확도가 떨어진다.

- Choosing scales and aspect ratios for default boxes
    + object detection을 수행할 feature map의 개수를 m으로 놓고, feature map의 인덱스를 k로 둔다. m을 6으로 가정하면, 각 feature map별로 scale level의 수식은 다음과 같다.

    <center><img src="/reference_image/MH.Ji/SSD(Single Shot MultiBox Detector)/8.PNG" width="70%"></center><br>

    + s_min=0.2, s_max=0.9이고, min과 max를 잡은 다음 그 사이를 m값에 따라 적당히 구간을 나눠주는 값이다. m=6으로 설정했을 때의 결과는 [0.2, 0.34, 0.48, 0.62, 0.76, 0.9]가 된다.

    + 이 값은 각각의 feature map에서 default box의 크기를 계산할 때 입력 이미지의 width, height에 대해서 얼만큼 큰 지를 나타내는 값이다. 즉, 첫 번째 feature map에선 입력 이미지 크기의 0.2 비율을 가진 박스를 default box로 놓겠다는 것이고, 마지막 feature map에선 0.9와 같이 큰 default box를 잡겠다는 의미이다.

    + 이제 각각 default box의 width와 height의 값을 계산해야 한다. 이때 정사각형뿐만 아니라, 다양한 비율을 가진(1:2, 2:1, ...) default box를 구하고자 한다.

    <center><img src="/reference_image/MH.Ji/SSD(Single Shot MultiBox Detector)/9.PNG" width="70%"></center><br>

    + 구해진 비율값으로 입력 이미지에서 default box가 위치할 중심점을 구할 수 있다.

    <center><img src="/reference_image/MH.Ji/SSD(Single Shot MultiBox Detector)/10.PNG" width="70%"></center><br>

    + 구해진 중심점 좌표들에 원래의 입력 이미지 크기를 곱해서 중심점을 구하고, 각각의 중심점마다 default box를 그려주면 된다.

    <center><img src="/reference_image/MH.Ji/SSD(Single Shot MultiBox Detector)/11.PNG" width="70%"></center><br>

- Hard negative mining

    + 20개의 클래스만 분류하는 문제에서 8,000개나 되는 배경으로 bounding box가 생길 수도 있다. 이러한 Imbalanced data를 해결하기 위해 highest confidence loss를 가지는 default box를 sorting하여 순서대로 추출한다.

    + loss가 큰 순서란, back ground로 판단하지 않은 box들. 즉, 잘 못만춘 경우를 positive sample 대비 최대 3배 정도만 사용하도록 설정한다.

        - [Imbalanced data를 처리하는 7가지 기술](https://ourcstory.tistory.com/240)

<br><br>

## 4. Experiment Result
- VGG16을 base network로 사용하였으며, fc6와 fc7을 convolution layer로 사용하였다. 그리고 pool5를 stride=2인 2x2에서 stride=1인 3x3으로 atrous algorithm을 사용하여 바꾸고, fc8과 dropout을 사용하지 않았다.

    + [Review: DeepLabv1 & DeepLabv2 — Atrous Convolution (Semantic Segmentation)](https://towardsdatascience.com/review-deeplabv1-deeplabv2-atrous-convolution-semantic-segmentation-b51c5fbde92d)

    + atrous algorithm을 간단히 설명하자면, convolution이나 pooling을 이웃하게 적용하는 것이 아니라, 한 칸씩 건너뛰어서 하는 방법이다.

<center><img src="/reference_image/MH.Ji/SSD(Single Shot MultiBox Detector)/12.PNG" width="70%"></center><br>

<center><img src="/reference_image/MH.Ji/SSD(Single Shot MultiBox Detector)/13.PNG" width="70%"></center><br>

- SSD의 큰 문제점은 작은 물체들을 잘 못찾는다는 것이다. 작은 물체들은 보통 앞 layer에서 detection이 될텐데, feature map이 충분히 abstract level이 높지 않기 때문에 성능이 낮을 수 있다(RetinaNet은 이를 극복함). 이러한 문제를 data augmentation 기법으로 극복하고자 한다.

    <center><img src="/reference_image/MH.Ji/SSD(Single Shot MultiBox Detector)/14.PNG" width="70%"></center><br>

    <center><img src="/reference_image/MH.Ji/SSD(Single Shot MultiBox Detector)/15.PNG" width="70%"></center><br>

    + *표시된 부분이 data augmentation을 사용한 부분\

<br><br>

## 참고자료
- [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf)
- [PR-132: SSD: Single Shot MultiBox Detector](https://www.youtube.com/watch?v=ej1ISEoAK5g)
- [갈아먹는 Object Detection [6] SSD: SIngle Shot Multibox Detector](https://yeomko.tistory.com/20?category=888201)