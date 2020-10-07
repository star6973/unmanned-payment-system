# KIRO & 롯데정보통신
## 물품 인식 및 얼굴 인식을 통한 무인 편의점 결제 시스템

### 팀원
1. 김정운
2. 권혁화
3. 지명화
4. 홍인화

### 기술 스택
1. Image Classification
2. Object Detection
3. Face Detection
4. Model Compression
5. Edge Computing

### 주차별 학습
| 주차 | 기간 | 주제 | 수행 과정 및 결과 |
|:-----:|:-----:|:-----:|:-----:|
| 1주 | 2020-08-03 ~ <br> 2020-08-07 </br> | 딥러닝 환경 구성 | 개인 노트북에 GPU가 없기에 윈도우 환경에서 MobaXterm을 통해 KIRO 8 GPU 서버 연결 |
| 2주 | 2020-08-10 ~ <br> 2020-08-14 </br> | 데이터셋 구축 1차 | 데이터셋 구축 장비 준비가 늦춰지면서, 3주차와 순서를 바꿈. Image Classification 조사(LeNet, AlexNet, VGGNet, GoogLeNet, ResNet) |
| 3주 | 2020-08-18 ~ <br> 2020-08-21 </br> | Image Classification 기술 조사 | Image Classification 기술 조사(PreActResNet ~ NASNet), 데이터셋 구축 중..(08/19 - 1,000장 완료, 8/20 - 2,000장 완료, 8/21 - Labeling 시작) |
| 4주 | 2020-08-24 ~ <br> 2020-08-28 </br> | 데이터셋 구축 2차 (8/24 - Labeling 끝) | Object Detection 기술 조사(R-CNN ~ YOLO) |
| 5주 | 2020-08-31 ~ <br> 2020-09-04 </br> | Object Detection 기술 조사 | 데이터셋 cropping(9/3 - train 9924장, val 2505장, test 5327장), 데이터셋으로 Image Classification 구현(9/4 - customCNN 구현, 9/5 - ResNet 구현, 9/6 - VGGNet 구현, 9/7 - DenseNet 구현), Object Detection 기술 조사(R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN) |
| 6주 | 2020-09-07 ~ <br> 2020-09-11 </br> | Face Detection 기술 조사 | 9/8 - Object Detection 발표(RODEO(혁화), IterDet(정운), SSD(명화), EfficientDet(인화)), 9/9 - Image Classification 테스팅 98%(ResNet) 완료 |
| 7주 | 2020-09-14 ~ <br> 2020-09-18 </br> | 기존 모델 다루기 실습 | 9/10-9/13 - Image Classification 테스팅 (VGGNet, ResNet, MobileNet, EfficientNet) 완료, 9/14 - Object Detection Tutorial, 9/17-20 논문 찾기(DetectorRS, NETNet, CenterNet) |
| 8주 | 2020-09-21 ~ <br> 2020-09-25 </br> | Transfer Learning 준비 | 9/21-9/23 - Object Detection Model Train, 9/24 - mmdetection(faster rcnn - mAP 98.20%, ssd - 97.36%, cascade rcnn - mAP 98.16%) 학습 완료(mAP=97) |
| 9주 | 2020-09-28 ~ <br> 2020-09-29 </br> | Transfer Learning 실습 | 9/28-9/29 - mmdetection(dynamic rcnn - 97.99%, detectors - 97.90%) 학습 완료  |
| 10주 | 2020-10-05 ~ <br> 2020-10-08 </br> | 결과 검토 및 모델 수정 | 10/5-10/6 - Object Detection 추론 및 테스팅 완료 |
| 11주 | 2020-10-12 ~ <br> 2020-10-16 </br> | GUI 개발 |  |
| 12주 | 2020-10-19 ~ <br> 2020-10-23 </br> | 기술별 보완 |  |
| 13주 | 2020-10-26 ~ <br> 2020-10-30 </br> | 추론기 제작 |  |
| 14주 | 2020-11-02 ~ <br> 2020-11-06 </br> | 동작 시험 및 디버깅 |  |
| 15주 | 2020-11-09 ~ <br> 2020-11-13 </br> | 발표 준비 및 최종 발표 |  |
<br><br>

## 피드백
### 2020-08-11
1. Image classification에 관한 추천 논문 자료
2. google drive에 공유자료 업로드

### 2020-08-18
1. ResNet

    F(x): 출력값과 실제값의 차이 -> 줄이자
    F(x)+x: 참값과 편차값을 더하여 출력값에 가까이 만들자
    
    ResNet은 vanishing gradient 문제를 해결하기 위해 등장한 것이 아니라, degradation을 해결하기 위해 등장

2. 1x1 convolution 연산

    1x1 convolution을 하는 이유는 단순 연산량을 줄이기 위해, feature를 추출하는 것은 동일하지만 3x3 filter를 사용할 때보다는 적을 수 밖에 없다. 
    하지만 정보량의 손실이 달라지지는 않는다.

3. bottleneck 구조

    identity shortcut은 그대로 차원을 사용

4. 데이콘 MNIST 경진대회에 네이버의 cutmix 사용해보기
5. Image Classification에서는 EfficientNet을 반드시 공부해 볼 것.

### 2020-08-19
1. Inception에서 stem layer와 reduction block의 개념
    - stem layer는 7x7 Conv로 이후의 모델들에서는 사라진 필터이다. 현재 최신 모델들은 3x3 Conv filter를 지향하고 있으며, 이를 넘어가는 필터를 사용하는 자들은 ㅅ ㅏ ㄹ ㅏ ㅁ ㅇ ㅣ ㅇ ㅏ ㄴ ㅣ ㄷ ㅏ..

    - reduction block은 grid 크기(filter의 크기)의 감소를 목적으로, 크기를 감소하면서 feature도 추출한다.

2. SqueezeNet에서 fire module은 squeeze layer + expand layer로 구성되어 있는데, 이 expand layer는 1x1 Conv와 3x3 Conv로 이루어져있다. 이때의 역할은?
    - 하이퍼파라미터의 값을 주어 input channel과 같은 크기의 채널로 맞추도록 팽창하는 역할
    
    - 예를 들어, input으로 128개의 채널이 들어오면, squeeze layer의 1x1 Conv를 통해 16채널로 줄였다가 expand layer의 1x1 Conv를 통해 64채널과, 3x3 Conv를 통해 64채널로 팽창시키고 이 둘을 합쳐 input 채널과 동일한 128 채널로 만들어준다.

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/172.png" width="70%"></center><br>

3. ShuffleNet의 구조
    - 연산량 감소를 위한 목적으로 생성한 모델

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/173.png" width="70%"></center><br>

4. docker gpu 사용

        local host: 172.16.201.167
        id: nvidia
        pw : nvidia

        docker run --gpus all -it --rm -v ~/MH.Ji:/MH.Ji nvcr.io/nvidia/pytorch:20.07-py3

### 2020-08-25
1. Data set을 합쳐서, 그 분포도를 확인해보기
    - 한쪽으로 치우쳐지면, 그 부분에만 학습이 많이 되서 overfitting이 될 가능성이 높음

2. Classification -> Localization -> Detection -> Segmentation의 역사
    - 처음에는 물체를 분류하기 시작했다가, 어디에 위치하는지를 보고 싶어서 Localization의 등장
    - 그 이후에는 어떤 물체인지를 판단하고 싶어서 detection의 등장
    - polygon 형태의 물체를 인식하는 방법인 segemntation이 등장하고, 개별 물체 탐지가 가능한 intance segmentation이 등장

3. Data set을 detection model에 학습시키기 위해선, 다양한 포맷이 있음
    - COCO, Pascal VOC, xml, json 등..

4. Object Detection 논문 공부하기(만들어놓은 데이터를 학습시켜보기 위해)
    - R-CNN을 기초로..
    - [deep learning object detection](https://github.com/hoya012/deep_learning_object_detection)

### 2020-09-02
1. Object Detection을 수행하기 전에, Image Classification이 back bone이 되어야 한다.
    - 대표적으로 YOLO는 DarkNet을 back bone으로 사용

2. 현재 만들어놓은 2,000장의 사진을 Image Classification하기 위해서, xmin, ymin, xmax, ymax 영역으로 잘라서 만든다.

### 2020-09-07
1. gpu 서버로 파일 전송 방법

        local server: 172.16.201.167
        
        scp vggnet.py nvidia@172.16.201.167:~/MH.Ji

### 2020-09-09
1. Image Classification 모델 및 코드 첨삭
    - gray scale에 대해 회의적
    - horiental flip -> 거울상이 생기는 것이라서 별로 비추...
    - augmentation 및 randomsampler를 보통 한다.(Nomalization 하지 않고.. -> 한 번 해보기)
    - 항상 함수를 사용하기 전에 pytorch docs를 한번 보고 눈으로 확인해야한다.
    - https://hoya012.github.io/blog/albumentation_tutorial/ 여기 참고해서 로테이션에 관한것은 가져다 써보기
    - 다른 optimizer 썼을 때는 튀는 경우가 많은데 SGD 쓰면 괜찮다.
    - scheduler는 learning rate를 계속 줄여서 미니멈한 것을 찾아내는 것이다. 
    - early stopping 의 patience가 scheduler보다 더 길어야한다. -> ealry stopping은 어느 정도 찾았을 때 진행이 더디면 끝내는 것.
    - validation을 참고해서 test data load를 만들어서 확인하는게 좋을거 같다. 
    - [성능 문제] -> 테스트 결과 5,000개 중 400개밖에 못 맞춤. -> Overfitting 문제는 아닌 것 같음. -> image_list가 어떤건지 뽑아봐서 확인할 필요가 있음 -> 확인 완료.. 0, 1, 10, 11, ... 클래스가 이런 형태였음

2. Pytorch 튜토리얼을 라인 바이 라인으로 분석해보자.

3. 과제
    - 다시 Image Classification 학습 해보기(최소 90% 이상은 나와야 함)
    - Object Detection 조사 및 발표

### 2020-09-10
1. torch.nn [loss_function] - by 권혁화, 김정운
    - binary_cross_entropy
        + Function that measures the Binary Cross Entropy between the target and the output

    - binary_cross_entropy_with_logits
        + Function that measures Binary Cross Entropy between target and output logits.

    - cosine_embedding_loss
        + See CosineEmbeddingLoss for details.

    - cross_entropy
        + This criterion combines log_softmax and nll_loss in a single function.

    - hinge_embedding_loss
        + See HingeEmbeddingLoss for details.

    - l1_loss
        + Function that takes the mean element-wise absolute value difference.
    
    - mse_loss
        + input x, predict y 간에 element-wise 로 loss값을 mean square 연산하여 구함
        + binary classification 에 대해서 non-convex하기 때문에 성능이 좋진 않음

    - margin_ranking_loss
        + 2개의 1D mini-batch Tensor( 입력x1, 입력x2 ) 및 1개의 1D mini-batch Tensor( label Y )에 대한 loss를 계산
        + 만약 y 가 1이면, 첫 번째 입력이 두 번쨰 입력보다 rank가 높다고 가정하고, -1 이면 더 낮다고 가정
        + 특수한 목적으로 각 item에 대해서 순위를 구해야 하는 경우에 사용

    - multilabel_margin_loss
        + multi-class classification 에서 hinge loss를 계산

    - multilabel_soft_margin_loss
        + 입력x 와 predict y 에 대해서 Max-entropy 를 사용하여 "1개 label에 대한 loss : 전체 label에 대한 loss"를 계산하는 방식

    - multi_margin_loss
        + multi-class classification에서 Input x 와 output y 사이의 hinge loss를 계산
        + hinge loss : SVM loss 라고도 함
            : "정답클래스 score"가 "정답이 아닌 클래스 score + Safety margin(=1)" 보다 크면, loss = 0  
            : otherwise "정답이 아닌 클래스 - 정답클래스 + 1"  
        + [출처]https://lsjsj92.tistory.com/391

    - nll_loss
        + C개의 classes를 분류하는 문제에 유용한 방식
        + nll_loss를 사용하는 경우 각 class의 weight 는 1D tensor여야 하는데, 이것은 특히 training set이 불균형할 때 유용하다
        + 즉, 얼마나 성능이 좋지 않은지 알 수 있다
        + (참고) L1 Loss function stands for Least Absolute Deviations. Also known as LAD. L2 Loss function stands for Least Square Errors.

    - smooth_l1_loss
        + element-wise 연산 값 또는 L1(Least Absolute Deviation)이 1 아래로 떨어지는 경우 제곱
        + MSE 방식에 비해서 덜 예민하며, Gradient 값이 증가하는 것을 막을 수 있음

    - soft_margin_loss
        + 입력 tensor 와 predicted tensor 간의 차이를 softmax 를 거쳐 -1 ~ 1 사이로 출력

2. torch.optim [optimizer] - by 홍인화  
    <center><img src="/reference_image/MH.Ji/Deep Learning Conecpt/optimizer.png" width="70%"></center><br>

3. torch.optim.lr_scheduler [lr_scheduler] - by 지명화
    - ReduceLROnPlateau
        + 학습이 잘 되고 있는지 아닌지에 따라 동적으로 lr을 변화. 보통 validation set의 loss를 인자로 주어서 사전에 지정한 epoch동안 loss가 줄어들지 않으면 lr을 감소시키는 방식.

    - LambdaLR
        + 각 파라미터 그룹의 lr을 지정된 함수의 초기 lr 배로 설정. lambda 함수를 하나 받아 그 함수의 결과를 lr로 설정.

    - StepLR
        + step_size epoch마다 감마 비율만큼 각 파라미터 그룹의 lr을 감소.

    - MultiStepLR
        + step마다가 아닌 지정된 epoch에만 gamma 비율로 감소

    - MultiplicativeLR
        + 각 매개 변수 그룹의 lr에 지정된 함수에 주어진 계수를 곱함.

    - ExponentialLR
        + lr을 지수함수적으로 감소.

    - CosineAnnealingLR
        + lr을 cosine 함수의 형태처럼 변화. lr이 커졌다가 작아졌다가 함.

    - CyclicLR
        + CLR에 따라 각 파라미터 그룹의 lr을 설정. 정책은 두 경계 사이의 lr을 일정한 빈도로 순환. 두 경계 사이의 거리는 반복 단위 또는 주기 단위로 확장. 매 배치 후 lr을 변경.

    - OneCycleLR
        + OneCycle에 따라 각 파라미터 그룹의 lr을 설정. 초기 학습률에서 최대 학습률로, 그 다음 최대 학습률에서 초기 학습률보다 훨씬 낮은 최소 학습률로 학습률을 어닐링(단련). 대규모 학습률을 사용하는 신경망의 매우 빠른 훈련. OneCycle은 모든 배치 후 학습률을 변경.

### 2020-09-16
1. Feedback
    - 권혁화
        + RODEO는 Incremetal(증분) learning에 대한 논문
        + 이는 기존에 학습된 모델에 새로운 학습 데이터가 발생할 경우, 모델을 처음부터 다시 학습하는 것이 아니라 새로운 데이터에 대해서만 학습시킨 뒤 업데이트
        + 하지만 Incremental Learning의 경우 성능이 보장되지 않고, 어려운 분야이므로, 현재는 연구분야에 그치고 있음. 따라서 새로운 Object Detection 모델을 선정 및 실습하고, 성능을 확인하는 것을 추천
        + Classification Augmentation에 대해서는 깔끔하게 진행되었음.
        + 추가적으로 ColorJitter에 Contrast를 조정해보는 것도 좋은 시도가 될 것 같음.


    - 김정운
        + 바운딩박스가 겹치는 것을 잘 잡아내는것이 필요하고 우리에게 좋은 접근이다.
        + Faster-RCNN이나 RetinaNet 이렇게 2개만 detector를 적용할 수 있을 뿐아니라 다른 모델로도 코팅이 가능해야할 것 같다.
        + 3D face detection을 찾아보기위해 Prnet을 검색하면 좋다.(3d face reconstruction)


    - 지명화
        + SSD는 작은 물체를 잡는 부분이 약함. 특히나 진행 중인 프로젝트는 제품들이 겹치는 부분이 많을 텐데, SSD는 bounding box를 하나하나씩 잡는데, RGB가 겹치면 객체가 다른 것으로 분류될 수도 있음. 
        + 새로운 논문을 찾아보는 것을 추천
        + 보통 연구를 진행할 때, 가장 무거운 모델부터 돌려서 정확도를 얼추 계산한 다음, 조금씩 가벼운 모델로 바꿔서 연구함.
        + ResNet50의 성능이 ResNet18보다 적게 나온 이유는 꼭 집어서 설명하기란 어렵지만, 단순한 이미지 분류에서 굳이 무거운 모델을 돌릴 이유는 없기 때문이다. 하지만, 성능이 적게나오는 이유를 정확히 알기 위해서는 ResNet18보다 큰, ResNet34, ResNet50, ... 등 여러 개를 돌려보고 난 후 생각해보자.

    - 홍인화
        + 좀 더 많은 어그먼테이션 시도가 필요.

2. Face Detection 모델은 다다음주 이하늘 멘토님이 추가로 설명 예정
    - 기본적으로 Face Detection은 Verification과 Identification이 있다.

3. 다음주까지 과제
    - Image Classification 성능 향상(data augmentation 적용, metrics 변경 등을 이용)
    - Object Detection 모델링

### 2020-10-07
1. Face Detection 교육
    - 유클라디안 거리 
        + n차원에서의 두 점 사이의 거리 [거리개념이 나온 이유는 아웃풋인 네모의 벡터값이 나오는데 그것으로 비교를 하려고할때 loss값을 벡터의 거리를 측정]
        + 유클라디안 거리는 가까울수록 1이다. 

    - K-NN 
        + 새로운 데이터가 들어왓을때 기존의 데이터 사이의 거리를 재서 이웃들을 뽑음 
        + classification = 출력은 소속된 항목
        + regression 출략 = 객체의 특성 , k개의 최근접 이웃한 데이터가 가진값의 평균 
        + 하이퍼파라미터 = 탐색할 이웃수(k), 거리측정법

    - 클러스터링 
        + 비지도학습 = 라벨링이없음. 그래서 클러스터링으로 특징이 가까운 친구들끼리를 같은 소속된 항목으로 갖는다. 

    - FaceNet
        + CNN을 태워서 인베딩함 
        + 임베딩한 거리가 가까울수록 유사한 사람이 됨 
        + 결과값인 벡터값을 뽑아서 라벨링의 벡터랑 
        + L2 = nomalize
        + triplet loss 
        + 새로운 데이터의 앵커값을 얼굴이 매치되는 positive과 매치가 안되는 negative를 한쌍씩 묶어서 학습한다. 

    - Face Detection 종류
        + verification - 동일인 인식 1:1
        + recognition - 사람 인식 1:N
        + clustering - 얼굴 분류 N:N -> 이번 프로젝트에서는 진행 x

2. Object Detection 피드백
    - detection의 정확도와 속도를 함께 가져갈 수 없기 때문에, 프로젝트의 목적성에 따라 선택하는 것이 맞음. 현재 진행하고 있는 프로젝트는 카메라 기반의 실시간이 보장되어야 한다.
    - pre-train 모델을 image classification에서 학습된 모델을 사용한다 하더라도 더 나은 결과를 보여주지는 않음. 원한다면 scratch 방식으로 pre-train을 하는 것이 좋다.

3. 다음주까지 과제
    - lfw dataset 다운로드 -> face recognition
    - face recognition은 기존의 학습된 모델을 쓰는게 제일 최고. 추가로 데이터를 분류하기 위해서는 svm으로 학습하는 것이 좋음.
    - FaceNet을 기본으로 공부, 추가적 논문 => 다음주 발표