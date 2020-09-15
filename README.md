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
| 7주 | 2020-09-14 ~ <br> 2020-09-18 </br> | 기존 모델 다루기 실습 | 9/10~9/13 - Image Classification 테스팅 (VGGNet, ResNet, MobileNet, EfficientNet) 완료, 9/14 - Object Detection Tutorial |
| 8주 | 2020-09-21 ~ <br> 2020-09-25 </br> | Transfer Learning 준비 |  |
| 9주 | 2020-09-28 ~ <br> 2020-09-29 </br> | Transfer Learning 실습 |  |
| 10주 | 2020-10-05 ~ <br> 2020-10-08 </br> | 결과 검토 및 모델 수정 |  |
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