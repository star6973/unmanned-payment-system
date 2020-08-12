# VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION

Abstract
    - 큰 이미지에서 CNN 의 depth가 정확도에 미치는 영향에 대한 연구 진행
    - 주요한 업적은 3by3의 매우 작은 convolution filter를 사용하여 깊이가 증가하는 네트워크에 대한 철저한 평가
    - 이것은 깊이를 16~19 가중치 layer로 할 경우에 이러한 선행기술들이 성공할 수 있다는 것을 보여준다.<br><br>
    - 이러한 발견은 ImageNet Challenge(2014)의 기초이며, 연구팀은 localization 과 classification 분야에서 각각 1위, 2위를 확보<br><br>
    - 또한 이들의 발표는 다른 data set 또한 일반화가 잘 이뤄지며, state of the art에도 등극
<br><br>
1. Introduction
    - 컴퓨터 비전 분야에서 CNN 이 성공을 거두면서 accuracy를 높이기 위한 다양한 시도<br><br>
    - ex1 : 더 작은 크기의 receptive window size를 사용하고, 첫째 convolution layer의 stride 더 작게 하는 방식
    - ex2 : 전체 이미지와 여러 스케일에 걸쳐 조밀하게 training 과 testing을 실시<br><br>
    - 본 논문에서 이것들 이외의 중요한 측면인 *depth* 에 대해서 언급
    - 논문의 끝에서 저자는 다른 parameter들을 수정하고, 지속적으로 convolution layer를 증가시켜 network의 depth를 증가시켰다.
    - 그것은 모든 층에서 매우 작은 convolution filter(3 by 3)를 사용했기 때문에 실현 가능하다.<br><br>
    - CNN 모델의 accuracy 향상의 결과로 state of the art 의 ILSVRC accuracy 와 classification 부문을 성과를 냈다.
    - 또한 상대적으로 단순한 pipeline을 가진 다른 이미지 인식 datasets에도 적용 가능하다는 성과를 얻었다.(예를 들면 fine tuning 없이 deep feature 데이터를 SVM으로 분류)
<br><br>
1. Convolution Network Configuration
    1.1 Architecture
        - Training 동안 Network의 입력 데이터의 크기는 224 by 224 RGB image의 고정된 크기를 사용
        - 유일한 전처리 과정은 Training set에서 계산된 RGB값의 평균을 각 픽셀에서 빼줌
        - convolution layer는 이미지를 거쳐 들어가는데, 이때 필터는 3 by 3으로 매우 작은 사이즈 사용
        - 논문에서 사용된 또 다른 필터는 1 by 1 필터이며, 입력 채널을 선형 변환하는 경우 사용
        - 3 by 3 convolution layer에서 stride는 1픽셀로 고정
        - 공간 pooling은 5개의 max-pooling layer에 의해서 수행되며, 크기는 2 by 2 픽셀이고, stride 는 2 이다.<br><br>
        - convolution layer층은 3층의 fully connected layer에 의해서 수행된다.
        - 처음 2개의 층은 각 4096 채널
        - 세 번째 층은 1000개의 ILSVRC 분류를 수행하기 때문에 각 클래스 수와 동일한 1000개의 채널을 가진다.
        
    
    