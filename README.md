# KIRO & 롯데정보통신
## 팀원

1. 김정운
2. 권혁화
3. 지명화
4. 홍인화


# 피드백
## 2020-08-11
1. Image classification에 관한 추천 논문 자료
2. google drive에 공유자료 업로드

## 20202-08-18
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

## 2020-08-19
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