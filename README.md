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