# 논문 내용 정리
### Abstract
- 학습되는 unreferenced functions 대신에 layer에 reference하는 residual functions를 학습시키는 형태로 변형시켜본다.
- 경험적인 실험을 통해 이러한 residual networks는 최적화하기도 쉽고, 상당히 증가된 layer에서도 높은 정확도를 얻을 수 있다.
- ImageNet dataset으로 실험한 결과, 3.57%의 에러를 달성했고, COCO object detection dataset에서는 28% 정도 향상되었다.

### 1. Introduction
- very deep model은 학습 데이터 속에 존재하는 대표적인 개념을 추출할 수 있어 학습 결과가 좋다. 하지만 과연 layer가 깊어질수록, 그 layer가 좋은 결과를 낼 수 있을까?
- 이러한 문제에 대해 2가지의 유명한 문제가 있다.
    
    1) vanishing/exploding gradient problem  
        + CNN에서 파라미터를 업데이트할 때, gradient값이 너무 큰 값이나 작은 값으로 포화되어 더 이상 움직이지 않아 학습의 효과가 없어지거나 학습 속도가 아주 느려지는 문제이다.
        + layer가 깊어질수록 이 문제는 더욱 심각해지며, 이 문제를 피하기 위해 batch normailzation(SGD), normalized initialization 등의 기법이 적용되지만, layer 개수가 일정 수를 넘어가게 되면 여전히 문제가 생긴다.
    
    2) degradation problem
        + layer가 깊어지면, 파라미터 수가 비례적으로 늘어나게 되어 overfitting이 아니더라도, layer를 추가하면서 더욱 deep한 모델은 training 에러를 만든다.
    
    <img1>

- Residual Learning
    + 본 논문은 layer를 100개 이상으로 깊게 하면서, 깊이에 따른 학습 효과를 얻을 수 있는 방법을 고민하였고, 그것이 바로 residual learning이다.
    + 평범한 CNN layer 구조에서는 다음과 같이 x를 입력받아 2개의 weight layer를 거쳐 출력 H(x)를 내며, 학습을 통해 최적의 H(x)를 얻는 것이 목표이며, weight layer의 파라미터 값은 그렇게 결정되어야 한다.
    <img2>
    
    + 만약 H(x)를 얻는 것이 목표가 아니라 H(x)-x를 얻는 것이 목표라면, 즉 출력과 입력의 차를 얻을 수 있도록 학습한다면 2개의 weight layer는 H(x)-x를 얻도록 학습이 되어야 한다. 여기서 F(x)=H(x)-x라고 한다면, 출력 H'(x)=F(x)+x가 된다.
    <img3>

    + 기존의 구조와 다른 점은 입력에서 바로 출력으로 연결되는 shortcut 연결이 생긴다. 이 shortcut은 파라미터가 없이 바로 연결되는 구조이기 때문에 연산량 관점에서는 덧셈이 추가되는 것 외에는 차이가 없다.

    + 그래도 이러한 관점으로 바뀐 구조에서는 좋은 효과를 얻을 수 있다. 만약 identity mapping이 최적이라면, nonlinear layers stack에 의해 identity mapping에 맞추는 것보다 잔차(H(x)-x)를 0으로 맞추는 것이 쉬울 것이다. 왜냐하면 최적의 경우라면 F(x)=H(x)-x가 0이 되어야 하기 때문이다.

    + F(x)가 거의 0이 되는 방향으로 학습을 하게 되면 입력의 작은 움직임을 쉽게 검출할 수 있게 된다. 그런 의미에서 F(x)가 작은 움직임, 즉 나머지(residual)를 학습한다는 관점에서 residual learning이라고 불리게 된다.

    + 또한, 입력과 같은 x가 그대로 출력에 연결되기 때문에 파라미터 수에 영향이 없으며, 덧셈이 늘어나는 것을 제외하면 shortcut 연결을 통한 연산량 증가는 없다. 그리고 몇 개의 layer를 건너뛰면서 입력과 출력이 연결되기 때문에 forward나 backward path가 단순해진다.

    + 전체적인 네트워크는 backpropagation과 함께 SGD로 훈련할 수 있으며, 쉽게 common libraries를 사용할 수 있다.

- 결과적으로 residual learning을 통해 다음 2가지 효과를 가질 수 있다.

    1) 기존의 상대적으로 간단하게 층을 쌓을 수 있는 "plain" 망은 depth가 증가함에 따라 더욱 훈련 에러가 발생하여 어려움이 있지만, residual net은 쉽게 최적화가 가능하다.

    2) 이전의 네트워크들보다 대체로 더 좋아진 결과들을 생산해내면서, 늘어난 깊이로 인해 쉽게 정확도를 개선할 수 있다.

### 2. Related Work
- Residual Representations

### 3. Deep Residual Learning


### 4. Experiments


# 새롭게 알게된 내용
### 1. 


# 참고자료
[ResNet](https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220761052425&proxyReferer=https:%2F%2Fwww.google.com%2F)