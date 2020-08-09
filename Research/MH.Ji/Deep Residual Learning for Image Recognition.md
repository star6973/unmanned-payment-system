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
    + VLAD
        + vocabulary tree 기반의 방법은 비주얼 워드의 빈도수만을 이용하고 있어서, 정보의 표현에 제약이 있지만, VLAD는 이러한 제약을 개선하고자 해당 비주얼 워드를 선택한 특징들의 통계적 특성을 반영하는 방법을 제안했다.
    
    + low-level vision과 컴퓨터 그래픽에서 편미분 방정식(PDE, Partial Differential Equations)를 해결하기 위해 널리 사용되는 방법은 Multigrid 방법이다. 이 방법은 시스템을 여러 scale의 하위 문제로 재구성한다. 여기서 각 하위 문제는 coarser(거칠다)하고 finer(미세하다)한 scale 사이의 잔차(residual) 해결법에 책임을 지고 있다.

- Shortcut Connections
    + 초기의 MLP 훈련은 네트워크 안에서부터 밖으로의 선형적인 layer를 추가하는 형태로 실습해왔다. 출력에 대한 입력에서는 몇 개의 중간 layer가 직접적으로 보조 분류기에 연결되어서, gradient의 소실/폭발 문제를 해결한다.
    + 본 논문에서는 shortcut connection을 실행하여, layer response, gradient 및 propaged errors를 중점적으로 다뤄볼 것이다.
    + "inception" layer는 shortcut branch와 몇 가지 deeper branch를 구성한다.
    + 

### 3. Deep Residual Learning


### 4. Experiments


# 새롭게 알게된 내용
### 1. Multigrid
- multigrid의 기본적인 아이디어는 projection이다.

- 편미분 방정식(PDE)을 많은 정확도로 풀고 싶다고 가정하고, 많은 점이 있는 매우 미세한 격자에서 도메인을 구별해보자. 반복적인 문제 해결법인 jacobi, gauss seidel 등을 사용해본다면, 하루 이상의 시간이 걸릴 것이다. 이러한 반복 문제 해결법이 빠르게 작동하지 않는 이유는 일반적으로 이와 같은 큰 방정식 시스템을 설정할 때, 행렬 자체의 고유값이 1에 매우 가깝기 때문이다.

- 이러한 문제가 발생하는 이유는 많은 반복 문제 해결법의 수렴 속도는 가장 큰 고유 값과 반비례하기 때문이다. 따라서 가장 큰 고유 값이 1에 가까울수록 반복 방법이 느려진다.

- Jacobi method
    + 선형 연립방정식에서 대각 행렬에 0이 포함되어 있지 않은 경우, 해를 계산하는 방법에는 크게 직접법(direct method)과 반복법(iterative method)가 있다. 두 방식의 가장 큰 차이는 전자는 단 한번의 행렬 계산으로 정확한 해를 구하지만, 후자는 행렬 계산을 반복하여 근사해를 구하는 것이다. 행렬의 크기가 작다면 전자의 방식이 효과적이지만, 행렬이 커지게 되면 후자가 효과적이다.

    + 반복 계산은 초기에 x를 임의의 값으로 추정하여 반복식의 우변에 대입하여 좌변의 x를 구하고, 이 값을 다시 반복식의 우변에 대입하여 새로운 x를 구하는 반복과정을 수행한다. 반복계산은 x가 원하는 허용오차 범위에 들어오게 되면 멈추게 된다.
    
    + Jacobi method는 가장 기본적인 반복법으로써, 반복계산의 회수를 줄여 신속하게 계산하기 위해 여러가지 유형의 변형된 반복법들이 소개되었다. 가장 대표적인 변형된 반복법으로는 가우스-시덜법(Gauss-Sidel method)이 있다.

- grid point가 적은 문제를 거친 그리드라고 하는데, 이것이 문제를 해결하는 것이 더 빠르다. 더 중요한 것은 거친 그리드의 솔루션이 더 미세한 그리드에서 문제를 해결하기 위한 좋은 출발점이다.

- 원래의 미세 그리드 문제에 적용되는 반복 문제 해결법에서 오류의 푸리에 모드를 보자. 처음 몇 번의 반복에서 고주파 오류의 대부분이 제거되지만, 여전히 남아있고 빠르게 사라지지 않는 저주파 오류가 있다. 실제로, 표준 반복 방법이 빠르게 수렴하지 못하는 이유가 바로 저주파 오류 때문이다.

- 굵은 격자에서 문제를 해결하면, 미세 격자보다 훨씬 낮은 주파수 오류를 훨씬 더 빠르게 제거할 수 있다. 따라서 거친 그리드의 문제를 해결하면 더 낮은 주파수 오류가 크게 줄어든다.




# 참고자료
[ResNet](https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220761052425&proxyReferer=https:%2F%2Fwww.google.com%2F)
[Jacobi method](https://kor.midasuser.com/nfx/techpaper/keyword_view.asp?pg=&sk=&bid=&nCat=&nIndex=&sHtml=&idx=293)
