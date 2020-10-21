#Loss Function 조사
1. mse_loss
    - input x, predict y 간에 element-wise 로 loss값을 mean square 연산하여 구함
    - binary classification 에 대해서 non-convex하기 때문에 성능이 좋진 않음

1. margin_ranking_loss
    - 2개의 1D mini-batch Tensor( 입력x1, 입력x2 ) 및 1개의 1D mini-batch Tensor( label Y )에 대한 loss를 계산
    - 만약 y 가 1이면, 첫 번째 입력이 두 번쨰 입력보다 rank가 높다고 가정하고, -1 이면 더 낮다고 가정
    - 특수한 목적으로 각 item에 대해서 순위를 구해야 하는 경우에 사용

1. multilabel_margin_loss
    - multi-class classification 에서 hinge loss를 계산

1. multilabel_soft_margin_loss
    - 입력x 와 predict y 에 대해서 Max-entropy 를 사용하여 "1개 label에 대한 loss : 전체 label에 대한 loss"를 계산하는 방식

1. multi_margin_loss
    - multi-class classification에서 Input x 와 output y 사이의 hinge loss를 계산
    - hinge loss : SVM loss 라고도 함<br>: "정답클래스 score"가 "정답이 아닌 클래스 score + Safety margin(=1)" 보다 크면, loss = 0<br>: otherwise "정답이 아닌 클래스 - 정답클래스 + 1" <br>  [출처]https://lsjsj92.tistory.com/391

1. nll_loss
    - C개의 classes를 분류하는 문제에 유용한 방식
    - nll_loss를 사용하는 경우 각 class의 weight 는 1D tensor여야 하는데, 이것은 특히 training set이 불균형할 때 유용하다
    - 즉, 얼마나 성능이 좋지 않은지 알 수 있다
    - (참고) L1 Loss function stands for Least Absolute Deviations. Also known as LAD. L2 Loss function stands for Least Square Errors.

1. smooth_l1_loss
    - element-wise 연산 값 또는 L1(Least Absolute Deviation)이 1 아래로 떨어지는 경우 제곱
    - MSE 방식에 비해서 덜 예민하며, Gradient 값이 증가하는 것을 막을 수 있음

1. soft_margin_loss
    - 입력 tensor 와 predicted tensor 간의 차이를 softmax 를 거쳐 -1 ~ 1 사이로 출력