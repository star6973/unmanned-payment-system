## ResNet
- Microsoft Research에서 제안한 구조이며, ILSVRC 2015 대회에서 1위를 한 모델이다.

- ResNet은 3x3 convolution이 반복 사용한다는 점에서 VGGNet과 유사한 구조를 가지고 있다. Layer의 개수에 따라 ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152 등 5가지 버전으로 나타낼 수 있다. Layer의 개수를 많이 사용할수록 연산량과 parameter의 개수는 커지지만 정확도도 좋아지는 효과를 얻을 수 있다. 즉, GoogLeNet이 22개 층으로 구성된 것에 비해, ResNet은 152개의 층을 가지면서 약 7배나 깊어졌다.

- 특징
    + [Deep Residual Learning for Image Recognition](https://github.com/star6973/lotte_studying/blob/myunghwa/Research/MH.Ji/Deep%20Residual%20Learning%20for%20Image%20Recognition.md)

- 실험
    + ResNet 팀은 실험을 위한 망을 설계하면서 VGGNet의 설계 철학을 많이 이용했다. 그래서 대부분의 convolutional layer은 3x3 kernel을 갖도록 하였으며, 복잡도를 줄이기 위해 max pooling(1곳 제외), hidden fully connected layer, dropout 등을 사용하지 않았다.

    + 또한, 다음 2가지 원칙을 지켰다.
        1) 출력 feature map의 크기가 같은 경우, 해당 모든 layer는 모두 동일한 수의 filter를 갖는다.  
        2) Feature map의 크기가 절반으로 작아지는 경우는 연산량의 균형을 맞추기 위해 필터의 수를 두 배로 늘린다. Feature map의 크기를 줄일 때는 pooling을 사용하는 대신에 convolution을 수행할 때, stride의 크기를 2로 설정하는 방식을 취한다.  

    + 비교를 위해 plain network와 residual network로 구별하였다. plain network의 경우도 VGGNet보다 filter의 수를 줄이고, 복잡도를 낮춤으로써 34 layer의 plain network가 19 layer의 VGGNet에 비해 연산량을 20% 미만으로 줄였다. Residual network의 구조를 간단하기 위해서, 매 2개의 convolutional layer마다 shortcut connection이 연결되도록 하였다.
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/122.png" width="50%"></center><br>

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/123.png" width="70%"></center><br>
    
    + 위의 표를 보면, 18-layer와 34-layer는 동일한 구조를 사용하였고, 다만 각 layer에 있는 convolutional layer의 수만 다르다는 것을 알 수 있다.

    + 실험 결과는 다음과 같다.
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/124.png" width="70%"></center><br>

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/125.png" width="70%"></center><br>

    + plain network의 경우는 34-layer의 결과가 18-layer보다 결과가 나쁘다는 것을 알 수 있고, residual network는 34-layer가 18-layer의 결과보다 좋다는 것을 알 수 있다. 또한, 학습의 초기 단계에서 residual net의 수렴 속도가 plain network보다 빠르다는 것을 알 수 있다.

    + 이번에는 학습에 걸리는 시간을 고려해서 ResNet-50, ResNet-101, ResNet-152에 대해서 기본 구조를 조금 변경시켰고, residual function은 1x1, 3x3, 1x1으로 아래 그림처럼 구성이 된다. 이러한 구조를 Bottleneck 구조라고 하며, 차원을 줄였다가 뒤에서 다시 차원을 늘리는 모습이 병목처럼 보인다고 붙인 이름이다.
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/126.png" width="70%"></center><br>

    + 이렇게 구성한 이유는 연산 시간을 줄이기 위함이다. 맨 처음 1x1 convolution은 NIN(Network-in-Network)이나 GoogLeNet의 Inception 구조에서 살펴본 것처럼 차원을 줄이기 위한 목적이다. 이렇게 차원을 줄인 뒤 3x3 convolution을 수행한 후, 마지막 1x1 convolution은 다시 차원을 확대시키는 역할을 한다. 결과적으로 3x3 convolution 2개를 연결시킨 구조에 비해 연산량을 절감시킬 수 있다.

    + ResNet의 모든 구조에 대한 top-1, top-5 error의 결과는 아래의 그림과 같다.
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/127.png" width="70%"></center><br>

- 참고자료
> [Deep Learning Image Classification Guidebook [1] LeNet, AlexNet, ZFNet, VGG, GoogLeNet, ResNet](https://hoya012.github.io/blog/deeplearning-classification-guidebook-1/)

> [ResNet [1], [2] - 라온피플 머신러닝 아카데미](http://blog.naver.com/laonple/220738560542)