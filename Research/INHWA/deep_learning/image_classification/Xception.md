## Xception, 2017

- 본 논문은 Inception 구조에 대한 고찰로 연구를 시작하였으며, 추후 많은 연구들에서 사용이 되는 연산인 depthwise-separable convolution 을 제안하고 있습니다. Inception v1, 즉 GoogLeNet에서는 여러 갈래로 연산을 쪼갠 뒤 합치는 방식을 이용함으로써 cross-channel correlation과 spatial correlation을 적절히 분리할 수 있다고 주장을 하였습니다. 쉽게 설명하자면, channel간의 상관 관계와 image의 지역적인 상관 관계를 분리해서 학습하도록 가이드를 주는 Inception module을 제안한 셈이죠.


#### Depthwise Convolution

- 기본적인 개념은 쉽다. 위 처럼 H*W*C의 conv output을 C단위로 분리하여 각각 conv filter을 적용하여 output을 만들고 그 결과를 다시 합치면 conv filter가 훨씬 적은 파라미터를 가지고서 동일한 크기의 아웃풋을 낼 수 있다. 또한 각 필터에 대한 연산 결과가 다른 필터로부터 독립적일 필요가 있을 경우에 특히 장점이 된다.

![separable](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FAyNcz%2FbtqAmtNdX2P%2FUTwjESXKnxRUYuPgHKTEp1%2Fimg.png)


#### Depthwise Separable Convolution 


![depthwise1](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FKy3Le%2FbtqAmtTZrEs%2FyUlkWdPU1HNa8STmjJe1NK%2Fimg.jpg)

- Depthwise convolution을 먼저 수행한 후 Pointwise convolution을 수행한다. 이를 통해서 3x3의 필터를 통해 conv 연산도 진행하고, 서로 다른 channel들의 정보도 공유하면서 동시에 파라미터 수도 줄일 수 있다.


- Xception은 여기서 더 나아가서 아예 channel간의 상관 관계와 image의 지역적인 상관 관계를 완벽하게 분리하는 더 높은 목표를 세우고 연구를 시작하였고, 위의 그림의 Figure 4와 같은 연산 구조를 이용하면 cross-channel correlation과 spatial correlation이 완벽하게 분리가 될 수 있음을 제안하였습니다.


![xception_architecture](https://hoya012.github.io/assets/img/image_classification_guidebook/34.PNG)

>> Xception의 architecture 구조는 다음과 같으며 Entry flow, Middle flow, Exit flow로 구분하여 그림을 그렸습니다. 다만 대부분의 연산들이 단순 반복되는 구조로 되어있어서 구현하기엔 어렵지 않습니다. ResNet에서 봤던 shortcut도 포함이 되어있고 Batch Normalization도 들어가 있는 등 정확도를 높이기 위한 여러 시도를 포함하고 있다
