## Wide ResNet, 2016

- 오늘의 마지막 소개드릴 논문은 2016년 BMVC에 발표된 “Wide Residual Networks” 논문입니다. 처음 소개드렸던 Pre-Act ResNet, 방금 소개드린 Stochastic Depth ResNet과 같이 ResNet의 성능을 높이기 위한 여러 실험들을 수행한 뒤, 이를 정리한 논문입니다. 나아가 정확도를 높이기 위해 Layer(Depth)만 더 많이 쌓으려고 해왔는데, Conv filter 개수(Width)도 늘리는 시도를 하였고, 여러 실험을 통해 Wide ResNet 구조를 제안하였습니다. 마지막으로, BN의 등장 이후 잘 사용되지 않던 dropout을 ResNet의 학습 안정성을 높이기 위해 적용할 수 있음을 보였습니다.

![widedropout](https://hoya012.github.io/assets/img/image_classification_guidebook/26.PNG)

- 위의 그림 1과 같이 Conv layer의 filter 개수를 늘리고, dropout을 사용하는 방법이 효과적임을 실험을 통해 보여주고 있습니다. 또한 병렬처리 관점에서 봤을 때, layer의 개수(depth)를 늘리는 것보다 Conv filter 개수(width)를 늘리는 것이 더 효율적이기 때문에 본인들이 제안한 WRN-40-4 구조가 ResNet-1001과 test error는 유사하지만 forward + backward propagation에 소요되는 시간이 8배 빠름을 보여주고 있습니다.

- 이 논문에서는 depth 외에 width도 고려해야 한다는 점이 핵심인데, 2019년에 학계를 뜨겁게 달궜던 EfficientNet 에서는 한발 더 나아가 width와 depth 뿐만 아니라 input resolution을 동시에 고려하여서 키워주는 compound scaling 방법을 통해 굉장히 효율적으로 정확도를 높이는 방법을 제안했습니다. WRN과 같은 연구가 있었기 때문에 EfficientNet도 등장할 수 있지 않았나 생각해봅니다.
