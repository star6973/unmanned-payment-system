## Inception-v2, 2016

#### inception 특징 

- Conv Filter Factorization
- Rethinking Auxiliary Classifier
- Avoid representational bottleneck

![inception-v2](https://hoya012.github.io/assets/img/image_classification_guidebook/15.PNG)

- 우선 Inception-v1(GoogLeNet)은 VGG, AlexNet에 비해 parameter수가 굉장히 적지만, 여전히 많은 연산량을 필요로 합니다. Inception-v2에서는 연산의 복잡도를 줄이기 위한 여러 Conv Filter Factorization 방법을 제안하고 있습니다. 우선 VGG에서 했던 것처럼 5x5 conv를 3x3 conv 2개로 대체하는 방법을 적용합니다. 여기서 나아가 연산량은 줄어들지만 receptive field는 동일한 점을 이용하여 n x n conv를 1 x n + n x 1 conv로 쪼개는 방법을 제안합니다.

- 그 다음은 Inception-v1(GoogLeNet)에서 적용했던 auxiliary classifier에 대한 재조명을 하는 부분입니다. 여러 실험과 분석을 통해 auxiliary classifier가 학습 초기에는 수렴성을 개선시키지 않음을 보였고, 학습 후기에 약간의 정확도 향상을 얻을 수 있음을 보였습니다. 또한 기존엔 2개의 auxiliary classifier를 사용하였으나, 실제론 초기 단계(lower)의 auxiliary classifier는 있으나 없으나 큰 차이가 없어서 제거를 하였다고 합니다.

![inception-v2](https://hoya012.github.io/assets/img/image_classification_guidebook/16.PNG)

- 마지막으론 representational bottleneck을 피하기 위한 효과적인 Grid Size Reduction 방법을 제안하였습니다. representational bottleneck이란 CNN에서 주로 사용되는 pooling으로 인해 feature map의 size가 줄어들면서 정보량이 줄어드는 것을 의미합니다. 이해를 돕기 위해 위의 그림으로 설명을 드리면, 왼쪽 사진과 같이 pooling을 먼저 하면 Representational bottleneck이 발생하고, 오른쪽과 같이 pooling을 뒤에 하면 연산량이 많아집니다. 그래서 연산량도 줄이면서 Representational bottleneck도 피하기 위해 가운데와 같은 방식을 제안하였고, 최종적으론 맨 오른쪽과 같은 방식을 이용하였다고 합니다.

![inception-v22](https://hoya012.github.io/assets/img/image_classification_guidebook/17.PNG)

- 위와 같은 아이디어가 적용된 최종 Inception-v2의 architecture 구조는 위의 표와 같습니다. 기존 Inception-v1은 7x7 conv 연산이 가장 먼저 수행이 되었는데, 위의 Factorization 방법에 의거하여 3x3 conv 연산 3개로 대체가 되었고, figure5, 6, 7의 기법들이 차례대로 적용이 되어있는 것을 확인하실 수 있습니다.


