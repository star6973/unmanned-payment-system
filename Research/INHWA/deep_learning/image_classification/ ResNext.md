## ResNext

![difference](https://hoya012.github.io/assets/img/image_classification_guidebook/36.PNG)

- 위의 그림은 ResNet과 ResNext의 가장 큰 차이점을 보여주고 있습니다. 기존 ResNet은 Res Block의 반복 구조로 이루어져 있고, 지난 2편에서 소개드렸던 여러 ResNet의 변형들도 ResNet의 width(filter 개수)와 depth(layer 개수)를 조절하는 시도를 하였는데, 본 논문에서는 width와 depth 외에 cardinality 라는 새로운 차원의 개념을 도입합니다.
( MobileNet에는 deep하고 narrow한게 더 효과가 좋았다)

- Cardinality는 한글로 번역하면 집합의 크기 또는 집합의 원소 개수를 의미하는데, CNN에서는 하나의 block 안의 transformation 개수 혹은 path, branch의 개수 혹은 group의 개수 정도로 정의할 수 있습니다. 위의 그림에서는 64개의 filter 개수를 32개의 path로 쪼개서 각각 path마다 4개씩 filter를 사용하는 것을 보여주고 있으며, 이는 AlexNet에서 사용했던 Grouped Convolution 과 유사한 방식입니다. 사실 AlexNet은 GPU Memory의 부족으로 눈물을 머금고 Grouped Convolution을 이용하였는데 ResNext에서는 이러한 연산이 정확도에도 좋은 영향을 줄 수 있음을 거의 최초로 밝힌 논문입니다.

#### cardinality 
- 예를들어 입력 채널이 256채널이고, Cardinality가 32이면, 8 채널씩 나눠서 group convolution을 진행하는 구조입니다.

![architecture](https://hoya012.github.io/assets/img/image_classification_guidebook/36.PNG)

- 이는 AlexNet에서 사용했던 Grouped Convolution 과 유사한 방식입니다. 사실 AlexNet은 GPU Memory의 부족으로 눈물을 머금고 Grouped Convolution을 이용하였는데 ResNext에서는 이러한 연산이 정확도에도 좋은 영향을 줄 수 있음을 거의 최초로 밝힌 논문입니다.


![architecture](https://hoya012.github.io/assets/img/image_classification_guidebook/37.PNG)

- ResNext의 building block은 가장 먼저 제안한 단순한 (a) 구조에서, 3x3 conv 이후 결과를 미리 concat해주고 1x1 conv를 해주는 (b) 구조, 마지막으로 초기 1x1 conv를 하나로 합치고 중간 3x3 conv를 grouped convolution으로 대체하는 (c) 구조가 모두 동일한 과정을 수행하며, 편한 구현을 위해 실제론 (c) 구조를 사용하였습니다

