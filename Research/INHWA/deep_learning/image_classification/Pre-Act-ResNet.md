## Pre-Act ResNet, 2016

- Pre-Act는 Pre-Activation의 약자로, Residual Unit을 구성하고 있는 Conv-BN-ReLU 연산에서 Activation function인 ReLU를 Conv 연산 앞에 배치한다고 해서 붙여진 이름입니다.

![preact](https://hoya012.github.io/assets/img/image_classification_guidebook/13.PNG)

- 우선 기존 ResNet에서 사용되던 identity shortcut을 5가지 다양한 shortcut으로 대체하는 방법과, 각각을 적용하였을 때의 실험 결과를 위에 그림에서 확인하실 수 있습니다. 실험 결과, 아무것도 하지 않았을 때, 즉 identity shortcut일 때 가장 성능이 좋은 것을 확인할 수 있습니다. 이에 대한 짤막한 discussion이 있는데, 제안하는 gating 기법과 1x1 conv등을 추가하면 표현력(representational ability)은 증가하지만 학습 난이도를 높여서 최적화하기 어렵게 만드는 것으로 추정된다고 설명하고 있습니다.

#### 액티베이션 함수의 위치에 따른 에러값

![preactresNet](https://hoya012.github.io/assets/img/image_classification_guidebook/14.PNG)

- 다음 그림은 activation function의 위치에 따른 test error 결과를 보여주고 있습니다. 기존엔 Conv-BN-ReLU-Conv-BN을 거친 뒤 shortcut과 더해주고 마지막으로 ReLU를 하는 방식이었는데, 총 4가지 변형된 구조를 제안하였고, 그 중 full pre-activation 구조일 때 가장 test error가 낮았고, 전반적인 학습 안정성도 좋아지는 결과를 보인다고 합니다.

