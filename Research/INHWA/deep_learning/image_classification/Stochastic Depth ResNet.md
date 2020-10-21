## Stochastic Depth ResNet, 2016

- 이번 논문은 2016년 ECCV에 발표된 “Deep Networks with Stochastic Depth”라는 논문이며, vanishing gradient로 인해 학습이 느리게 되는 문제를 완화시키고자 stochastic depth 라는 randomness에 기반한 학습 방법을 제안합니다. 이 방법은 2019년 말 ImageNet 데이터셋에 대해 State-of-the-art 성능을 달성한 “Self-training with Noisy Student improves ImageNet classification” 논문 리뷰 글에서 noise 기법으로 사용된 기법입니다. 사실 이 논문은 새로운 architecture를 제안했다고 보기는 어렵습니다. 기존 ResNet에 새로운 학습 방법을 추가했다고 보는게 맞지만, ResNet의 layer 개수를 overfitting 없이 크게 늘릴 수 있는 방법을 제안했습니다

- 비슷한 아이디어로는 여러분들이 잘 아시는 Dropout이 있습니다. Dropout은 network의 hidden unit을 일정 확률로 0으로 만드는 regularization 기법이며, 후속 연구론 아예 connection(weight)을 끊어버리는 DropConnect(2013 ICML) 기법, MaxOut(2013 ICML), MaxDrop(2016 ACCV) 등의 후속 연구가 존재합니다. 위의 방법들은 weight나 hidden unit(feature map)에 집중했다면, Stochastic depth란 network의 depth를 학습 단계에 random하게 줄이는 것을 의미합니다.

![StochasticDepthResNet](https://hoya012.github.io/assets/img/image_classification_guidebook/24.PNG)

>>ResNet으로 치면 확률적으로 일정 block을 inactive하게 만들어서, 해당 block은 shortcut만 수행하는, 즉 input과 output이 같아져서 아무런 연산도 수행하지 않는 block으로 처리하여 network의 depth를 조절하는 것입니다. 이 방법은 학습시에만 사용하고 test 시에는 모든 block을 active하게 만든 full-length network를 사용합니다.

![result](https://hoya012.github.io/assets/img/image_classification_guidebook/25.PNG)

- Stochastic Depth ResNet은 CIFAR-10, SVHN 등에 대해선 test error가 줄어드는 효과가 있지만, ImageNet과 같이 복잡하고 큰 데이터 셋에서는 별다른 효과를 보지 못했습니다. 다만 CIFAR-10과 같이 비교적 작은 데이터셋에서는 ResNet을 1202 layer를 쌓았을 때 기존 ResNet은 오히려 정확도가 떨어지는 반면 Stochastic Depth ResNet은 정확도가 향상되는 결과를 보여주고 있습니다.

