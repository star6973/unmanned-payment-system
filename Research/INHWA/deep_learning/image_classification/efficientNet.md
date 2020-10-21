#### Model Scaling

![image1](https://hoya012.github.io/assets/img/efficientnet/2.PNG)

- 일반적으로 ConvNet의 정확도를 높일 때 잘 짜여진 모델 자체를 찾는 방법도 있지만, 기존 모델을 바탕으로 Complexity를 높이는 방법도 많이 사용합니다.

- 위의 그림은 이미 존재하는 모델의 size를 키워주는 여러 방법들을 보여주고 있습니다. 대표적으로 filter의 개수를(channel의 개수를) 늘리는 width scaling 와 layer의 개수를 늘리는 depth scaling과 input image의 해상도를 높이는 resolution scaling 이 자주 사용됩니다.

- 기존 방식들에서는 위의 3가지 scaling을 동시에 고려하는 경우가 거의 없었습니다.

>> 직관적으로 생각해보면 Input image가 커지면 그에 따라서 receptive field도 늘려줘야 하고, 더 커진 fine-grained pattern들을 학습하기 위해 더 많은 channel이 필요한 건 reasonable한 주장입니다. 



#### Compound Scaling

- 이 논문에서는 모델(F)를 고정하고 depth(d), width(w), resolution(r) 3가지를 조절하는 방법을 제안하고 있는데, 이때 고정하는 모델(F)를 좋은 모델로 선정하는 것이 굉장히 중요합니다. 아무리 scaling factor를 조절해도, 초기 모델 자체의 성능이 낮다면 임계 성능도 낮기 때문입니다. 이 논문에서는 MnasNet과 거의 동일한 search space 하에서 AutoML을 통해 모델을 탐색하였고, 이 과정을 통해 찾은 작은 모델을 EfficientNet-B0 이라 부르고 있습니다.

-  이 논문에서는 MnasNet과 거의 동일한 search space 하에서 AutoML을 통해 모델을 탐색하였고, 이 과정을 통해 찾은 작은 모델을 EfficientNet-B0 이라 부르고 있습니다.


