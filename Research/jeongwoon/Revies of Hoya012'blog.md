#### * data Augmentation - 데이터 변조 작업을 말한다.
#### - 이미지 데이터에서 예를 들자면 다음과 같이 3가지가 있다.
#### -- 좌우반전 : 좌우데이터를 각각 만들어주고 각각의 데이터 개수를 반씩 넣어주면 학습성능이 좋아진다.
#### -- 이미지자르기 : 동물의 꼬리와 귀를 보고 그 동물로 판단할 화률이 각각 50%, 30%라고 할때 각각의 잘라진 이미지를 따로 넣어서 학습시키면 성능이 좋으진다.
#### -- 밝기조절 : 어두운것부터 밝은 이미지 데이터까지 모두 넣어주면 성능이 좋아진다.

#### * Augmentation을 하는 중요한 이유
#### - Preprocessing과 augmentation을 하면 성능이 대체로 좋아진다.
#### - 원본에 추가되는 개념이니 성능이 떨어지지 않은다.
#### - 쉽고 패턴이 정해져있다.

#### * data preprocessing - 데이터 전처리과정이다. 주로 벡터화와 값정규화가 있다.


#### VGG (2014)
#### - 3X3 convolution filter를 깊게 쌓는다는 것이 핵심이다.
#### - 11, 13, 16, 19개 등 더 많은 수의 layer를 사용한다.
#### - 이렇게 3X3 filter를 중첩하여 쌓는 이유는 3개의 conv layer를 중첩하면 1개의 7X7 conv layer와 receptive field가 같아지지만, activation function을 더 많이 사용할 수 있어서 더 많은 비선형성을 얻을 수 있으며 parameter수도 줄어드는 효과를 얻을 수 있다.
##### ** receptive field : 수용장, 입력이 [16X16X20]인 경우라면 일반적으로 convolution에서 하나의 필터의 크기는 [wXhX20]가 되어야 한다. 이때 receptive filed는 wXhX20이다.

#### GoogLenet(Inception) (2014)
#### - 우선 Inception module 이라 불리는 block 구조를 제안한다. 기존에는 각 layer 간에 하나의 convolution 연산, 하나의 pooling 연산으로 연결을 하였다면, inception module은 총 4가지 서로 다른 연산을 거친 뒤 feature map을 channel 방향으로 합치는 concatenation을 이용하고 있다는 점이 가장 큰 특징이며, 다양한 receptive field를 표현하기 위해 1x1, 3x3, 5x5 convolution 연산을 섞어서 사용을 했다. 이 방식을 Naïve Inception module 이라고 한다. 여기에 추가로, 3x3 conv, 5x5 conv 연산이 많은 연산량을 차지하기 때문에 두 conv 연산 앞에 1x1 conv 연산을 추가하여서 feature map 개수를 줄인 다음, 다시 3x3 conv 연산과 5x5 conv 연산을 수행하여 feature map 개수를 키워주는 bottleneck 구조를 추가한 Inception module with dimension reduction 방식을 제안했다. 이 덕에 Inception module의 연산량을 절반이상 줄일 수 있다.
#### - GoogLeNet은 Inception module을 총 9번 쌓아서 구성이 되며, 3번째와 6번째 Inception module 뒤에 classifier를 추가로 붙여서 총 3개의 classifier를 사용하였고, 이를 Auxiliary Classifier 라 부릅니다. 가장 뒷 부분에 Classifier 하나가 존재하면 input과 가까운 쪽(앞 쪽)에는 gradient가 잘 전파되지 않을 수 있는데, Network의 중간 부분, 앞 부분에 추가로 softmax Classifier를 붙여주어 vanishing gradient를 완화시킬 수 있다고 주장하고 있습니다. 다만 Auxiliary Classifier로 구한 loss는 보조적인 역할을 맡고 있으므로, 기존 가장 뒷 부분에 존재하던 Classifier보단 전체적으로 적은 영향을 주기 위해 0.3을 곱하여 total loss에 더하는 식으로 사용을 하였다고 합니다. 학습 단계에만 사용이 되고 inference 단계에선 사용이 되지 않으며, 이유론 inference 시에 사용하면 성능 향상이 미미하기 때문입니다.
#### - 마지막으로 대부분의 CNN의 대부분의 parameter를 차지하고 있는 Fully-Connected Layer를 NIN 논문에서 제안된 방식인 Global Average Pooling(GAP) 으로 대체하여 parameter 수를 크게 줄이는 효과를 얻었습니다. GAP란 각 feature map의 모든 element의 평균을 구하여 하나의 node로 바꿔주는 연산을 뜻하며, feature map의 개수만큼의 node를 output으로 출력하게 됩니다. GoogLeNet에서는 GAP를 거쳐 총 1024개의 node를 만든 뒤 class 개수(ImageNet=1000)의 output을 출력하도록 하나의 Fully-Connected layer만 사용하여 classifier를 구성하였습니다. 그 덕에 AlexNet, ZFNet, VGG 등에 비해 훨씬 적은 수의 parameter를 갖게 되었습니다.
