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

#### ResNet(2015)
#### - ResNet은 3x3 conv가 반복된다는 점에서 VGG와 유사한 구조를 가지고 있습니다. Layer의 개수에 따라 ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152 등 5가지 버전으로 나타낼 수 있으며, ILSVRC 2015 대회에선 ResNet-152로 1위를 차지하였습니다. Layer 개수를 많이 사용할수록 연산량과 parameter 개수는 커지지만 정확도도 좋아지는 효과를 얻을 수 있습니다.
#### - 2개의 conv layer마다 옆으로 화살표가 빠져나간 뒤 합쳐지는 식으로 그려져 있습니다. 이러한 구조를 Shortcut 이라 부릅니다. 일반적으로 Shortcut으로는 identity shortcut, 즉 input feature map x를 그대로 output에 더해주는 방식을 사용합니다.
#### - Output feature map의 개수가 2배로 커질 때 마다 feature map의 가로, 세로 size는 절반으로 줄여주는 방식을 이용하고 있으며, 이 때는 pooling 대신 stride=2를 갖는 convolution 연산을 이용하는 점이 특징입니다. 이 경우, Shortcut에서도 feature map size를 줄여주어야 하며, 이 때는 identity shortcut 대신 projection shortcut 을 이용합니다. 이러한 shortcut 구조를 통해 vanishing gradient에 강인한 학습을 수행할 수 있게됩니다.
#### - 또한 ResNet-50 이상의 모델에서는 feature map의 개수가 많다 보니 연산량도 많아지게 되는데, Inception module에서 보았던 bottleneck 구조를 차용하여 bottleneck residual block 을 중첩하여 사용하는 점이 특징입니다.
#### -마지막으론 같은 2015년에 제안이 되었고, 지금도 굉장히 자주 사용되는 방식인 Batch Normalization(BN) 을 Residual block에 사용을 하였으며, Conv-BN-ReLU 순으로 배치를 하였습니다.

## 2.

#### Pre act ResNet(2016)
#### - 처음 소개드릴 architecture는 2016년 CVPR에 발표된 “Identity Mappings in Deep Residual Networks” 라는 논문에서 제안한 Pre-Act ResNet 입니다. Pre-Act는 Pre-Activation의 약자로, Residual Unit을 구성하고 있는 Conv-BN-ReLU 연산에서 Activation function인 ReLU를 Conv 연산 앞에 배치한다고 해서 붙여진 이름입니다. ResNet을 제안한 Microsoft Research 멤버가 그대로 유지된 채 작성한 후속 논문이며 ResNet의 성능을 개선하기 위해 여러 실험을 수행한 뒤, 이를 분석하는 식으로 설명을 하고 있습니다.
#### - 기존엔 Conv-BN-ReLU-Conv-BN을 거친 뒤 shortcut과 더해주고 마지막으로 ReLU를 하는 방식이었는데, 총 4가지 변형된 구조를 제안하였고, 그 중 full pre-activation 구조일 때 가장 test error가 낮았고, 전반적인 학습 안정성도 좋아지는 결과를 보인다고 합니다.
#### - Original의 경우 2번째 BN을 거쳐서 feature map이 normalize되어도 shortcut과 더해지면서 다시 unnormalized된 채로 다음 Conv 연산으로 전달되는 반면, 제안한 full pre-activation 구조는 모든 Conv 연산에 normalized input이 전달되기 때문에 좋은 성능이 관찰되는 것이라고 분석하고 있습니다.

#### Inception-v2(2016)
#### - Inception-v2의 핵심 요소는 크게 3가지로 나눌 수 있습니다. 1. Conv Filter Factorization 2. Rethinking Auxiliary Classifier  3.Avoid representational bottleneck
#### - 우선 Inception-v1(GoogLeNet)은 VGG, AlexNet에 비해 parameter수가 굉장히 적지만, 여전히 많은 연산량을 필요로 합니다. Inception-v2에서는 연산의 복잡도를 줄이기 위한 여러 Conv Filter Factorization 방법을 제안하고 있습니다. 
#### - VGG에서 했던 것처럼 5x5 conv를 3x3 conv 2개로 대체하는 방법을 적용합니다. 여기서 나아가 연산량은 줄어들지만 receptive field는 동일한 점을 이용하여 n x n conv를 1 x n + n x 1 conv로 쪼개는 방법을 제안합니다.
#### - 그 다음은 Inception-v1(GoogLeNet)에서 적용했던 auxiliary classifier에 대한 재조명을 하는 부분입니다. 여러 실험과 분석을 통해 auxiliary classifier가 학습 초기에는 수렴성을 개선시키지 않음을 보였고, 학습 후기에 약간의 정확도 향상을 얻을 수 있음을 보였습니다. 또한 기존엔 2개의 auxiliary classifier를 사용하였으나, 실제론 초기 단계(lower)의 auxiliary classifier는 있으나 없으나 큰 차이가 없어서 제거를 하였다고 합니다.
#### - 마지막으론 representational bottleneck을 피하기 위한 효과적인 Grid Size Reduction 방법을 제안하였습니다. representational bottleneck이란 CNN에서 주로 사용되는 pooling으로 인해 feature map의 size가 줄어들면서 정보량이 줄어드는 것을 의미합니다. 이해를 돕기 위해 위의 그림으로 설명을 드리면, 왼쪽 사진과 같이 pooling을 먼저 하면 Representational bottleneck이 발생하고, 오른쪽과 같이 pooling을 뒤에 하면 연산량이 많아집니다.
#### - 기존 Inception-v1은 7x7 conv 연산이 가장 먼저 수행이 되었는데, 위의 Factorization 방법에 의거하여 3x3 conv 연산 3개로 대체가 되었습니다.