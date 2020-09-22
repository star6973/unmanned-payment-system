## DenseNet
- DenseNet의 등장 이전까지의 연구들에서는 네트워크 아키텍처를 보면 큰 진전이 없었다. 2014년까지의 네트워크 구조는 거의 아래와 같은 구조를 따르고 있다.
<center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/160.png" width="70%"></center><br>

- 그나마 GoogLeNet의 Inception Module이 참신한 구조의 변화를 보여주었지만, 이 역시나 Convolution의 큰 틀에서 벗어나지 못했다. 오히려 구조가 VGGNet 보다 더 복잡하다는 이유로 활용성 측면에서 떨어지는 결과를 낳았다. 그러다 2015년에 딥러닝 구조의 가장 큰 문제점이었던 degradation 문제를 해결했다고 주장하는 ResNet이 등장하였다.
    + degradation problem
        - 네트워크의 깊이를 깊게할 수록 vanishing gradient 문제가 발생하면서 오히려 성능이 떨어지는 문제
        - 이때 degradation은 overfitting에 의해서 생겨나는 것이 아니라, 더 많은 layer를 넣을수록 training error가 더 높아진다.

- 하지만 ResNet 이후로 참신한 네트워크 구조에 대한 아이디어는 나오지 않았다. 대부분의 연구들은 전부 기술적인 부분에만 관련있고, 네트워크 구조에 획기적인 변화를 주는 경우가 없었다. 즉, 자체적인 변화보다는 ResNet을 큰 뼈대로 유지하고 약간의 트릭을 추가하는 연구들이 많이 진행되었다.

- 그러다 2017년 CVPR 컨퍼런스에 Densely Connected Network라는 네트워크 구조에 획기적인 변화를 주어 연구 결과가 발표되었다. 일반적인 네트워크의 구조는 위에 있는 수식과 같이, convolution - activation - pooling의 순차적인 조합이다. ResNet은 이러한 네트워크에 skip connection을 추가해서 degradation problem을 해소했다. 이에 더해 DenseNet은 ResNet의 skip connection과 다른 Dense connectivity를 제안했다.

<center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/161.png" width="70%"></center><br>

- ResNet의 skip connection은 x_i = H(x_(i-1)) + x_(i-1)와 같은 수식을 가지면서 입력값을 출력값에 더해주는 것이지만, DenseNet의 dense connectivity는 x_i = H([x_0, x_1, x_2, ..., x_(i-1)])과 같은 수식을 가지면서 입력값을 계속해서 출력값의 채널 방향으로 합쳐주는 것이다.
    + ResNet의 경우에는 입력이 출력에 더해지는 것이기 때문에 종단에 가서는 최초의 정보가 흐려질 수 밖에 없지만, DenseNet의 경우에는 채널 방향으로 그대로 합쳐지는 것이기 때문에 최초의 정보가 비교적 온전히 남아있게 된다.

    + 수식에서 [x_0, x_1, x_2, ..., x_(i-1)]로 표현된 부분은 i번째 layer이전의 feature map을 concatenation한 것으로 보면 되며, H()는 Batch Normalization + ReLU + 3x3 Convolution 으로 표현된 함수라고 할 수 있다.
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/164.png" width="70%"></center><br>

    + 따라서 각 레이어에서 k개의 feature map을 만든다고 하면 i번째 레이어에서는 k_0 + k x (i-1)만큼의 input feature map을 갖게 된다.
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/163.png" width="70%"></center><br>

    + DenseNet에서 제안하는 구조는 위와 같은 그림으로, feature들은 서로 concatenation하면서 결합이 된다. l번째 층은 모든 선행 convolution block의 feature map들로 구성된 l개의 입력을 가지며, 각 feature map은 모든 L-1개의 후속 레이어로 전달된다. 이것은 기존의 아키텍처에서 L개의 connection 대신에 L(L+1)/2개의 connection을 도입하는 것이다. 또한, Dense connectivity pattern에서는 중복되는 feature map을 다시 학습할 필요가 없기 때문에 기존의 네트워크보다 더 적은 파라미터의 수만 필요하다.

- 특징
    1) 모든 feature map들을 차곡차곡 쌓아오기 때문에 레이어 사이에 최대한 가치있는 정보가 사라지지 않고 전달할 수 있도록 할 수 있다는 것이다(vanishing gradient 개선 및 feature propagation 강화).  
        - 네트워크의 깊이가 깊어질수록 처음에 가지고 있던 정보가 사라지는 문제가 발생할 수 있는데, 이러한 문제를 다룬 연구들의 공통점은 전부 초반부 layer를 후반부 layer로 이어주려는 것이었다.

        - DenseNet은 이 문제를 처음 쌓은 층을 네트워크가 깊어져도 차곡차곡 쌓아가는 것으로 해결할 수 있다고 제시한다.
        <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/162.png" width="70%"></center><br>

        - ResNet의 경우도 skip connection으로 어느정도 정보가 사라지는 문제를 해결할 수 있다고 하지만, dense connectivity를 사용하면 아예 초반 레이어의 정보를 쌓아가며 뒤쪽 레이어까지 효율적으로 전달할 수 있다고 한다. 이는 뒤집어 말하면 error를 back propagation할 때도 더 효율적으로 전달한다는 말이 된다.

    2) 기존의 네트워크보다 파라미터의 수를 많이 줄일 수 있다(parameter의 수 절약).  
        - 하나의 레이어당 약 12개 정도의 filter를 가지며, 이전의 feature map들을 계속 쌓아가면서 전체적인 네트워크 내부의 정보들을 효율적으로 가져간다. 이는 마지막 classifier에 네트워크 내부의 전체 feature map을 골고루 입력할 수 있게 만들어주며(dropout 기법이 필요 x) 동시에 전체 파라미터의 개수를 줄여도 네트워크가 충분히 학습이 가능하게 만들어준다.

    3) Regularizing의 효과를 가지고 있어서 작은 데이터셋에서도 overfitting을 줄여준다고 한다(overfitting 감소).  

- 구조
    + Transition Layer
        - convolution network의 가장 큰 특징 중 하나는 pooling을 이용해 feature map의 size를 줄이는 것이다. 따라서 DenseNet은 네트워크 전체를 몇 개의 Dense block으로 나눠서 같은 feature map size를 가지는 레이어들은 같은 dense block으로 묶는다.
        <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/165.png" width="70%"></center><br>

        - 위의 그림에서는 총 3개의 dense block으로 나누어진다. 같은 block내의 레이어들은 전부 같은 feature map size를 가지게 된다. 그리고 네모 박스를 친 pooling과 convolution 부분을 transition layer라고 부른다. 이 layer에서는 Batch Normalization, ReLU, feature map dimension을 조절하기 위한 1x1 convolution layer, 그리고 2x2 average pooling layer가 위치한다. 1x1 convolution을 통해 feature map의 개수를 줄여주며, 이때 줄여주는 정도를 나타내는 hyper parameter를 theta라고 하며, 0.5로 설정하고 있다. 이 과정을 compression이라 표현하고 있으며, 즉 transition layer를 통과하면 feature map의 개수(채널)가 절반으로 줄어들고, 2x2 average pooling layer를 통해 feature map의 가로, 세로 크기 또한 절반으로 줄어든다. 만약 theta를 1로 설정하면 feature map의 개수는 그대로 유지된다.

        - 이와 같이 Transition layer는 feature map의 가로, 세로 사이즈를 줄여주고, feature map의 개수를 줄여주는 역할을 담당하고 있다.

    + Growth Rate
        - 각 feature map끼리 연결이 되는 구조이기에 자칫 feature map의 채널의 개수가 많은 경우, 계속해서 channel-wise로 concat되면서 채널이 많아질 수 있다. 그래서 DenseNet에서는 각 layer의 feature map의 채널의 개수를 굉장히 작은 값을 사용하며, 이때 각 layer의 feature map의 채널 개수를 growth rate(k)라고 부른다.

        - 수식 H_l가 k개의 feature map을 생성한다고 하면, l번째 layer는 k_0 + k x (l-1)개의 feature map이 입력된다고 할 수 있다(등차수열). 여기서 k_0는 dense block의 첫 번째의 input layer의 feature map의 개수이다. 논문에서는 DenseNet과 기존의 네트워크의 다른 점 중 하나로 매우 좁은 레이어를 가지고 있다고 언급한다. 다음은 k=6인 경우 feature map이 어떻게 쌓이는지 간단히 도식화한 그림이다.

        <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/166.png" width="70%"></center><br>

        - 결론적으로, DenseNet과 기존의 네트워크 구조의 중요한 차이점은 very narrow layer(k=12)를 가질 수 있다는 것이다. 이러한 효과는 각 layer들이 block 내의 모든 이전 feature map의 정보를 가지고 있기 때문에, 네트워크의 "collective knowledge"에 접근할 수 있다는 것이다. Feature map을 네트워크의 global state로 볼 수 있으며, 각 layer는 각자의 growth rate를 feature map에 추가하기 때문에 growth rate는 각 global state에 기여하는 새로운 정보의 양을 조절한다고 할 수 있다(input layer의 feature map을 제외하고). 또한, 한 번 사용된 global state는 기존의 네트워크 아키텍처와 달리 concatenate로 연결되어 있기 때문에 네트워크의 어디에서나 액세스할 수 있으며, layer-to-layer로 복제할 필요가 없다. 이러한 이유로, k값이 작아도 학습에는 충분하다고 할 수 있다.

    + Bottleneck Layer
        - ResNet과 Inception 등에서 사용되는 bottleneck layer의 아이디어는 DenseNet에서도 찾아볼 수 있다. 즉, BN -> ReLU -> 3x3 Conv 구조에서 BN -> ReLU -> 1x1 Conv -> BN -> ReLU -> 3x3 Conv의 Bottleneck 구조를 적용한다.
        <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/167.png" width="70%"></center><br>

        <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/168.png" width="70%"></center><br>

        - 3x3 convolution 전에 1x1 convolution을 거쳐서 입력 feature map의 채널의 개수를 줄이는 것까지는 같은데, 그 뒤로 다시 입력 feature map의 채널 개수만큼을 생성하는 대신 growth rate만큼의 feature map을 생성하는 것이 차이점이며, 이를 통해 연산량을 줄일 수 있다고 한다. 또한, DenseNet의 Bottleneck layer는 1x1 convolution 연산을 통해 4 x growth rate개의 feature map을 만들고, 그 뒤에 3x3 convolution을 통해 growth rate개의 feature map으로 줄여주는 것을 볼 수 있다.

    + Composite Function
        - ResNet의 구조에서 activation function의 위치에 따라 성능이 다를 수 있는데, 이러한 실험을 한 논문이 [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf)으로, DenseNet은 BN-ReLU-Conv 순서의 pre-activation의 구조를 사용하였다.

        <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/169.png" width="70%"></center><br>

- 실험
    + 논문에서는 총 3가지 데이터셋(CIFAR, SVHN, ImageNet)을 사용하여 실험했다고 한다.
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/170.png" width="70%"></center><br>

    + 위의 실험 결과 표를 보면, k가 증가하면 파라미터의 수가 많아지고, Bottleneck Compression을 사용하면 파라미터의 수가 줄어드는 것을 확인할 수 있다.

    + 다른 네트워크들에 비해 파라미터의 수가 훨씬 적은 것을 볼 수 있다.

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/171.png" width="70%"></center><br>

    + 표면적으로는 DenseNet은 ResNet과 유사하지만 입력과 출력을 concatenation함으로써, 모든 layer에서 학습된 feature map을 모든 후속 layer에서 액세스할 수 있게 되는 것은 다르다. 위의 그림에서 DenseNet의 모든 변형과 유사 성능의 ResNet의 파라미터 효율성을 비교한 결과를 보여준다. 확실히 ResNet에 비해 parameter efficiency가 좋고, ResNet과 유사한 성능을 달성하는데 필요한 parameter의 개수는 1/3에 불과하다.

- 참고자료

> [DenseNet Tutorial [1] Paper Review & Implementation details](https://hoya012.github.io/blog/DenseNet-Tutorial-1/)

> [DenseNet(Densely connected Convolutional Networks) - 1, 2, 3](https://jayhey.github.io/deep%20learning/2017/10/15/DenseNet_1/)

> [DenseNet](https://velog.io/@dyckjs30/DenseNet)

> [DenseNet](https://poddeeplearning.readthedocs.io/ko/latest/CNN/DensNet/)