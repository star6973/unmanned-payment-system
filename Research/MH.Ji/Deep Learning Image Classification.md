# Image Classification의 종류 정리

## LeNet
- MLP가 가지는 한계점인 input의 픽셀수가 많아지면 parameter가 기하급수적으로 증가하는 문제를 해결할 수 있는 CNN 구조 제시하였다.

- 특징
    + input을 2차원으로 확장하였고, parameter sharing을 통해 input의 픽셀 수가 증가해도 parameter 수가 변하지 않는다는 특징을 가지고 있다.

- 구조
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/99.PNG" width="70%"></center><br>

    + C1 layer에서는 32x32 픽셀의 image를 input으로 사용하고, 6개의 5x5 필터와 convolution 연산을 해준다. 그 결과, 6장의 28x28 feature map을 산출한다.

    + S2 layer에서는 C1 layer에서 출력된 feature map에 대해 2x2 필터를 stride=2로 설정하여 sub sampling을 진행한다. 그 결과, 14x14 feature map으로 축소된다. sub sampling 방법은 average pooling 기법을 사용하였는데, original 논문인 [Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)에 따르면, 평균을 낸 후에 한 개의 trainable weight을 곱해주고, 또 한 개의 trainable bias를 더해준다고 한다. activation function은 sigmoid를 사용한다.

    + C3 layer에서는 6장의 14x14 feature map에 convolution 연산을 수행해서 16장의 10x10 feature map을 산출한다.

    + S4 Layer에서는 16장의 10x10 feature map에 sub sampling을 진행해 16장의 5x5 feature map으로 축소시킨다.

    + C5 layer에서는 16장의 5x5 feature map에 120개의 5x5x16 필터를 convolution 연산을 수행해서 120장의 1x1 feature map을 산출한다.

    + F6 layer는 84개의 유닛을 가진 Feed Forward Neural Network로, C5 layer의 결과를 연결시켜준다.

    + Output layer는 10개의 Euclidean radial Bias Function(RBF) 유닛으로 구성되어 F6 layer의 결과를 받아서, 최종적으로 이미지가 속한 클래스를 분류한다.

- 참고자료
> [Deep Learning Image Classification Guidebook [1] LeNet, AlexNet, ZFNet, VGG, GoogLeNet, ResNet](https://hoya012.github.io/blog/deeplearning-classification-guidebook-1/)

> [LeNet-5의 구조](https://bskyvision.com/418)
<br><br>

## AlexNet
- LeNet-5와 구조가 크게 다르지 않으며, 2개의 GPU로 병렬연산을 수행하기 위해 병렬적인 구조로 설계되었다. 왜냐하면 그 당시에 사용한 GPU인 GTX 580이 3GB의 VRAM을 가지고 있는데, 하나의 GPU로 사용하기엔 메모리가 부족하기 때문이다.

- 특징
    + Multiple GPU Training
    + activation function은 ReLU를 사용하였다. tanh보다 빠르게 수렴하는 효과를 얻을 수 있다고 한다.
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/101.PNG" width="70%"></center><br>
    
    + normalization은 Local Response Normalization(LRN)을 사용하였고, Pooling의 커널 사이즈를 stride보다 크게 하는 Overlapping Pooling을 사용하였다. 이외에도 Dropout, PCA를 이용한 data augmentation 기법을 사용하였다.

        + Dropout
            - overfitting을 막기 위한 regularization 기술의 일종으로, fully connected layer의 뉴런 중 일부를 생략하면서 학습을 진핸한다. 즉, 몇몇의 뉴런의 값들을 0으로 바꾸어 forward 및 backward propagation에 영향을 주지 못한다.

            - dropout은 train에만 적용되고, test에는 모든 뉴런을 사용한다.
        
        + Overlapping Pooling
            - CNN에서 pooling은 feature map의 크기를 줄이는 역할로, 보통 average pooling과 max pooling 2가지를 사용한다. average pooling은 sliding window에 걸려있는 값 중 평균값을 선택하고, max pooling sliding window에 걸려있는 값 중 최대값을 선택한다.

            - pooling시에는 통상적으로 겹치는 값이 없게 하지만, original 논문인 [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)에서는 max pooling을 사용하면서 크기를 3으로, stride를 2로 주어 sliding이 겹치는 부분이 발생한다(overlapping pooling).
            
            - LeNet의 경우 average pooling을 사용하였지만, AlexNet에서는 max pooling을 사용하였다. 또한, AlexNet의 경우 pooling kernel이 움직이는 보폭인 stride를 커널 사이즈보다 작게하는 overlapping pooling을 적용하였다.
            <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/102.PNG" width="70%"></center><br>

            - overlapping pooling을 하면 중첩이 되면서 진행하지만, non-overlapping pooling을 하면 중첩이 되지 않는다. original 논문은 overlapping pooling을 하면 top-1, top-5 error를 줄이는데 효과가 있다고 한다.
            <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/104.PNG" width="70%"></center><br>

        + Local Response Normalization(LRN)
            - 활성화된 뉴런이 주변 이웃 뉴런들을 억누르는 현상을 lateral inhibition 현상이라고 하며, 이를 모델링한 것이 LRN이다. 강하게 활성화된 뉴런의 주변 이웃들에 대해서 normalization을 실행한다. 주변에 비해 어떤 뉴런이 비교적 강하게 활성화되어 있다면, 그 뉴런의 반응은 더욱더 돋보일 것이다. 반면에 강하게 활성화된 뉴런 주변도 모두 강하게 활성화되어 있다면, LRN 이후에는 모두 값이 작아질 것이다.

            - 이러한 LRN은 ReLU 활성화 함수를 처리하는데 유용하다. 만약 ReLU를 통해 활성화된 뉴런이 여러개가 있으면, 정규화를 위해 LRN이 필요하기 때문이다.
            <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/103.PNG" width="70%"></center><br>

            - original 논문에서는 top-1, top-5에서 각각 1.4%, 1.2%의 성능 향상이 발생하였다. 하지만 이 방법은 지금에 와서는 성능상의 이점이 없어 잘 사용하지 않는다.

        + Data Augmentation
            - overfitting을 막기 위한 dropout과 같은 regularization의 방법이다. overffiting을 막기 위한 가장 좋은 방법은 데이터의 양을 늘리는 것인데, 하나의 이미지를 가지고 여러 장의 비슷한 이미지를 만들어내면서 데이터의 양을 늘릴 수 있다. 즉, 같은 내용을 담고 있지만 위치가 살짝 다른 이미지들이 생산된다.
            <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/105.PNG" width="70%"></center><br>

- 구조
    + AlexNet은 8개의 layer로 구성되어 있다. 5개의 convolution layer와 3개의 fully connected layer로 구성되어 있다.
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/99.PNG" width="70%"></center><br>

    + input layer에는 224x224 사이즈의 RGB 컬러 이미지가 입력된다(224x224x3)

    + 첫 번째 레이어에서 convolution layer는 96개의 11x11x3 사이즈의 필터로 convolution 연산을 한다. stride=4로 설정, padding은 사용하지 않아 결과적으로, 55x55x96 feature map이 산출된다. convolution layer는 항상 activation function이 붙어있기 때문에, ReLU를 사용하여 활성화해준다. pooling layer는 3x3 overlapping max pooling을 stride=2로 시행하여 결과적으로, 27x27x96 feature map을 산출한다.

    + 두 번째 레이어에서 convolution layer는 256개의 5x5x48 사이즈의 필터로 convolution 연산을 한다.
    stride=1로 설정, padding=2로 설정하여 결과적으로, 27x27x256 feature map이 산출된다. activation function 역시 ReLU로 활성화한다. pooling layer도 마찬가지로 3x3 overlapping max pooling을 stride=2로 시행하여 결과적으로, 13x13x256 feature map을 산출한다. 첫 번째 레이어와 다르게, local response normalization을 시행하여 feature map이 유지된다.

    + 세 번째 레이어에서 convolution layer는 384개의 3x3x256 사이즈의 필터로 convolution 연산을 한다.
    stride=1로 설정, padding=1로 설정하여 결과적으로, 13x13x384 feature map을 얻게 된다. activation function 역시 ReLU로 활성화한다.

    + 네 번째 레이어에서 convolution layer는 384개의 3x3x192 사이즈의 필터로 convolution 연산을 한다.
    stride=1로 설정, padding=1로 설정하여 결과적으로 13x13x384 feature map을 얻게 된다. activation function 역시 ReLU로 활성화한다.

    + 다섯 번째 레이어는 네 번째 레이어와 동일하게 activation까지 진행하며, pooling layer에서 3x3 overlapping max pooling을 stride=2로 시행하여 결과적으로, 6x6x256 feature map을 산출한다.

    + 여섯 번째 레이어는 fully connected layer로, 6x6x256 feature map을 flatten해주어 6x6x256=9216 차원의 벡터로 변환한다. 이를 여섯 번째 레이어의 4096개의 뉴런과 연결시키고, 그 결과를 ReLU로 활성화해준다.

    + 일곱 번째 레이어도 fully connected layer로, 4096개의 뉴런을 가지고 있으며, 여섯 번째 레이어와 동일하게 진행한다.

    + 여덟 번째 레이어도 fully connected layer로, 1000개의 뉴런을 가지고 있으며, 일곱 번째 레이어와 연결하여 softmax function을 적용해 확률을 구한다.

- 참고자료
> [Deep Learning Image Classification Guidebook [1] LeNet, AlexNet, ZFNet, VGG, GoogLeNet, ResNet](https://hoya012.github.io/blog/deeplearning-classification-guidebook-1/)

> [LeNet-5의 구조](https://bskyvision.com/421)

> [AlexNet 논문 요약 정리](https://s3nsitive.tistory.com/entry/AlexNet-%EB%85%BC%EB%AC%B8-%EC%9A%94%EC%95%BD-%EC%A0%95%EB%A6%AC)
<br><br>

## VGGNet
- 2014년 ILSVRC에서 GoogLeNet과 함께 큰 주목을 받으며, 준우승을 차지한 모델이다. 준우승이었지만, 구조적인 측면에서 GoogLeNet보다 훨씬 간단한 구조로 되어 있으며, 향후 모델의 발전 역사에서 VGGNet 모델부터 시작해서 네트워크의 깊이가 확 깊어졌다.

- Depth
    + VGGNet이 발표되기 전까지만해도 AlexNet이나 ZFNet과 같은 대부분의 모델은 8 layer의 수준이었다. 하지만 GoogLeNet과 VGGNet 모두 이전 구조에 비해 훨씬 deep해진다.
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/106.PNG" width="70%"></center><br>

    + VGGNet의 연구팀은 origin 논문 [Very deep convolutional networks for large-scale image recognition](https://mohitjainweb.files.wordpress.com/2018/06/very-deep-convolutional-networks-for-large-scale-image-recognition.pdf)에서 밝히듯이 원래는 depth가 딥러닝 모델에 어떤 영향을 주는지 연구를 하기 위해 VGGNet을 개발한 것 같다.

    + 따라서 깊이의 영향만을 최대한 확인하고자, convolution 필터 커널의 사이즈는 가장 작은 3x3으로 고정했다. 만약 필터 사이즈의 크기가 크게 되면, 그만큼 이미지의 사이즈가 축소되기 때문에 네트워크의 깊이를 충분히 깊게 만들기 어렵기 때문이다.

    + 총 6개의 구조(A, A-LRN, B, C, D, E)를 만들어 성능을 비교했다. 여러 구조를 만든 이유는 기본적으로 깊이의 따른 성능 변화를 비교하기 위함이다.
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/107.PNG" width="70%"></center><br>

    + 진행했던 연구를 보면, AlexNet에서 사용되던 LRN은 A와 A-LRN 구조의 성능을 비교해보면서 성능 향상에 효과가 없다는 것을 밝혔다. 그래서 더 깊은 구조인 B, C, D, E에서는 LRN을 적용하지 않는다. 

- Filter
    + VGGNet의 구조를 파악하기 전에 먼저 집고 넘어가야 하는 것이 바로 3x3 필터를 사용하는 것이다. 3x3 필터를 2번 convolution 연산을 하는 것과 5x5 필터를 1번 convolution 연산을 하는 것이 결과적으로 동일한 사이즈의 feature map을 산출한다는 것이 핵심이다. 만약 3x3 필터로 3번 convolution 연산을 한다면, 7x7 필터로 1번 convolution 연산을 하는 것과 대응된다.
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/108.PNG" width="70%"></center><br>

    + 그렇다면 3x3 필터로 3번 convolution하는 것이 7x7 필터로 1번 convolution 연산을 하는 것보다 나은 점은 무엇인가? 바로, 가중치 또는 파라미터의 수의 차이다. 3x3 필터가 3개면 총 27개의 가중치를 가지는 반면, 7x7 필터는 1개만이어도 총 49개의 가중치를 가진다. CNN에서 가중치는 training에서 사용되는 파라미터이기 때문에, 학습 속도에 영향을 준다. 즉, 사용되는 파라미터의 개수가 적어질수록 학습 속도가 빨라진다는 것이다. 동시에 layer의 개수 역시 늘어나지면서 feature에 비선형성을 증가시켜주어 feature가 점점 더 유용해진다.
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/109.PNG" width="70%"></center><br>

    + 위에서 나오는 feature에 비선형성을 증가시켜주는 것이 좋다는 이유를 알아보자. 만약 Neural Network에서 activation function을 linear function을 사용하면 신경망의 depth를 깊게 하는 의미가 사라진다. activation function이 h(x)=cx인 선형함수라고 가정할 때, 3층 네트워크가 되면 y(x)=h(h(h(x)))가 되는데 이는 y(x)=c*c*c*x가 된다. 이는 y(x)=ax라는 linear한 구조이기 때문에, 여러 층을 구성하는 이점을 살릴 수 없다.

    + B 구조는 3x3 convolution layer를 2번 사용하는 경우와 5x5 convolution layer를 1번 사용하는 모델을 만들어 실험을 했는데, 결과는 3x3 필터를 2번 사용하는 경우가 5x5 필터를 1번 사용하는 것보다 top-1 error에서 7% 성능을 높일 수 있었다.

- 특징
    + LRN을 적용하지 않는다.
    + 학습 parameter의 수를 적게 만들고, depth의 깊이를 늘리기 위해 3x3 필터를 사용한다.

- 구조
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/110.PNG" width="70%"></center><br>

    + input layer에서는 224x224 사이즈의 컬러 이미지를 입력받는다(224x224x3)

    + 첫 번째 레이어에서 convolution layer에서 64개의 3x3x3 필터로 입력 이미지를 convolution 연산을 해준다. stride=1, zero padding=1로 설정해준다. 이 두 개의 하이퍼파라미터는 다음 모든 convolution layer에서 동일하게 적용한다. 결과적으로 64장의 224x224x64의 feature map을 산출한다. activation function은 ReLU를 사용하였고, 마찬가지로 output layer를 제외하고 모든 convolution layer에서 동일하게 적용한다.

    + 두 번째 레이어에서 convolution layer에서 64개의 3x3x64 필터로 convolution 연산을 해주어, 결과적으로 224x224x64의 feature map이 산출된다. pooling layer에서는 2x2 필터로 stride=2로 적용하여 max pooling을 하여, 결과적으로 feature map의 사이즈를 112x112x64로 줄인다.

    + 세 번째 레이어에서부터는 AlexNet과 비슷한 맥락으로 진행한다. 자세하게 살펴보려면 [VGGNet의 구조 (VGG16)](https://bskyvision.com/504)로 GoGo..

- VGGNet은 AlexNet이나 ZFNet처럼 224x224 크기의 컬러 이미지를 입력으로 받아들이고, 1개 혹은 그 이상의 convolutional layer 뒤에 max pooling layer가 오는 단순한 구조가 되어 있다. 또한, 맨 마지막 단에는 fully connected layer가 온다. 이러한 구조 덕분에 간단하고 이해나 변형이 쉬운 장점을 가지고 있지만, 파라미터의 수가 엄청나게 많기 때문에 학습 시간이 오래 걸린다.

- 참고자료
> [Deep Learning Image Classification Guidebook [1] LeNet, AlexNet, ZFNet, VGG, GoogLeNet, ResNet](https://hoya012.github.io/blog/deeplearning-classification-guidebook-1/)

> [VGGNet의 구조 (VGG16)](https://bskyvision.com/504)

> [VGGNet [1], [2] - 라온피플 머신러닝 아카데미](http://blog.naver.com/laonple/220738560542)
<br><br>

## GoogleNet
- 2014년 ILSVC 대회에서 1위를 한 모델이며, Google 연구팀이 개발하였다. 19층인 VGG19보다 좀 더 깊은 22층으로 구성되어 있다.

- 일반적인 CNN 구조는 feature extraction 부분(convolution layer + pooling layer)과 classifier 부분(fully connected neural network)으로 구성이 도니다. convolutional layer와 pooling layer를 번갈아 사용하는 layer를 여러 개 사용하여 feature를 추출하고, 최종 feature map을 classifier 역할을 하는 fully connected neural network를 이용하여 처리한다.

- GoogLeNet의 구조를 알기 전에 NIN(Network in Network)의 구조와 설계 철학을 알고 들어가자. 왜냐하면 GoogLeNet을 개발한 구글은 자신들의 구조를 설계함에 있어 크게 2개의 논문을 참조하고 있으며, 그 중 Inception Module 및 전체 구조에 관련된 부분은 NIN 구조를 발전시킨 것이기 때문이다.

- NIN 설계자는 CNN의 convolutional layer가 local receptive field(지역적인 수용 영역, mask filter)에서 feature를 추출하는 능력은 우수하지만, filter의 특징이 linear하기 때문에 non-linear한 성질을 갖는 feature를 추출하기엔 어려움이 있으므로, 이 부분을 극복하기 위해 feature map의 개수를 늘려야하는 문제에 주목했다. 하지만, 필터의 개수를 늘리게 되면 연산량이 늘어나는 문제가 있다. 따라서 NIN 설계자는 local receptive field 안에서 좀 더 feature를 잘 추출할 수 있는 방법을 연구하였으며, 이를 위해 micro neural network를 설계하였다. 이들은 convolution을 수행하기 위한 field 대신에 MLP를 사용하여 특징을 추출하도록 하였다.

<center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/119.png" width="70%"></center><br>

- MLP를 사용하였을 때의 장점은 convolution kernel보다는 non-linear한 성질을 잘 활용할 수 있기 때문에 feature를 추출할 수 있는 능력이 우수하다는 점이다. 또한, 1x1 convolution을 사용하여 feature map을 줄일 수 있도록 하였으며, 이 기술을 Inception Module에 적용하였다.

- NIN 구조가 기존 CNN과 또 다른 점은 CNN의 최종단에서 흔히 보이는 fully connected neural network가 없다는 점이다. 대신에 최종단에 global average pooling을 사용하였다. 이는 앞에 효과적으로 feature map을 추출하였기 때문에, 이렇게 추출된 map에 대한 pooling만으로도 충분하다고 주장하고 있다. 즉, average pooling만으로 classifier 역할을 할 수 있기 때문에 overfitting의 문제를 회피할 수 있고, 연산량이 대폭 줄어드는 이점도 얻을 수 있다.

- CNN의 최종단에 있는 fully connected NN은 전체 free parameter 중 90% 수준에 육박하기 때문에, 많은 파라미터의 수로 인해 overfitting에 빠질 가능성이 아주 높으며, 이를 피하기 위해선 Dropout 기법을 적용해야 한다. 하지만 NIN의 경우는 average pooling만으로 classification을 수행할 수 있기 때문에 overfitting에 빠질 가능성이 현저하게 줄어들게 된다.

- 이러한 NIN 구조는 GoogLeNet 모델의 '네트워크 안의 네트워크'라는 구조에 영향을 많이 끼친 구조라고 할 수 있다.

- 특징
    1) 1x1 Convolution
        <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/112.PNG" width="70%"></center><br>

        - 위의 구조도를 보면 곳곳에 1x1 필터의 convolution 연산이 있음을 확인할 수 있다.

        - 1x1 convolution은 feature map의 개수를 줄이는 목적으로 사용된다. feature map을 줄일수록 연산량이 줄어들 수 있기 때문이다. 

        - 예를 들어, 480장의 14x14의 feature map을 48장의 14x14로 줄여보도록 하자. 첫 번째 방법은 1x1 convolution을 사용하지 않고, 5x5 필터로 convolution 연산을 해주자. 이때 필요한 연산횟수는 (14x14x48) x (5x5x480) = 112.9M이 된다. 두 번째 방법은 16개의 1x1x480의 필터로 convolution 연산을 해보자. 결과적으로 16장의 14x14의 feature map이 산출된다. 다시 이 14x14x16 feature map을 48개의 5x5x16의 필터로 convolution 연산을 해주면 48장의 14x14 feature map으로 줄어들 수 있다. 그럼 이때 필요한 연산횟수는 (14x14x16) x (1x1x480) + (14x14x48) x (5x5x16) = 5.3M이다. 첫 번째 방법에 비해 훨씬 많은 연산량을 줄일 수 있다.
    
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/113.PNG" width="70%"></center><br>

    2) Inception Module  
        <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/114.PNG" width="70%"></center><br>

        - GoogLeNet은 총 9개의 inception module을 포함하고 있다. 기존에는 layer간에 1 convolution + 1 pooling 연산으로 연결하였다면, inception module은 총 4가지 서로 다른 연산을 거친 뒤 feature map을 channel 방향으로 합치는 concatenation을 이용하고 있다는 점이다.

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/115.PNG" width="70%"></center><br>

        - 위의 구조에서 노란색 블럭으로 표현된 1x1 convolution을 제외하고, 3x3, 5x5 convolution 연산을 섞어서 사용하는 방식을 Naive Inception Module이라고 부른다. 또한, 여기에 추가로 3x3 convolution와 5x5 convolution 연산이 많은 연산량을 차지하고 있기 때문에, 두 convolution 연산 앞에는 1x1 필터를 적용한 convolution 연산을 추가하여 feature map의 개수를 줄이고, 다시 거꾸로 3x3, 5x5 convolution 연산을 수행하여 feature map을 키워주는 bottleneck 구조를 추가하였다.

        - AlexNet, VGGNet 등의 이전 CNN 모델들은 하나의 layer에 동일한 사이즈의 필터를 이용해서 convolution 연산을 해줬던 것과는 차이가 있다. 따라서 Inception module 덕분에 다양한 종류의 특성이 도출될 뿐만 아니라, 연산량이 절반 이상을 줄일 수 있었다.

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/116.PNG" width="70%"></center><br>
    
    3) Global Average Pooling  
        - AlexNet, VGGNet 등에서는 FC layer들이 망의 후반부에 연결되어 있다. 하지만 GoogLeNet FC 방식 대신에 global average pulling이란 방식을 사용한다. global average pooling은 전 층에서 산출된 feature map들을 각각 평균낸 것을 이어서 1차원 벡터를 만들어주는 것이다. FC layer와 같이 1차원 벡터로 만들어주어야 최종적으로 이미지 분류를 위한 softmax 함수로 연결할 수 있기 때문이다.

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/117.png" width="70%"></center><br>

        - 이 방식을 통해 가중치의 개수를 상당히 많이 없애줄 수 있다. 만약 FC 방식을 사용한다면 훈련이 필요한 가중치의 개수가 7x7x1024x1024=51.3M이지만, global average pooling을 사용하면 가중치가 단 한 개도 필요하지 않다. 이러한 방식 덕분에 AlexNet, ZFNet, VGGNet 등에 비해 훨씬 적은 수의 파라미터를 가질 수 있다.

    4) Auxiliary Classifier  
        - 네트워크의 깊이가 깊어질수록 Back Propagation 과정에서 가중치를 업데이트하는데 사용되는 gradient가 점점 작아져서 0이 되어버리는 Vanishing Gradient 문제가 발생하며, 이로 인해 학습 속도가 아주 느려지거나 overfitting 문제가 발생한다. gradient 값들이 0 근처로 가게 되면, 학습 속도가 느려지거나 파라미터의 변화가 별로 없어 학습 결과가 나빠지는 현상이 발생하기 때문이다.
        
        - 최근의 DNN에서는 activation function을 주로 ReLU를 사용하는데, sigmoid나 cross-entropy를 사용할 때보다 많은 이점이 있기 때문이다. 하지만 여러 layer를 거치면서 작은값들이 계속 곱해지다 보면, 0 근처로 수렴되면서 역시 vanishing gradient 문제에 빠질 수 있고, 망이 깊어질수록 이 가능성이 커진다. 따라서 네트워크 내의 가중치들이 제대로 훈련되지 못하는데, 이 문제를 극복하기 위해서 GoogLeNet은 네트워크 중간에 두 개의 보조 분류기(Auxiliary Classifier)를 달아주었다.

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/121.png" width="70%"></center><br>

        - 위의 그림에서 왼쪽은 보조 분류기가 없는 경우에 iteration이 커질수록 grdient가 현저하게 작아지는 것을 보여주며, 오른쪽은 보조 분류기가 없는 경우는 파란색 점선, 보조 분류기가 있는 경우는 빨간색 실선으로 표시된다. 보다시피 빨간색 실선이 gradient 값이 다시 증가하게 되면서, 보다 더 안정적인 학습 결과를 얻을 수 있다.

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/118.png" width="70%"></center><br>

        - 이 보조 분류기들로 구한 loss는 보조적인 역할을 맡고 있으므로, 기존 가장 뒷 부분에 존재하던 classifier보단 전체적으로 적은 영향을 주기 위해 0.3을 곱해 total loss에 더하는 식으로 사용했다고 한다.

        - 학습 단계에서만 사용이 되고 테스팅 단계에선 사용이 되지 않았는데, 그 이유는 테스팅시에 성능 향상이 미미하기 때문이다.

- 구조
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/111.PNG" width="70%"></center><br>

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/120.png" width="70%"></center><br>

    + patch size / stride
        - kernel의 크기와 stride의 간격을 말한다. 최초의 convolution에 있는 7x7/2의 의미는 receptive field의 크기가 7x7인 filter를 2픽셀 간격으로 적용한다는 뜻이다.

    + output size
        - 산출되는 feature map의 크기 및 개수를 나타낸다. 112x112x64의 의미는 224x224 크기의 이미지에 2픽셀 간격으로 7x7 필터를 적용하여 총 64개의 feature map이 얻어졌다는 뜻이다.

    + depth
        - 연속적인 convolution layer의 개수를 의미한다. 첫 번째 convolution layer는 depth가 1이고, 두 번째와 inception이 적용되어 있는 부분은 모두 2로 되어 있는 이유는 2개의 convolution을 연속적으로 적용하기 때문이다.
    
    + 1x1
        - 1x1 convolution을 의미하며, 그 행에 있는 숫자는 1x1 convolution을 수행한 뒤 얻어지는 feature map의 개수를 말한다.

    + 3x2 reduce
        - 3x3 convolution 앞쪽에 있는 1x1 convolution을 의미하며 마찬가지로 inception (3a)에 있는 수를 보면 96이 있는데, 이것은 3x3 convolution을 수행하기 전에 192차원을 96차원으로 줄인 것을 의미한다.

    + 3x3
        - 1x1 convolution에 의해 차원이 줄어든 feature map에 3x3 convolution을 적용한다. inception (3a)에 있는 숫자 128은 최종적으로 1x1 convolution과 3x3 convolution을 연속적으로 적용하여 128개의 feature map을 얻었다는 뜻이다.

    + 5x5 reduce
        - 3x3 reduce와 동일한 방식이다.
    
    + 5x5
        - 3x3과 동일한 방식이다.

    + pool / proj
        - max pooling과 max pooling 뒤에 오는 1x1 convolution을 적용한 것을 의미한다. inception (3a) 열에 있는 숫자 159K는 총 256개의 feature map을 만들기 위해 159K의 free parameter가 적용되었다는 뜻이다.

    + Ops
        - 연산의 수를 나타낸다. 연산의 수는 feature map의 수와 입출력 feature map의 크기에 비례한다. inception (3a)의 단계에서 총 128M의 연산을 수행한다.

- 중요한 점
    + 효과적으로 grid의 크기를 줄이는 방법
        1) 단순히 pooling layer를 적용하는 것보다, inception module과 같이 나란히 적용하는 것이 효과적이다.  
        2) 사이즈가 큰 kernel 대신에 convolution kernel에 대한 인수분해 방식을 적용하고, 앞뒤 일부 구조 및 feature map의 개수를 조정하는 것만으로도 같은 사이즈의 결과를 가져오고, 성능도 상당히 개선시킬 수 있다.

- 참고자료
> [Deep Learning Image Classification Guidebook [1] LeNet, AlexNet, ZFNet, VGG, GoogLeNet, ResNet](https://hoya012.github.io/blog/deeplearning-classification-guidebook-1/)

> [GoogLeNet의 구조](https://bskyvision.com/504)

> [GoogLeNet [1], [2], [3], [4], [5] - 라온피플 머신러닝 아카데미](https://m.blog.naver.com/laonple/220686328027)


## ResNet


