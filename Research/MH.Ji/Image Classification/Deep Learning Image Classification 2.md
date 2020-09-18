# Image Classification의 종류 정리 2

## SqueezeNet
- 딥러닝 모델에서 동일한 성능을 가지는 모델이 파라미터의 수가 더 적다면..?
    + More efficient distributed training
        - 병렬 학습시에 더 큰 효율을 발생시킬 수 있다.
    
    + Less overhead when exporting new models to clients
        - 자율주행과 같이 실시간으로 서버와 소통해야 하는 시스템의 경우, 매우 좋다. 데이터 전송 자체가 크지도 않기 때문에 서버 과부하도 적게 걸리고, 업데이트도 자주 할 수 있게 된다.

    + Feasible FPGA and embedded deployment
        - FPGA(일종의 반도체 소자)는 보통 10MB 이하의 휘발성 메모리를 가지고 있다. 작은 모델은 직접적으로 FPGA에 모델을 심을 수 있으며, 이는 다른 기관을 통해 inference할 경우 생기는 bottleneck 현상이 없어진다. 또한 ASIC(Application-Specific Integrated Circuits)에 직접적으로 CNN을 배치할 수 있다.

- SqueezeNet의 연구팀은 위와 같은 효과를 발생시킬 수 있도록, AlexNet의 파라미터를 50배 이상 줄여서 0.5MB 이하의 model size를 가질 수 있는 architecture 구조를 제안하고 있다.

- original 논문[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://openreview.net/pdf?id=S1xh5sYgx)에서는 총 3가지 종류의 SqueezeNet의 구조를 제안하고 있으며, 모든 architecture는 Fire module이라는 block으로 구성된다. ResNet을 구성하고 있는 residual block과 같은 개념의 block이라고 생각하면 된다.

- Design Strategies
    1) Replace 3x3 filters with 1x1 filters  
        + 모든 3x3 convolution filter를 1x1 필터로 교체한다. 이는 1x1 필터가 3x3 필터에 비해 9배나 더 적은 파라미터를 가지고 있기 때문이다.

    2) Decrease the number of input channels to 3x3 filters  
        + 만약 모델의 layer 하나가 전부 3x3 필터로 구성되어 있다면, 파라미터의 총 수는 (input channel) x (number of filters) x (3x3) 개와 같다. 따라서 3x3 필터 자체의 수를 줄이고 이에 더해 3x3으로 들어가는 input channel의 수도 줄여야 한다. 본 논문에서는 squeeze layer를 사용하여 input channel을 3x3 filter 수만큼 줄인다.
    
    3) Downsample late in the networks so that convolution layers have large activation  
        + downsampling part를 네트워크 후반부에 집중시키는 방법을 사용한다. 보통 downsample은 max pooling 또는 average pooling 또는 필터 자체의 stride를 높이는 방식으로 이미지의 spatial resolution을 줄이게 된다. 이렇게 줄여서 한번에 필터가 볼 수 있는 영역을 좁히면서 해당 이미지의 정보를 압축시키는 것이다. 논문의 저자들은 모든 조건이 동등하다는 가정하에 큰 activation map을 가지고 있을수록 성능이 더 높다는 것에서 영감을 얻었다. 따라서 SqueezeNet의 네트워크 후반부에 downsample을 넣는 방법을 취한다.

- Fire Module
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/128.png" width="70%"></center><br>

    + Fire module은 총 두 가지의 layer로 이루어져있다. 첫 번째 layer는 1x1 convolution filter로 이루어져있고, "Sqeeze Layer"라고 한다. 두 번째 layer는 1x1과 3x3 convolution filter로 이루어져있고, "Expand Layer"라고 한다. 첫 번째 layer인 1x1 convolution에서는 filter의 개수를 줄이고(squeeze), 두 번째 layer인 1x1 convolution과 3x3 convolution을 통해 filter의 개수를 늘려주는(expand) 연산을 수행한다. Activation function은 ReLU를 사용하며, 3개의 convolution layer의 filter의 개수는 hyper parameter이다.

    + 총 3개의 hyper parameter는 s1x1, e1x1, e3x3가 있다. 먼저, s1x1는 squeeze layer에서 1x1 filter의 총 개수이다. e1x1은 expand layer에서의 1x1 filter의 총 개수이며, e3x3는 expand layer에서의 3x3 filter의 총 개수이다. Fire module을 만들 때는 s1x1의 값을 e1x1 + e3x3보다 더 작게 설정해준다. 이는 design stratigies의 두 번째 전략처럼 3x3 필터로 들어가는 input channel의 수를 제한할 수 있게 한다. 즉, 다음 그림과 같이 input으로 128개의 채널이 들어오면, 1x1 convolution 연산을 통해 16개의 채널로 줄였다가, 다시 1x1 convolution 연산을 통해 64개, 3x3 convolution 연산을 통해 64개를 만들고, 이것을 통해 다시 128개의 채널 output을 만든다.

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/132.png" width="70%"></center><br>

    + 이러한 방식으로 weight size는 획기적으로 줄이면서, accuracy는 AlexNet과 동급 혹은 그 이상인 모델을 설계할 수 있었다.

- 구조
    + SqueezeNet의 이론의 흐름
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/131.png" width="70%"></center><br>

    + SqeezeNet의 전체적인 구조는 다음 그림과 같다. 모든 SqeezeNet은 전부 1개의 convolution filter를 거치고 나서 max pooling이 이어진다. 그 이후에 8개의 fire module로 이루어져 있고, 마지막에 convolution filter를 거치고 GAP(Global Average Pooling)으로 마무리가 된다. Pooling layer를 conv1, fire4, fire8, conv10 이후에 배치하며 design stratigies의 세 번째 전략을 취했다고 볼 수 있다. 

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/129.png" width="70%"></center><br>
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/130.png" width="70%"></center><br>

    + NIN, GoogLeNet 등에서 사용했던 것처럼 FC layer 대신 GAP를 이용하고, 실험에서는 추가적으로 pruning 기법과 compression 기법 등을 같이 적용하여 최종적으로 AlexNet 대비 ImageNet Accuracy는 비슷하거나 약간 더 높은 수치를 얻었다. 또한, Model size는 50배에서 510배까지 줄일 수 있음을 보였다. 또한, pruning, compression 등의 모델 경량화 기법들을 많이 사용하며, architecture 관점에서도 연산량을 줄이기 위한 시도를 보여주었다.

- 실험
    
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/133.png" width="70%"></center><br>
    
    + AlexNet과 비교를 통해 단순 SqueezeNet만을 사용해도 50배 가까이 모델 사이즈가 줄어들었으며, 기존 AlexNet의 top-1, top-5 accuracy에 근접하거나 뛰어넘는 모습을 보여주고 있다. 또한, uncompressed된 32bit의 데이터 타입을 사용한 SqueezeNet과 deep compression을 적용한 8bit, 6bit 짜리 데이터 타입을 사용한 결과, 510배까지 줄어들었으며 성능도 큰 차이가 나지 않았다. 즉, SqueezeNet 또한 모델 압축에 굉장히 유연하다는 뜻이다.

- 중요한 점
    + 모델의 정확도를 올리는 것에 초점을 두지 않고 CNN의 구조가 모델의 크기와 정확도에 어떤 영향을 끼치는지 알아보기 위해, microarchitecture exploration(모델 세부 구조 탐색)과 macroarchitecture exploration(모델 전체 구조 탐색)에 대해 알아보자.

    + CNN Microarchitecture Metaparameters
        - Fire module은 hyper parameter가 3개로 구성되며, SqueezeNet은 총 8개의 Fire module로 구성되어 있기에 총 24개의 hyper parameter를 가지고 있다. 본 논문에서는 24개의 파라미터를 전부 통제하는 파라미터를 metaparameter라고 지명하였다.

        - SqueezeNet의 전체적인 파라미터 수식은 다음과 같이 설정하고, 값을 바꾸면서 성능을 확인한다. 다음 그림에서 SR(Squeeze Ratio)는 squeeze layer에서 expand layer로 들어가는 input channel의 수를 줄여주는 역할을 한다.
        <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/134.png" width="70%"></center><br>
        <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/135.png" width="70%"></center><br>

    + CNN Macroarchitecture Design Space Exploration
        - 모델의 세부 부분들에 대한 최적화가 끝나고, 전체 구조 탐색에 대해 다음과 같이 총 3가지 모델에 대한 실험을 진행한다.
        <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/136.png" width="70%"></center><br>

            1) Vanilla SqueezeNet  
            2) SqueezeNet with simple bypass connections between some Fire modules  
            3) SqueezeNet with complex bypass connections between the remainig Fire modules  

        - bypass connection은 ResNet에서 쓰이는 skip connection과 같은 개념이다. 그림의 가운데를 보면, fire module을 1개 이상 건너뛰지 않고, bypass connection이 연결되어 있다. 이는 fire module의 input과 output 모두 같은 수의 채널을 가지고 있어야되기 때문이다. 이러한 한계점으로 논문의 저자들은 complex bypass connection이라는 개념을 추가한다. 단순히 1x1짜리 convolution을 거치면서 채널의 수를 맞춰주는 것이다. 이렇게하면 각 fire module의 output 채널의 수가 달라도 숫자를 맞춰서 element-wise addition(요소별 연산 덧셈, 두 벡터와 행렬에서 같은 위치에 있는 원소끼리 덧셈)을 해줄 수 있다.

    + Fire module의 구조에서 squeeze layer가 expand layer보다 필터 수가 더 적은데, 이는 중요한 정보가 bottleneck에서 사라질 수 있는 문제가 있다. 하지만 bypass connection을 추가하면 중요한 정보도 손실없이 쉽게 흘러갈 수 있다.

    + 3가지 종류의 SqueezeNet의 정확도의 성능은 Simple Bypass SqueezeNet > Complex Bypass SqueezeNet > SqueezeNet 순으로 정확도가 높았다. 
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/137.png" width="70%"></center><br>

- 참고자료

> [Deep Learning Image Classification Guidebook [3] SqueezeNet, Xception, MobileNet, ResNext, PolyNet, PyramidNet, Residual Attention Network, DenseNet, Dual Path Network (DPN)](https://hoya012.github.io/blog/deeplearning-classification-guidebook-3/)

> [SQEEZENET(모델 압축)](https://jayhey.github.io/deep%20learning/2018/05/26/SqueezeNet/)

> [[Keras] SqueezeNet Model (CNN) 이란? - 1 (이론편)](https://underflow101.tistory.com/27?category=826164)

<br><br><br>

## Xception
- GoogLeNet의 Inception 구조에 대한 고찰로 연구를 시작하였으며, 추후 많은 연구들에서 사용이 되는 연산인 "depthwise-separable convolution"을 제안하고 있다. Inception-v1, 즉, GoogLeNet에서는 여러 갈래로 연산을 쪼갠 뒤 합치는 방식을 이용함으로써 cross channel correlation과 spatial correlation을 적절히 분리할 수 있다고 주장하고 있다. 쉽게 설명하자면, 채널간의 상관관계와 image의 지역적인 상관관계를 분리해서 학습하도록 가이드를 주는 Inception module을 제안한 것이다.

    + 기존의 convolution layer는 2개의 spatial dimension(width, height)과 channel dimension으로 이루어진 3D 공간에 대한 filter를 학습하려고 시도한 것이다. 따라서 single convolution kernel은 cross-channel correlation과 spatial correlation을 동시에 mapping하는 작업을 수행한다고 할 수 있다.

    + GoogLeNet의 Inception module의 기본 아이디어는 이러한 cross-channel correlation과 spatial correlation을 독립적으로 볼 수 있도록 일련의 작업을 명시적으로 분리함으로써, 이 프로세스를 보다 쉽고 효율적으로 만드는 것이다.

    + 일반적인 Inception module의 경우, 우선 1x1 convolution으로 cross-correlation을 보고, input보다 작은 3~4개의 spatial 공간에 mapping한다. 그다음 보다 작아진 3D 공간에 3x3 혹은 5x5 convolution을 수행하여 spatial correlation을 mapping한다.
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/144.png" width="70%"></center><br>

    + Inception module은 기존의 위와 같은 구조를 단순화시키고, cross-channel correlation과 spatial correlatino이 함께 mapping이 되지 않도록 분리하는 형태로 구조를 변화시켰다.

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/145.png" width="70%"></center><br>
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/146.png" width="70%"></center><br>
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/147.png" width="70%"></center><br>

    + 위의 마지막 그림은 extreme version의 Inception moduel로, 먼저 1x1 convolution으로 cross-channel correlation을 mapping하고, 모든 output channel들의 spatial correlation들의 spatial correlation을 따로 mapping한다. 

- Xception은 Inception module이 지향하고자 한, 채널간의 상관관계와 image의 지역적인 상관관계를 완벽하게 분리하는 더 높은 목표를 세우고 연구를 시작하였고, 그것이 바로 depthwise separable convolution이다. 위의 그림에 나오는 extreme version의 module은 depthwise separable convolution과 거의 동일하다고 할 수 있지만, 2가지의 차이점이 있다.
    1) Operation의 순서  
       : Inception에서는 1x1 convolution을 먼저 수행하는 반면, Tensorflow와 같이 일반적으로 구현된 depthwise separable convolution은 channel-wise spatial convolution을 먼저 수행한 뒤(depthwise convolution)에 1x1 convolution을 수행(pointwise convolution)한다.

    2) 첫 번째 operation 뒤의 non-linearity 여부
       : Inception에서는 두 operation 모두 non-linearity로 ReLU가 뒤따르는 반면, separable convolution은 일반적으로 non-linearity 없이 구현된다. 실험을 통해 연산의 지역 정보와 채널간의 상관관계를 연산하는 사이에 non-linearity 함수가 있으면 성능이 크게 저하된다는 사실을 알게되었기 때문이다.

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/143.png" width="70%"></center><br>

- 구조
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/148.png" width="70%"></center><br>

    + Xception 구조는 36개의 convolution layer로 feature extraction을 수행한다. Entry flow를 시작으로, 8회 반복되는 middle flow, 마지막에는 exit flow를 거치는 구조이다.

    + 모든 convolution과 separable convolution의 뒤에는 BN(Batch Normalization)이 뒤따른다.
        + BN(Batch Normalization)이란? [Deep Learning Concept](https://github.com/star6973/lotte_studying/blob/master/Research/MH.Ji/Deep%20Learning%20Concept.md)를 참고
    
    + 요약하자면, Xception 구조는 residual connection이 있는 depthwise separable convolution의 linear stack으로 볼 수 있다. 

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/149.png" width="70%"></center><br>

- 참고자료

> [Deep Learning Image Classification Guidebook [3] SqueezeNet, Xception, MobileNet, ResNext, PolyNet, PyramidNet, Residual Attention Network, DenseNet, Dual Path Network (DPN)](https://hoya012.github.io/blog/deeplearning-classification-guidebook-3/)
 
> [Xception](https://datascienceschool.net/view-notebook/0faaf59e0fcd455f92c1b9a1107958c4/)

> [(Xception) Xception: Deep Learning with Depthwise Separable Convolutions 번역 및 추가 설명과 Keras 구현](https://sike6054.github.io/blog/paper/fifth-post/)

<br><br><br>

## MobileNet
- MoblieNet은 컴퓨터 성능이 제한되거나 배터리 퍼포먼스가 중요한 곳에서 사용될 목적으로 설계된 CNN 구조이다.

- Cloud Computing vs Edge Computing

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/138.png" width="70%"></center><br>
    
    + 클라우드 컴퓨팅은 여러 디바이스들에서 나온 정보들을 클라우드에서 전부 처리하는 환경이다. 네이버의 NDrive, 구글의 Docs 등이 클라우드 컴퓨팅의 대표적인 예라고 할 수 있다. 클라우드 컴퓨팅이 탄생하면서 여러 기업들에 각광받으며 클라우드 환경으로 전환하였다. 그러나 클라우드 컴퓨팅에도 여러 문제가 있었다. 클라우드 서비스를 이용하는 사람들이 기하급수적으로 늘어나면서 서버 및 데이터 센터에서 처리할 수 있는 데이터의 양을 넘어서기 시작했고, 수집한 데이터를 분석하고 송신하는 과정에서 바랭하는 데이터 지연 현상도 문제가 발생했다. 또한, 컴퓨팅의 통신 과정에서 보안 문제도 발생하며, 데이터 처리 속도, 용량 및 보안 등의 문제를 해결하기 위해 탄생한 것이 엣지 컴퓨팅이다.

    + 엣지 컴퓨팅은 클라우드에서 모든 연산을 처리하는 것이 아니라, 모바일 디바이스들이 직접 연산을 하거나, edge들에서 데이터 연산을 하여 cloud에 데이터를 뿌려주는 것이다. 즉, 클라우드 컴퓨팅은 데이터를 처리하는 곳이 데이터 센터에 있는 반면 엣지 컴퓨팅은 스마트폰과 같은 장치에서 데이터를 처리한다.

    + 엣지 컴퓨팅의 장점 3가지
        1) 데이터 부하 감소  
           : 클라우드 컴퓨팅에서는 처리해야 할 데이터 양이 많을수록 시스템에 부하가 생기는 반면, 엣지 컴퓨팅은 해당 기기에서 발생되는 데이터만 처리하기 때문에 부하를 줄일 수 있다.

        2) 보안  
           : 클라우드 컴퓨팅은 중앙 서버 아키텍처로 데이터 전송부터 보안을 강화해야 하는 반면, 엣지 컴퓨팅은 데이터 수집과 처리를 자체적으로 처리하기 때문에 클라우드 컴퓨팅에 비해 상대적으로 보안이 좋다고 할 수 있다.

        3) 장애대응  
           : 클라우드 컴퓨팅을 사용했을 때 서버가 마비되면 치명적인 타격을 입지만, 엣지 컴퓨팅을 사용하면 자체적으로 컴퓨팅을 수행하기 때문에 효과적으로 장애를 대응할 수 있다.

    + 이러한 엣지 컴퓨팅 환경은 MobileNet과 같이 비대한 크기의 네트워크보다는 빠른 성능이 필요한 곳에서 사용한다.

- Techniques for Small Deep Neural Networks
    + DNN에서 작은 네트워크를 만들기 위한 기법으로 다음과 같이 있다.
        1) Remove fully-connected layers
        2) Kernel reduction(3x3 -> 1x1)
        3) Channel reduction
        4) Evenly spaced downsampling
            - 초반에 downsampling을 많이 하면 accuracy가 떨어지지만, 파라미터의 수가 적어짐.
            - 후반에 downsampling을 많이 하면 accuracy가 좋아지지만, 파라미터의 수가 많아짐.

        5) Depthwise separable convolutions
            - depthwise convolution은 채널 숫자는 줄어들지 않고, 한 채널에서의 크기만 줄어든다.
            - pointwise convolution은 채널 숫자가 하나로 줄어든다.

        6) Shuffle operations
        7) Distillation & Compression

    + MobileNet은 위의 7가지 중 3) channel reduction, 5) depth separable convolutions 7) distillation & compression 기법을 사용한다.

- Depthwise Seperable Convolutions
    + 채널의 수를 증가시키면서 구성된 convolution-pooling의 구조는 이해도 쉽고 구현하기도 쉽지만, 모바일 환경에서 구동시키기엔 convolution 구조가 무겁다. 따라서 이를 해결하고자 새로운 convolution 연산인 depthwise separable convolution이 등장한 것이다.

    + Xception 모델에서 나온 개념으로, convolution 연산을 각 채널별로 시행하고 그 결과에 1x1 convolution 연산을 취하는 것이다. 기존의 convolution이 모든 채널과 지역 정보를 고려해서 하나의 feature map을 만들었다면, depthwise convolution은 각 채널별로 feature map을 하나씩 만들고, 그 다음 1x1 convolution 연산을 수행하여 출력되는 feature map의 수를 조정한다. 이때의 1x1 convolution 연산을 pointwise convolution이라고 한다.

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/139.png" width="70%"></center><br>

    + 위와 같은 구조를 하면 어떤 장점이 있을까? 커널이 3개라 가정할 때,
        1) 기존의 convolution의 경우, (3x3)x3(R, G, B)의 커널이 3개이므로 파라미터의 수는 3x3x3x3 = 81개가 된다.  
        2) depthwise separable convolution의 경우, (3x3)x1의 커널이 3개(depthwise)(채널 별로 분리, R, G, B), (1x1)x3(출력의 채널을 3으로 설정)의 커널이 3개(pointwise)이므로 파라미터의 수는 3x3x1x3 + 1x1x3x3 = 36개가 된다.  

    + depthwise separable convolution은 다음 그림과 같은 효율이 있다.
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/141.png" width="70%"></center><br>

- 구조
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/140.png" width="70%"></center><br>

    + MobileNet의 구조는 VGGNet의 구조와 비슷하지만, 기존의 convolution을 depthwise separable convolution으로 대체하고, pooling 대신에 stride를 2로 설정하여 사이즈를 축소하고 있다.

- 참고자료

> [Deep Learning Image Classification Guidebook [3] SqueezeNet, Xception, MobileNet, ResNext, PolyNet, PyramidNet, Residual Attention Network, DenseNet, Dual Path Network (DPN)](https://hoya012.github.io/blog/deeplearning-classification-guidebook-3/)
 
> [MobileNet이란? 쉬운 개념 설명](http://melonicedlatte.com/machinelearning/2019/11/01/212800.html)

> [[논문리뷰] MobileNet V1 설명, pytorch 코드(depthwise separable convolution)](https://minimin2.tistory.com/42)

<br><br><br>

## ResNext
- vision recognition에 대한 연구는 "feature engineering"에서 "network engineering"으로 변화하는 추세이다. 따라서 feature가 수작업으로 만들어지는 것이 아닌, model의 architecture를 만드는 것으로 옮겨지고 있다.

- 하지만 architecture를 디자인하는 것은, 특히나 layer의 층이 깊어질수록 hyper parameter의 증가로 그 난이도가 어려워지고 있다. 같은 모양의 여러 블록을 쌓는 VGGNet처럼, ResNet도 VGGNet과 같은 방식으로 계승했고, 이 간단한 rule은 hyper parameter의 선택을 보다 간단하게 만들어주었다. 또한, VGGNet과 달리 Inception module은 낮은 연산량으로도 높은 정확도를 이끌어낼 수 있다고 증명했다. Inception module은 계속 발전하고 있지만, 메인 아이디어는 split-transform-merge 형태의 전략이다.

<center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/152.png" width="70%"></center><br>

- ResNext는 하나의 입력을 group convolution을 통해 여러개로 나누고, 1x1 convolution으로 입력을 transform하고, concat을 통해 merge를 진행한다. 또한, 기존의 ResNet보다 연산량은 줄이면서 더 높은 성능을 보여주었다.

- 구조
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/153.png" width="70%"></center><br>

    + 위의 그림은 ResNet과 ResNext의 기본 구성으로, ResNext에서 나오는 파라미터 C는 Cardinarity로, 새로운 차원의 개념을 도입한다. cardinality는 집합의 크기 또는 집합의 원소의 개수를 의미하는데, CNN에서는 하나의 block 안의 transformation 개수 혹은 path, brach의 개수 혹은 group convolution의 수로 볼 수 있다. 그림에서 64개의 filter의 개수를 32개의 path로 쪼개서 각각 path마다 4개씩 filter를 사용하는 것을 보여주고 있는데, 이는 AlexNet에서 사용했던 grouped convolution과 유사한 방식이다.

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/154.png" width="70%"></center><br>

    + 이전의 ResNet에서, ResNet50 이하의 깊이를 갖는 구조에서는 basic block, 즉 블록을 하나 쌓을 때, convolution을 2번 진행하였다. 하지만 ResNext에서는 2개의 블록만 쌓게 된다면 group convolution의 의미가 없어져 성능 향상에 의미가 없게 된다. 따라서 ResNext에서는 block의 depth가 3 이상일 때부터 성능이 향상된다고 한다.

- 실험
    + ImageNet dataset을 사용하며, input image를 224x224 random crop하였다.
    
    + shortcut connection을 위해서는 identity connection을 사용했다.

    + downsampling은 convolution 3, 4, 5 layer에서 진행하였으며, 각 layer의 첫 번째 블록에서 stride=2로 설정하였다.

    + SGD optimizer, mini-batch 256, 8 GPU를 사용했으며, weight decay=0.0001, momentum=0.9로 설정하였다.

    + learning rate는 0.1로 시작하여 학습을 진행하면서 3번에 걸쳐 1/10로 감소시켰다.

    + weight initialization을 사용하였고, 모든 convolution 이후에는 BN을 수행하였고, 그 이후에는 ReLU로 활성화 시켰다.

- 특징
    + ResNext의 큰 특징이라고 하면 group convolution이다. 이때 group의 수를 cardinality라고 하는데, group의 수를 늘릴수록 더 낮은 연산량을 가질 수 있다. 따라서 같은 연산량을 갖는 네트워크라고 하면, group을 늘리면 더 깊은 채널을 가질 수 있다.

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/155.png" width="70%"></center><br>

    + 위의 표는 파라미터를 일정 수준으로 유지하면서 cardinality와 block width를 변경해주면서 비교한 표이다. group의 수를 늘리면 더 많은 채널을 이용할 수 있다.

- 참고자료

> [Deep Learning Image Classification Guidebook [3] SqueezeNet, Xception, MobileNet, ResNext, PolyNet, PyramidNet, Residual Attention Network, DenseNet, Dual Path Network (DPN)](https://hoya012.github.io/blog/deeplearning-classification-guidebook-3/)

> [ResNeXt:Aggregated Residual Transformations for Deep Neural Networks](https://blog.airlab.re.kr/2019/08/resnext)

<br><br><br>

## Residual Attention Network
- ResNet에 Attention mechanism을 convolution network에 접목시킨 구조이다.
- Attention이란?
    + RNN은 출력이 바로 이전 입력까지만 고려해서 정확도가 떨어진다. 전체 입력 문장을 고려하지 않고 다음 문장을 생성하기 때문이다. 그래서 seq-to-seq 모델이 등장하게 되었다. RNN은 시퀀스에서 작동하고 후속 단계의 입력으로 자신의 출력을 사용하는 네트워크이다. seq-to-seq는 2개의 RNN으로 구성된 모델이다. Encoder와 Decoder로 구성되며, Encoder는 입력 시퀀스를 읽고 단일 벡터를 출력하고 이 단일 벡터는 Context Vector라고도 불린다. Decoder는 Context Vector를 읽어 출력 시퀀스를 생성한다.

    + seq-to-seq 모델은 시퀀스 길이와 순서를 자유롭게 하여 두 언어간의 번역과 같은 task에 이상적이다. 하지만, LSTM의 한계와 마찬가지로 입력 문장이 매우 길면 효율적으로 학습하지 못한다.

    + seq-to-seq 모델의 2가지 문제가 있는데, 1) 하나의 고정된 크기의 벡터에 모든 정보를 압축하려고 하니까 정보 손실이 발생한다. 2) Vanishing Gradient의 문제가 존재한다. 의 문제를 가지고 있다.

    + 이러한 문제와 한계를 보정하기 위해 중요한 단어에 집중(attention)하여 decoder에 바로 전달하는 Attention 기법이 등장하였다.

    + Attention의 기본 아이디어는 Decoder에서 출력 단어를 예측하는 매 시점마다, Encoder에서 전체 입력 문장을 다시 한 번 참고한다는 점이다. 단, 전체 입력 문장을 전부 다 동일한 비율로 참고하는 것이 아니라, 해당 시점에서 예측해야 할 단어와 연관이 있는 입력 단어 부분을 좀 더 집중해서 보게 된다.

    + 이미지에서의 attention은 image classification과 detection 등에 적용되면서, 강인한 feature에 집중하여 추출하도록 할 때 사용된다.

- Attention의 아이디어를 computer vision 문제에 접목시켰다. 다음 그림과 같이 Attention을 적용하기 전에는 feature map이 분류하고자 하는 물체의 영역에 집중하지 못하는 경향이 있는데, Attention을 적용하면 feature map을 시각화했을 때, 물체의 영역에 잘 집중하고 있는 것을 확인할 수 있다.
<center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/156.PNG" width="70%"></center><br>

- 특징
    + Attention module의 수를 증가시키면, 성능이 일정하게 늘어난다. 또한 각각의 module은 서로 다른 형식의 attention을 감지하도록 학습된다.

    + 기존의 DNN 구조에 바로 적용하여 end-to-end로 학습이 가능하다.

    + 여러 Attention module을 쌓는 대신, 하나의 네트워크로 마스크를 생성하는 방법도 있지만 몇 가지 단점이 있다.
        - 첫 번째는 복잡하거나 많은 모양 변화를 가지는 경우에는 서로 다른 방식의 attention을 가지도록 모델링이 되어야만 한다. 하지만 그렇게 되려면 각 layer의 feature가 서로 다른 attention 마스크를 가지도록 해야 하는데, 하나의 마스크로는 불가능하다.

        - 두 번째는 하나의 module은 하나의 feature에만 영향을 주기 때문에, 잘못 적용하면 다음에 수정하기란 매우 힘들다.

    + 이러한 단점을 해결하기 위해 각 Trunk Branch에 붙은 Soft Mask Branch는 그 feature에 맞는 specialized된 마스크를 제공한다.

- 구조
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/157.PNG" width="70%"></center><br>

    + Residual Attention Network는 여러 Attention Module을 쌓아서 만들었다. 각 Attention Module은 Soft Mask Branch와 Trunk Branch로 이루어져 있다.

    + Trunk Branch
        - feature를 만들어내는 브랜치로, 일반적인 convolution 연산이 수행된다.

    + Soft Mask Branch
        - Attention mechanism에 따르면, Mask Branch는 fast feed-forward sweep과 top-down feedback step을 가지고 있다. 첫 번째 것은 전체 이미지의 글로벌 정보를 수집하고, 다음 단계로 글로벌 이미지 정보를 원래 feature map에 통합하게 된다. convolution network에서는 이를 bottom-up top-down의 fully convolutional 구조로 풀어지게 된다.

        - 입력된 데이터를 약간의 residual unit을 통과시킨 다음 max pooling을 몇 번 적용하여 receptive field를 증가시킨다. 가장 낮은 해상도까지 다다르면 input feature의 글로벌 정보는 확장되어 각 위치로 들어간다. 다시 residual unit을 몇 번 통과시킨 뒤, max pooling과 같은 수로 linear interpolation으로 출력을 upsampling하면, 원래의 input feature와 같은 크기로 확장할 수 있다.

        - 마지막으로 1x1 convolution 연산을 적용한 뒤 sigmoid 활성화를 하여, 출력값을 0에서 1로 조절한다. 여기에 bottom-up과 top-down 사이에 skip connection을 추가하여 스케일간 정보를 얻도록 하였다.
    
    + Attention Residual Learning
        - 단순히 attention module을 쌓는 것만으로는 성능이 올라가지 않는데, 0에서 1사이의 값을 가진 마스크가 계속 적용되면서 feature들이 점점 약해지기도 했고, 마스크가 Trunk Branch의 residual unit의 identical mapping 성질을 깨버리기도 했기 때문이다.

        - 따라서 마스크가 identical mapping을 유지할 수 있도록 하기 위해 attention module의 출력값 H(x)를 수정하였다. 식 $$H_{i,c}(x) = (1 + M_{i,c}(x)) * F_{i, c}(x)$$ 에서 $$M(X)$$는 0에서 1사이의 범위를 가지기 때문에 만약 값이 0인 경우, 원래의 특징 $$F(x)$$을 내보내도록 될 것이다. 이것을 attention residual learning이라고 한다. 즉, 마스크의 역할은 feature 중 더 좋은 feature를 강조하고, noise feature를 약하게 하는 역할을 할 수 있게 된다.

    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/158.PNG" width="70%"></center><br>

    + Attention을 주는 방식이 Spatial Attention과 Channel Attention이 있다.
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/159.PNG" width="70%"></center><br>

    1) Channel Attention  
        - Feature map의 "what"에 집중된다.
        - 일반적으로 공간의 차원을 압축하여 정보를 집계하는 average pooling을 사용하며, 여기에 채널에 대한 더 세밀한 관심을 inference하기 위해 max pooling을 동시에 사용한다. 이처럼 독립적으로 사용하는 것보다 모두 이용하는 것이 네트워크의 표현력을 크게 향상시키는 것을 확인할 수 있다.

    2) Spatial Attention   
        - feature의 공간적 관계를 이용하는 것으로, feature map의 "where"에 집중한다.
        - channel attention을 보완하는 정보적인 부분으로, channel attention처럼 average/max pooling을 적용하고, 이를 연계하여 효율적인 feature를 생성한다.
        - convolution layer를 적용하여 생성하며, 2개의 pooling을 사용하여 feature map의 채널 정보를 집계한 후 강조하거나 억제할 위치를 인코딩한다.

- 참고자료

> [Deep Learning Image Classification Guidebook [3] SqueezeNet, Xception, MobileNet, ResNext, PolyNet, PyramidNet, Residual Attention Network, DenseNet, Dual Path Network (DPN)](https://hoya012.github.io/blog/deeplearning-classification-guidebook-3/)

> [Residual Attention Network for Image Classification](http://www.navisphere.net/6130/residual-attention-network-for-image-classification/)

> [밑바닥부터 이해하는 어텐션 메커니즘](https://glee1228.tistory.com/3)

> [Convolutional Block Attention Module](https://velog.io/@wjdrbwns1/Convolutional-Block-Attention-Module)

<br><br><br>

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

<br><br><br>