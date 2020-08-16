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

>

>

>

<br><br><br>

## MobileNet

## ResNext


## PolyNet


## PyramidNet


## Residual Attention Network


## DenseNet


## Dual Path Network(DPN)