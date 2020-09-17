# MobileDets_Searching for Object Detection Architectures for Mobile Accelerators
## 논문 리뷰
- Abstract
    - Depth-wise convolution 기반의 Inverted bottelneck 구조가 현재까지 널리 사용되고 있음
    - 본 논문에서는 layer 내에서 효과적인 convolution 을 재방문하여, mobile accelerator의 최적화 된 design 에 대해서 연구
    - 즉, search space 내에서 검색된 convolution을 일부 포함시킨 결과, latency 와 accuracy 사이에 trade off 에서 상당한 성과를 낼 수 있었음<br>이런 과정은 NAS(Neural Architecture Search)를 거쳐 진행<br><br><br>
1. Introduction
    - 최근의 computer vision application에서는 높은 성능을 위해서 다수의 network를 포함하지만, 이것은 소요되는 자원을 증가시키고, 이는 성능-연산 사이의 trade off를 고민하게 만든다
    - 불행히도 인간의 기술로 만들게 되면 시간 소요가 크고, 최선의 결과물을 완성할 수 없다<br><br>
    - 위 문제를 해결하기 위해서 다양한 방법들이 제안되었고, 특히 NAS(Neural Architecture Search) 방법이 model을 찾는데 있어서, 정확할 뿐 아니라 효과적으로 능력을 수행한다
    - NAS 알고리즘이 많은 진보가 있었음에도 불구하고, _<u>IBN</u>_ (Inberted Bottelneck) 구조는 계속 주목을 받았다
        - 이는 parameter의 수와 FLOPs 를 줄이는데 매우 좋고, 게다가 depth-wise convolution을 CPU에 최적화 될 수 있도록 해준다
        - 그러나 몇몇 mobile accelerators에서는 최적화되지 않았다
            - 예를들면 Edge TPU accelerators & Qualcomm DSPs 는 mobile divices 에서 regular convolution 을 가속화하기 위해서 각광받는 모델이다.
            - 여기서 특정 shape를 가지는 tensor 와 kernel 차원수가 있는데, regular convolution은 Edge TPU accelerator의 depthwise variation 보다 하드웨어의 사용을 3X 더 효과적으로 만들어준다<br> (훨씬 더 연산량이 많음(7X more FLOPs)에도 불구하고 더 효과적이다)<br> ==> 이것이 배타적(exclusive)으로 IBN을 사용하는 것에 대한 의문을 갖도록 했다<br><br>
    - 본 논문에서는 배타적인 IBNs의 사례를 Driving application 의 Object Detection 에 대해서 조사했고, object를 tracking 하는 것이 필수적인 경우 ex)자율주행 자동차, 비디오 감시 등. 에서 널리 사용되고 있었다
    - 전통적으로 object detectors 는 backbone 디자인으로 classification을 재사용했다 ==> 이런 간편한 방법(재사용)은 NAS에 의해서 최선의 방법이 아니라는 것이 증명됐다<br><br>
    - 본 논문에서는 search space 의 크기를 키우고, IBNs 와 Full convolution sequence (_T_ ensor - _D_ ecomposition - _B_ ased search space 에서 영감을 받음) 를 포함시켰다<br>이것은 mobile accelerator에서 광범위하게 좋은 결과를 나타낸다<br><br>
    - 효과적으로 새로운 building block을 할당하기 위해서, 논문에서는 다양한 mobile 플랫폼들을 타겟에 추가하여 latency-aware architecture search를 진행했다
        - 먼저, object detection 전용 NAS 하드웨어 플랫폼에서 사용될 때 지속적으로 성능을 향상시키는지 보여준다
        - 또한, TDB space에서 SOTA의 모델들을 가져오는 conducting architecture search 를 보여준다<br><br>
    - Contribution
        - 오직 IBN만 사용하는 search space는 현대의 mobile accelerator 중 일부에서는 최선이 아닐 수 있음을 밝혀냄
        - 유효한 convoultions 를 재방문하는 TDB search space 의 사용을 제안
        - 새로운 accelerator에서 NAS가 어떻게 높은 성능을 내는 architecture를 찾는지 설명<br><br><br>
 1. Related Work
    1. Mobile Object Detection
    1. Mbile Neural Architecture Search(NAS)
    1. NAS for mobile Object Detection
         - NAS 의 대부분은 classification 과 backbone으로 이미 학습된 feature extractor를 다시 제안하는 것에 그쳤다
         - 최근, 많은 논문에서는 더 나은 latency-accuracy trade-off 를 얻도록 object detection으로부터 직접 search 를 했다
         - 지금까지 depth-wise 와 feature pyramid 방식은 이런 플랫폼에 덜 최적화 되어있다
         - MnasFPN 은 backbone을 위해서 search 하지 않으며, latency를 위한 bottleneck 구조이다<br><br>
         - 그에 비해 논문에서는 SSD head에 기반을 두었고, full-convolution 에 기반을 둔 새로운 search space를 제안하였다
         - 그리고 그것은 mobile accelerator 을 더 잘 처리할 수 있게해준다<br><br><br>
1. Revisiting Full Convs in Mobile Search Spaces
    - __IBNs are all we need?__
        - IBN은 parameter 수와 FLOPs를 줄이기 위해서 design되며, depthwise-separable 커널이 mobile CPU에서 잘 동작하도록 해준다
        - 그러나 모든 FLOPs는 같지 않고, 특히 Edge TPU & DSPs에서 depthwise-separable 방식보다 regular convolution 은 3배 더 빠르고 7배 더 FLOPs 가 많다
        - IBN 만의 search space를 사용하는 것은 mobile accelerator에 대해서는 최선의 방법은 아이디어는 아니었지만, 이것으로부터 영감을 얻었고,<br>revisiting regular full conv를 재방문하여 IBN만의 search place를 확장시켜, mobile accelerator만의 space를 만들었다
        - 특히, 논문에서는 2개의 flexible layer를 적용하여 각각 channel의 팽창과 압축을 하였다.<br>이에 대한 자세한 설명은 다음 내용에서 이어진다<br><br><br>
    1. Fused Inverted Bottleneck Layers (Expansion)
        - depthwise-separable conv 는 IBN(Inverted BottleNeck)의 중요한 요소이다
        - depthwise-separable 연산은 expensive full conv 연산을 대체하기 위한 것으로, spatial dimension 연산을 위한 depthwise convolution 과 channel dimension 연산을 위한 1 X 1 convolution 로 진행된다
        - 그러나, expensive 방식의 개념은 FLOPs 또는 parameter의 수를 기반으로 널리 정의되는데, 현대의 mobile accelerators에 대해서는 굳이 필수는 아니다<br><br>
        - regular convolution 을 일부 포함하기 위해서 논문에서는 IBN들의 _1x1 conv 과 그것의 다음 K x K depthwise conv를 합진 것_ 으로 수정할 것을 제안하였다(나중에 expansion ratio와 연결)
        - 논문에서는 expansion의 개념을 full conv 를 channel size로 확장하였다. 그러므로 이들 __layer를 inverted bottleneck__ 또는 간단하게 __Fused Convolution layer__ 라고 명명하였다<br><br><br>
    1. Generalized BottleNeck Layers(Comparession) 