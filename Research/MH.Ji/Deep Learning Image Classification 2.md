# Image Classification의 종류 정리 2

## SqueezeNet
- AlexNet의 파라미터를 50배 이상 줄여서 0.5MB 이하의 model size를 가질 수 있는 architecture 구조를 제안하고 있다.

- original 논문[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://openreview.net/pdf?id=S1xh5sYgx)에서는 총 3가지 종류의 SqueezeNet의 구조를 제안하고 있으며, 모든 architecture는 아래에 있는 그림의 Fire module로 구성된다.

<center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/128.png" width="70%"></center><br>

<center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/129.png" width="70%"></center><br>

- Fire module은 1x1 convolution으로 filter의 개수를 줄인 뒤(squeeze) 1x1 convolution과 3x3 convolution을 통해 filter의 개수를 늘려주는(expand) 연산을 수행한다. 3개의 convolution layer의 filter의 개수는 하이퍼파라미터이며, 자세한 구조는 다음과 같다.

- 구조
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/130.png" width="70%"></center><br>

    + NIN, GoogLeNet 등에서 사용했던 것처럼 FC layer 대신 GAP를 이용하고,

## MobileNet

## ResNext


## PolyNet


## PyramidNet


## Residual Attention Network


## DenseNet


## Dual Path Network(DPN)