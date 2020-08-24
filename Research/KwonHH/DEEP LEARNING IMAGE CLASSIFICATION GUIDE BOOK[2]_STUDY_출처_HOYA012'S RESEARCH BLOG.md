# Deep Learning Image Classification Guidebook [2]
## LeNet, AlexNet, ZFNet, VGG, GoogLeNet, ResNet

1.Pre-Act ResNet(2016)
    1. 특징
    - Identity Mappings in Deep Residual Networks 논문에서 소개
    - Pre-Act = Pre Activation
        - Residual unit(Conv-BN-ReLU)의 ReLU를 Conv 앞에 배치<br><br>
    1. 논문 내용
        1. 기존 ResNet에서의 Identity Shortcut을 5가지 다양한 shortcut으로 대체
        - original
        - constant scaling
        - exclusive gating
        - shortcut-only gain
        - conv shortcut
        - drop out shortcut<br><br>
        - 결론 : gating기법과 1x1 conv등을 추가하면 representational ability는 증가하지만
            학습 난이도 상승으로 최적화가 어려워지는 것으로 추정
            : identity shortcut 방식(아무것도 하지 않았을 때)이 가장 성능 좋음<br><br>
        1. Activation Function(ReLU) 위치에 따른 성능
        - original : Conv – BN – ReLU – Conv – BN - addition
        - BN after addition
        - ReLU before addition
        - ReLU only pre activation
        - full pre-activation<br><br>
        - full pre-activation 이 가장 성능이 좋고, 안정도 좋음
        - original 의 경우 BN 이후 addition하면 다시 unnormalized 상태로 다음으로 전달
        - full pre-activation 구조는 모든 Conv 연산에 normalized input 전달되므로 성능이 좋아지는 것으로 추정<br><br><br>
1. Inception-v2(2016)
    1. 특징
        1. Conv Filter Factorization
            - Inception-v1은 VGG, AlexNet에 비해서 parameter수가 굉장히 적지만, 많은 연산을 한다는 단점이 있었음
            - v2에서는 이런 연산 복잡도를 줄이기 위해서 Conv Filter Factorization 제안
            - n x n conv => 1 x n + n x 1 conv로 쪼개는 방법 제안(receptive field 동일)<br>
        1. Rethinking Auxiliary Classifier
            - auxiliary classifier가 학습 초기 수렴성을 개선시키지 않았고
            - 학습 후기에는 약간의 정확도 향상을 얻을 수 있는 부분
            - 그리고 기존에 2개의 auxiliary classifier 사용
               ==> auxiliary classifier 는 있으나 없으나 큰 차이가 없으므로 제거<br>
        1. Avoid representational bottleneck
            - Grid Size Reduction 제안
                - representational bottleneck : CNN에서 pooling으로 인해서 feature map size가 줄어들면서, 정보량이 줄어드는 것
                - pooling을 뒤에 하면 연산량이 많아짐
                - 이런 두 가지 문제를 피하기 위해서 다음 사진과 같은 방식 제안<br>
            ![InceptionV2](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/noname02.png)<br><br>             
1. Inception-v3 (2016)
    - v2의 architecture는 그대로, 여러 학습 방법을 적용
        - Model Regularization via Label smoothing
            - one hot 대신 smoothed label 생성
        - Training Methodology
            - momentum optimizer->RMSProp optimizer->Gradient clipping with threshold 2.0->Evaluation using a running average of the parameters computed over time
        - BN-auxiliary classify
            - Auxiliary classifier 의 FC layer에 BN추가<br><br>
1. Inception-v4 (2016)  
1. Inception-ResNet(2016)<br><br>
1. Stochastic Depth ResNet(2016)
    1. 특징
        - Self-training with Noise Student improves ImageNet classification 논문에 소개
            - 새로운 architecture 제안보다는 기존 ResNet에 새로운 학습 방법을 추가하여 	ResNet layer 수를 overfitting 없이 크게 증가 가능
        - Stochastic Depth 방식 제안
            - randomness에 기반한 학습 방식
            - vanishing gradient로 인해서 학습이 느리게 되는 문제 완화
             cf. Dropconnect(아예 weight connection끊는 방법), Maxout, MaxDrop ...
                 위 방법들은 hidden unit(feature map)에 집중했다면 Stochastic depth란 network의 depth를 학습 단계에 random하게 줄임<br>
           - ResNet으로 예를 들면 확률적으로 특정 block을 inactive하게 하여 해당 block은 shortcut만 수행 => input과 output이 같아져서 아무런 연산도 수행하지 않는 block
             => network의 depth를 조절 ; 학습시에만 사용, test시에는 모든 block active<br>
        - CIFAR-10, SVHN 등에 대해서는 test error가 줄어들었지만, ImageNet과 같이 복잡하고 큰 데이터 셋에서는 효과가 미미
            - 단, CIFAR-10과 같은 비교적 작은 데이터 셋에서 ResNet을 1202 layer를 쌓았을 때 정확도가 떨어지지만, Stochastic Depth ResNet은 정확도 향상

1. Wide ResNet(2016)
    1. 특징
        - 기존 ResNet 보완 Network의 경우 정확도를 높이기 위해서 Depth(layer를 더 쌓음)만 높이려고 하지만, Width(Conv filter개수)를 늘리는 시도도 하였음
        - BN이후에는 잘 쓰이지 않던 dropout을 적용
        - 병렬처리 관점에서 효율이 좋아지기 때문에 WRN-40-4 구조가 ResNet-1001과 test error는 비슷하지만 forward+backward propagation에 소요되는 시간이 8배 빠름<br><br>
        - 이후 2019년 등장한 EfficientNet : Compound scaling :Width, Depth뿐만 아니라 Input resolution 동시에 고려