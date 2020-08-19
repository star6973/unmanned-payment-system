# Deep laerning Image Classification Guide book[4]
## SENet, ShuffleNet, CondenseNet, MobileNetV2, PNASNet, MnasNet
### 2018년도에 제안된 CNN architecture

1. SENet (Squeeze and Excitation Network, 2018)
    - 2018 CVPR 논문, 2017 ILSVRC 에서 1위
    1. 특징
        - Squeeze 연산
             - feature map을 spatial dimension( = H x W)을 따라서 합쳐줌 ; Global Average Pooling 사용
             - 이후 Excitation 연산이 뒤따름
         - Excitation 연산
            - self - gating 메커니즘을 통해 구현 ; input vector와 모양이 같은 output vector로 embedding 시킴
            - 즉, channel마다 weight를 줌 ; 다른 말로 channel 방향으로 attention 을 준다고 함
            - extention 연산을 통해 vector에 색이 입혀진 것을 확인할 수 있으며, 색이 배정된 vector와 input featur map(U)를 각 요소별로 곱하면, output feature map이 생성됨 -> output feature map의 각 channel마다 이전 단계에서 구한 weight(색깔)이 반영
    ![Figure1 A Squeeze and Excitaion block](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/%ED%98%B8%EC%95%BC%20%EB%B8%94%EB%A1%9C%EA%B7%B8%20%EA%B0%80%EC%9D%B4%EB%93%9C4%20%EA%B3%B5%EB%B6%80/01.png)<br><br>

    1. Squeeze and Excitation block 의 특징
        1. 이미 존재하는 CNN architecture 에 붙여서 사용 가능
            - 예 : ResNet network에서 Squeeze and Excitation block을 추가하면 SE-ResNet
            - 본 논문의 저자들은 이외에도 잘 알려진 다른 network에 SE block을 추가해본 결과 미미한 수치의 연산량은 증가하였지만 대신 정확도가 많이 향상<br>
    ![Squeeze and Excitation block 활용 예시](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/%ED%98%B8%EC%95%BC%20%EB%B8%94%EB%A1%9C%EA%B7%B8%20%EA%B0%80%EC%9D%B4%EB%93%9C4%20%EA%B3%B5%EB%B6%80/02.JPG)<br><br>
        1. 실험을 통해 알아낸 결과
            1. 
                - 아래 그림에서 SE block을 추가했을 때 Fully connected layer를 거쳐서 channel이 C에서 C/r로 줄었다가 다시 C로 변경됨. 이때의 r값을 reduction ratio라고 함<br>
        ![reduction ratio](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/%ED%98%B8%EC%95%BC%20%EB%B8%94%EB%A1%9C%EA%B7%B8%20%EA%B0%80%EC%9D%B4%EB%93%9C4%20%EA%B3%B5%EB%B6%80/03.JPG)
                - 이때 정확도와 complexity를 모두 고려한 최적의 r값은 16
            1. Squeeze 연산에서 Max와 Average 연산 중 Average 연산이 더 효과적
            1. Excitation 연산에서 ReLU, Tanh 보다 __Sigmoid를 쓰는 것이 효과적__<br><br>
            
1. ShuffleNet(2018)
    - 소개 : CVPR, "ShuffleNet : An Extremely Efficient Convolutional Neural Network for Mobile"
    1. 특징
        1. MobileNet과 함께 경량화 된 CNN architecture
        1. Group Convolution & Channel Shuffle
            - Depthwise Separable Convolution 연산이 제안된 이후 경량화 된 CNN에는 필수적임
            - 연산량을 줄이기 위해서 제안되었던 1x1 conv 와  Group Convolution을 적용하여 MobileNet보다 더 효율적인 구조 제안
            - 이러한 MobileNet 과 선의의 경쟁구도가 되면서 MobielNet V2 가 탄생하는 계기<br><br>
            
1. CondenseNet(2018)
    - 소개 : CVPR, "CondenseNet : An Efficient DenseNet using Learned Group Convolutions"
    - MobileNet, ShuffleNet과 같이 경량화 된 architecture<br>
    ![DenseNet vs CondenseNet](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/%ED%98%B8%EC%95%BC%20%EB%B8%94%EB%A1%9C%EA%B7%B8%20%EA%B0%80%EC%9D%B4%EB%93%9C4%20%EA%B3%B5%EB%B6%80/04.JPG)<br>
    - 좌측 1 : DenseNet 구조, 우측 2 : CondenseNet 구조<br>
    1. 특징
        - Learned Group Convolution 이라는 방법을 DenseNet에 접목
            - Learning Group Convolution
                - Group Convolution 사용<br>
![group convolution process](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/%ED%98%B8%EC%95%BC%20%EB%B8%94%EB%A1%9C%EA%B7%B8%20%EA%B0%80%EC%9D%B4%EB%93%9C4%20%EA%B3%B5%EB%B6%80/05.JPG) 
                    - Group의 수(G)는 hyper-parameter
                    - Condensation factor(C) 는 hyper-parameter
                    - 위 그림은 G = 3(빨강, 초록, 노랑)이고, C(Condensation) = 3인 경우. 각 training 은 condensing stage, optimization stage로 구성되어 있음
                    - condensing stage의 수는 C-1개이고, 각 condensing stage를 거치며 1/C개를 제외한 나머지(2/C)개의 connection을 제거하는 구조
                    - test 단계에서는 학습 때의 결과를 사용하며, 위 그림에서 2, 4 feature map의 경우 connection이 없고, 반대로 5, 12 feature map은 2가지 group과 연결되어 있는데, 이를 index layer를 통해서 순서를 맞춰준 뒤 group convolution 수행??
                    - 즉, __condensing stage를 통해서 학습 과정에서 자동으로 부족한 network를 제거__할 수 있었고, 이는 __pruning을 학습을 통해 자동으로 하는듯한 효과__
        - Network Pruning 아이디어 접목
            - _Network Pruning_ : layer간에 중요하지 않은 연결을 제거하는 방식
    1. DenseNet과는 다른점
        1. 다른 resoultion을 갖는 feature map 간에도 densely 연결을 하였다.
            - 이 경우 pooling을 통해서 resoultion을 맞춰줌
        1. feature map의 size 가 줄어들 때 growth rate를 2배 키워줌
<br><br>
1. MobileNet V2 (2018)
    1. 특징
        - Depthwise Separable Conv 사용 (MobileNet v1 에서도 사용)
        - Linear Bottle Neck
        - Inverted Residual
    