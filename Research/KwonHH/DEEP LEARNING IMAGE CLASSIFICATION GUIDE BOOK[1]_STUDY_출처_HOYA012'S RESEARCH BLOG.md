# Deep Learning Image Classification Guidebook [1]
## LeNet, AlexNet, ZFNet, VGG, GoogLeNet, ResNet


1. LeNet5(1998)
    1. 등장 배경
        - MLP의 한계
            - input의 pixel 수가 많아지면 parameter수가 기하급수적으로 많아지는 문제 지적
            - local distortion에 취약한 문제
    1. 특징
        - Input을 1차원적으로 바라보던 관점에서 2차원으로 확장
        - parameter sharing을 통해 input수가 증가해도 parameter수가 변하지 않음
        - 손글씨 인식을 위해 제안된 architecture
            -당시 컴퓨터 성능에 의해서 32x32라는 작은 size의 image를 input으로 사용하고, 	  layer 수도 2개, Conv layer 2개, FC layer 2개 등

1. AlexNet(2012)
    1. 특징
        - ILSVRC 대회에서 2012년 우승한 architecture
        - 224x224 size의 RGB 3channel image를 input으로 사용
        - Multi GPU Training : 위 그림과 같이 2갈래로 나눠지며 중간에 중간결과를 공유 ; 당시 GPU의 Memory가 부족
        - activation 함수로 ReLU 사용 ; tanh 사용할 때보다 빠르게 수렴 가능
        - LRN(Local Response Normalization) 사용 ; 최근에는 거의 사용하지 않음
        - Overlapping pooling 사용 : pooling의 kernel size를 stride보다 크게 함
        - Drop out
        - PCA를 이용한 data augmentation 기법

1. ZFNet(2013)
    1. 특징
    - ILSVRC 대회에서 2013년 우승
    - AlexNet 기반
        - Conv layer의 filter size 11->7
        - stride 4->2
        - Conv layer의 filter 개수 증가
    - architecture에 집중하기보다는 학습이 진행됨에 따라 feature map의 시각화, 모델이 어느 영역으로부터 예측을 하는지 관찰(Occlusion 기반의 attribution 기법 등) 등의 시각화 측면에 집중

1. VGG(Visual Geometry Group, 2014)
    1. 특징
        - Oxford Network
        - ILSVRC에서 2014년 2위
        - 이전 방식들과 다르게 비교적 작은 크기(3x3)의 conv filter를 깊게 쌓음
            ex)AlexNet, ZFNet 8개의 layer, VGG : 11 / 13 / 16 / 19 layers
            - why? 3개의 3x3 layer를 중첩하면 1개의 7x7 conv layer를 사용할 때와 	receptive field가 같지만, activation 함수를 더 많이 사용할 수 있음<br>
            => 더 많은 비선형성을 얻을 수 있음, parameter 수 감소

1. GoogLeNet(Inception, 2014)
    1. 특징
        - ILSVRC 2014 우승
        - 이름은 Google + Le Network
        - NIN(Network In Network) 논문의 영향을 많이 받음
        - 총 22개의 layer로 구성<br><br>

        1. Inception Module 이라는 block 구조 제안
            - concatenation : 총 4가지 서로 다른 연산을 거친 뒤 feature map을 channel 방향으로 합침
                - 기존 방식 : 각 layer 간에 하나의 conv, 하나의 pooling 연산으로 연결
            - Inception module을 총 9번 쌓아서 구성
        
        1. Naive Inception module
            - 다양한 receptive field를 표현하기 위해서 1x1, 3x3, 5x5 conv 연산 섞어서 사용
        1. Inception module with dimension reduction
            - bottle neck 구조 : Naive Inception module에 의한 연산량이 많기 때문에 1x1 conv를 추가하여 feature map의 수를 줄인 다음에 3x3, 5x5 conv 연산을 수행하여 feature map 개수를 다시 키워주는 방식
            - 연산량을 절반이상 줄일 수 있었음

        1. Auxiliary classifier : 3, 6번째 Inception module에 classifier를 추가로 붙여서 총 3개의 classifier
            - 기존의 방식으로는 input과 가까운 초기 gradient는 잘 전파되지 않을 수 있는데, network의 앞, 중간에 softmax classifier를 추가하여 vanishing gradient 완화
            - 단, Auxiliary classifier로 구한 loss는 보조적인 역할만하기 위해서 0.3을 곱한 뒤 total loss에 더해줌
            - training시에만 사용하고, inference에서는 사용하지 않음
        1. Global Average Pooling(GAP)
            - NIN 논문에서 제안된 방식
            - 모든 element의 평균을 구해서 하나의 node로 바꿔주는 연산
            - feature map 개수만큼 node output으로 출력
                - GoogleNet에서는 1024->ImageNet class(1000개) 수만큼 output 출력
                - Fully Connected layer만 사용하여 classifier를 구성
                - Fully Connected layer에서 parameter 수를 크게 줄일 수 있음

1. ResNet(2015)
    1. 특징
        - Microsoft Research에서 제안한 구조
        - ILSVRC에서 2015년 1위
        - Residual Block 사용
            - layer개수가 늘어날수록 연산량, parameter 개수가 증가하지만, 정확도 향상도 기대할 수 있음
            - Shortcut ; identity shortcut ; input feature map을 그대로 ouput 더해줌
                - Outupt feature map의 개수가 2배로 커질 때마다 feature map의 가로, 세로 size를 절반으로 줄여주는 방식 이용
                - 이 경우 pooling 대신 stride=2를 갖는 conv연산을 이용함
                - 또한 shortcut에서도 feature map size를 줄여주어야 하며, 이때 identity short cut 대신 projection shortcut 이용
                 ==> vanishing gradient에 강인한 학습 가능
                - ResNet-50 이상의 모델에서는 feature map의 개수 많으므로 연산량도 증가하는데, Inception module의 bottleneck 구조를 차용하여 bottleneck residual block 사용
        - Batch Normalization(BN)
            - Residual block에 BN 사용
            - Conv -> BN -> ReLU 순으로 배치