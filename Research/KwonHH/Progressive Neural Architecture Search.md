# Progressive Neural Architecture Search
## 논문 리뷰
Abstract
- 최근 발표된 강화학습 or Evolutionary Algorithm 기반의 learning the structure 방식보다 효율적인 방법을 제시<br>
=> SMBO 방식 : Sequential Model-based Optimization<br>
    - surrogate model이 structure space를 찾도록 학습하는 동안 찾는 structure의 복잡성이 점점 증가<br>
    => RL 방식(Zoph et al. 2018) 보다 최대 5배 효과적이고, 8배 연산이 더 빠름<br>
    => CIFAR-10 과 ImageNet에서 SOTA에 등극<br><br>
1. Introduction
    - Architecture Search 의 방법은 여러가지가 있지만 크게 RL, EA 로 나눌 수 있음
        1. EA 방식
            - 각 structure가 string처럼 encode 됨 -> search 과정 중에 string들이 무작위로 변형 및 재결합 -> 각 string이 train 및 validation set에서  evaluate 됨 -> 가장 우수한 동작을 하는 모델이 children 을 만든다<br>
        1. RL 방식
            - agent는 model의 structure를 정하기 위해서 연속된 동작을 수행->정해진 model은 train 및 validation 과정에서 reward를 return -> reward 값은 RNN controller 를 update함<br><br>
    - 비록 위 두가지 방식이 manually designed 된 architecture에 비해 좋은 성능을 보이지만, 상당한 연산 자원을 필요로 함<br>
      RL 방식의 경우 P100 GPU 500개로 4일의 시간이 소요됨<br><br>
    - 본 논문에서 accuracy 분야에서 이전 SOTA에 등재된 CNN 모델보다 5배 덜 model evaluation 과정을 하는 방식을 제안
        - Learning transferable architectures for scalable image recognition, Zoph et al.(2018) 처럼 전체 CNN model을 찾는 것이 아니라 _Convolutional Cell_ 을 찾음
        - 이때 cell은 B(block)로 구성되고, B(block)은 2개의 입력과 연산자로 구성
        - 이런 cell 구조는 training set의 size 와 CNN의 적당한 running time 에 따라 특정 갯수로 쌓임
        - 이렇게 모듈화 된 구조는 서로 다른 dataset으로 전이가 쉽도록 해줌<br><br>
    - 논문 저자들은 cell structure space 를 찾기 위해서 heuristic search 방식을 사용함(간단한 model 에서 점점 model 이 복잡해지는 방향으로)<br>또한 연결되지 않은 structure 들에 대해서는 pruning 적용<br>
        - b 회 동안 알고리즘을 반복하면서 관심있는 데이터셋에서 K개의 size = b 인 후보 cell을 설정
        - 과정이 매우 복잡하기 때문에 surrogate function을 학습시켜 structure의 학습 없이 그 값을 예측할 수 있도록 함
            - size = b 인 K 개의 후보들을 size = b+1 인 K`개의 children으로 확장 ( K` >> K)
            - 전체 K`개의 children에 대해서 surrogate function을 적용하여 rank를 매김 -> 상위 K개의 children 을 뽑고, train 및 evaluate
            - 위 과정을 b = B(cell 하나에 들어가는 블록의 수) 일때까지 반복<br><br>
    - 본 논문에서 간단한 model->복잡한 model로의 과정은 search space에서 structure를 통째로 정하는 방식에 비해 여러 장점이 있음
        1. 간단한 structure는 train이 빠르게 됨 => surrogate 를 train하기 위한 초기 결과를 빠르게 얻을 수 있음
        1. surrogate는 약간 더 큰 structure만 예측하도록 할 수 있음 cf. trust-region method
        1. search space를 작은 search space의 곱으로 분해할 수 있고, 그것은 만은 block를 가진 model들을 잠재적으로 찾을 수 있도록 해줌<br><br><br><br>
1. Architecture Search Space
    1. Cell Topoligies
        - Cell 은 fully convolutional network로 H X W X F 의 tensor 를 다른 H` X W` X F` 으로 사상된다
            - 이때 stride = 1 이면 H` = H, W` = W 이고, stride = 2 이면 H` = H/2, W` = W/2 이다
        - 공간적 activation 이 반으로 줄어들 떄마다 filter 개수(F)를 두배로 하기 위해서 공통의 heuristic 을 적용하면,<br>F` = F 이기 위해서 stride = 1, F` = 2F 이기 위해서 stride  = 2 이다<br><br>
        - Cell 은 B개의 block들로 구성된 DAG 로 설명된다
        - 각 block은 2개의 input tensor 와 1개의 ouput tensor 로 매핑된다
        - 논문 저자들은 b라는 block 로 구성된 c의 cell을 5개의 튜플로 정한다<br>: I1, I2, O1, O2, C <br>
            - Ib ( : I1, I2) 는 block으로의 input을 의미
            - O ( : O1, O2 ) 는 input에 operation이 적용된 것을 의미
            - C 는 출력에 대해서 어떻게 O1, O2를 결합하여 feature map을 그릴 것인가에 대한 의미<br> => H^c_b 라는 기호로 표현
            - 가능한 입력의 집합 Ib 는 현재 cell의 모든 이전 블록의 집합이다<br> {H^c_1, ..., H^c_b-1}, H^c-1_B ; 이전 cell의 output, H^c-2_B ; 이전이전 cell의 output
            - 연산자 space O는 다음의 8가지 함수를 가진다<br>3X3 depthwise-separable convolution    /    identity<br>5x5 depthwise-separable convolution    /    3x3 average pooling<br>7x7 depthwise-separable convolution    /    3x3 max pooling<br>1x7 followed by 7x1 convolution    /    3x3 dilated convolution
                - 이는 Learning transferable architectures for scalable image recognition, Zoph et al.(2018) 에서 사용한 것보다<br> 13개 더 적게 사용한 것<br> : RL 방식에서 발견한 것 중 사용되지 않는 것들을 제거
            - 또한 RL 방식에서 사용된 elementwise additon, concatenation을 모두 사용하지 않고,addition만 사용<br> : RL방식에서 concatenation은 사용하지 않는다는 것을 발견함<br> => 따라서 block은 4-tuple로 정의<br><br>
            - search space를 정량화 하여 search problem의 magnitude가 잘 보이도록 하였다
                - b번째 block을 Bb라고 할 때, 이것의 크기는 <br>|Bb| = |Ib|^2 * |O|^2 * |C| ( |Ib| = 2 + b - 1, |O| = 8, |C| = 1 )<br>
                
              
            

          