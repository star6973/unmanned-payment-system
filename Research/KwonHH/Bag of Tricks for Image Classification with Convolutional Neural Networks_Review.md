# Bag of Tricks for Image Classification with Convolutional Neural Networks 논문리뷰

- Abstract
    - 최근 많은 연구에 의해서 이미지 분류에 있어서 많은 기술 개선이 이뤄짐
    - 하지만 이와 같은 내용들은 간단히 언급되거나 코드에서만 볼 수 있음
    - 본 논문에서는 각 기술들의 검토와 최종 모델 정확도에 미치는 영향을 경험적 검토
    - 특히 ResNet을 이용해 ImageNet의 데이터셋을 분류해보고 성능 향상을 확인
<br><br>
1. Introduction
    - 2012년 AlexNet의 소개 후 CNN은 이미지 분류에 지배적인 접근
    - VGG, NiN, Inception, Resnet, DenseNet, NASNet 등
    - 동시에 이미지 분류 정확도 꾸준히 향상
    - 그러나 이러한 정확도 향상은 단순히 architecture 발달로 인한 것만은 아님
    - loss function, data 전처리, optimization 방식 등 역시 주요한 역할
    - 대부분의 논문들은 그런 것들을 간단히 언급하고 넘어감. 본 논문에서는 그것들을 검토 및 평가
    - 그런 기법, 모델 architecture는 정확도를 향상시키지만 계산 과정을 복잡하게 만들지 않음
    - 본 논문에서는 ResNet-50과 트릭들을 적용한 후의 ResNet-50을 비교함
<br><br>
1. Training Procedures<br>
![Training_Preprocess](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/Saved%20Pictures/bag%20of%20tricks%20image01.png)<br>
According to Tong He et al., [Bag of Tricks for Image Classification with Convolutional Neural Networks], Amazon Web Services, 2019, p2  
- 논문에 제시된 방법으로 전처리를 하고, validation 하는 동안 “이미지의 가로세로 비율을 유지하면서 짧은 가장자리의 크기를 256 픽셀로 조정”하고, “224X224 영역을 찾아 RGB 채절 정규화” 수행
- 이는 훈련과 유사한 환경에서 진행하기 위해 필요한 과정
    1. Weight 초기화
        - 0으로 초기화 하면 back propagation algorithm 에 의해서 학습할 때 학습이 되지 않음
        - 제대로 된 값으로 초기화 할 경우 더욱 성능이 향상
        - 초기에는 RBM(Restricted Boltzmann Machine)에 의해서 초기화하였고, 이후 Xavier 와 HE 방식이 개발되어 널리 쓰이고 있음
        - 본 논문에서 convolution 층과 fully-connected 층의 weight를 Xavier 알고리즘에 의해서 초기화<br>
        - 파이썬 코드 예시<br>W = np.random.randn(fan_in, fan_out)/np.sqrt(fan_in)<br>입력(fan_in)과 출력(fan_out)사이의 난수를 입력의 제곱근으로 나눠줌<br>

    1. Bias 초기화
        - 0으로 초기화<br>

    1. Optimizer
        - NAG(Nesterov Accelerated Gradient) Descent 사용
        - 기존의 Gradient Descent Optimizer 보다 개선된 알고리즘 : 관성으로 인한 업데이트 추가
<br><br>
1. Efficient Training
    1. Large-Batch Training
        1. Linear scalinig learing
            - mini batch SGD에서 gradient descending 은 각 batch에서 무작위로 샘플을 뽑아 random으로 진행
            - batch size를 키우면 SGD의 예측값을 변화시키진 않지만, variance 감소 즉, gradient의 노이즈가 감소
            - 따라서 gradient의 반대 방향에 대해서는 learning rate 증가시켜 진행
            - ResNet-50에 대해서 learning rate 초기값을 0.1로 하고 batch size를 256으로 할 때, batch size를 b로 증가시킨다면 그때의 learning rate 값은 0.1*b/256으로 계산
            
        1. learning rate warmup
            - 초반에는 parameter들이 무작위 값이기 때문에 해와는 멀리 떨어지게 된다.
            - 너무 큰 값으로 learning rate를 설정하면 결과가 불안정하게 된다.
            - 경험적으로 warmup을 위해서는 초기에는 learning rate를 작게 설정하고, training이 안정적일 때 learning rate를 키워준다.
            - 수식으로 정리하면 초기 batch size를 m이라고 가정할 때, 초기 learning rate(eta)와 m의 관계는 i*eta/m
            
        1. zero gamma
            - internal covariate shift를 해결하기 위해서 고안된 BN(Batch Normalization)<br><br>*batch normalization*<br>Batch normalization is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch. This has the effect of **stabilizing the learning process** and **dramatically reducing the number of training epochs required** to train deep networks
            - 입력을 각 층의 배치에 표준화하는 기술 -> 학습 과정을 안정화, 학습 에포치 수를 극적으로 줄여줌<br>
            - Batch Normalization 하는 과정에서 input x는 로 standardize 함
            - 경험적으로 gamma 값은 1, beta값은 0으로 할 경우 초기에 얕은 layer들을 학습하는 것처럼 train이 쉬워짐
            
        1. No bias decay
            - weight decay는 종종 weight 와 bias를 포함한 모든 학습 parameter에 적용된다.
            - 하지만 그것은 over fitting을 피하기 위해서 weight를 정규화 하는 경우에만 해당
        1. Low-precision training
           - FP16에서 연산하고 다시 FP32로 변환하여도 크게 정확도가 감소하지 않지만, 반면에 연산 속도가 증가
<br><br>
1. Training Refinements
    1. Cosine Learning Rate Decay
    - learninig rate 조정은 training에 있어서 매우 중요
    - learining rate warmup(3.1) 이후 논문의 저자는 안정적으로 초기 learninig rate 값을 감소시킨다. 이때 매우 널리 사용되는 방법이 자연대수에 의해 계산되는 *step decay*
    - step decay : decrease rate at 0.1 per 30 epochs<br><br>
    - 반면에 cosine 함수에 의한 *cosine decay*도 사용됨<br>수식 : batch = t 일 때, learning rate = 0.5*(1+ cos((t*pi)/T))*learning rate<br>
    - step decay vs. cosine decay
        - cosine decay는 초기에 learning rate가 천천히 감소하다가 이후 선형적으로 감소, 이후 다시 천천히 감소
        - step decay와 비교하면 cosine decay는 처음부터 decay가 되지만 step decay가 감소할 때도 10배 큰 laerning rate를 유지(이는 잠재적으로 training progress 향상시킴)
        ![Learing_rate_Schedule](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/Saved%20Pictures/bag%20of%20tricks%20image02.png)<br>
        According to Tong He et al., [Bag of Tricks for Image Classification with Convolutional Neural Networks], Amazon Web Services, 2019, p6<br>
<br><br>
    1. Label smoothing
    - 분류기 마지막 층은 label의 수(K)와 같은 size를 가지며, predicted confidence score를 출력한다.
    - 이 점수들은 softmax 함수에 의해서 q로 변환되어 예상 확률값을 나타내고 다음과 같이 계산된다.<br>
    ![Softmax_function](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/Saved%20Pictures/bag%20of%20tricks%20image03.png)<br>
    According to Tong He et al., [Bag of Tricks for Image Classification with Convolutional Neural Networks], Amazon Web Services, 2019, p6<br>
    
    - Cross Entropy
        - Entropy : <br>
        ![Entropy](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/Saved%20Pictures/bag%20of%20tricks%20image04.PNG)
        <br>
        - Cross Entropy : <br>
        ![Cross_Entropy](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/Saved%20Pictures/bag%20of%20tricks%20image05.PNG)
        <br>
        ![Cross_Entropy](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/Saved%20Pictures/bag%20of%20tricks%20image06.PNG)
        실제 yc 일 확률분포  
        ![Cross_Entropy](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/Saved%20Pictures/bag%20of%20tricks%20image07.PNG)
        yc가 yc임을 예측할 확률 분포<br>
        - 예측 모형은 실제 분포인 q를 모르고, 모델링을 통해서 q의 분포를 예측하고자 하는 것
        - 머신러닝을 통한 예측 모형에서 훈련데이터에서는 실제 분포인 q를 알 수 있기 때문에 cross entropy를 계산 가능        
        - 일반적으로 Cross Entropy > Entropy<br><br>
        - 논문의 저자는 l(p,q)의 각 parameter p, q가 서로 비슷해지도록 업데이트 한다. 이때 어떻게 ![eq](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/Saved%20Pictures/bag%20of%20tricks%20image08.PNG) 를 알 수 있을까?
        - 최적의 해결은 ![eq](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/Saved%20Pictures/bag%20of%20tricks%20image09.PNG) 를 다른것들이 충분히 작을 때는 무한대로 놓는 것이다. 다시 말해서, 잠재적으로 overfitting 될 수 있는 완전히 구별된 score 출력을 권장
        - 이러한 label smoothing 아이디어는 Inception-V2 모델에서 처음 제안되었다.
        - 그리고 그때 실제 확률분포 qi= i가 y일 때 1-epsilon, 그렇지 않으면 epsilon/(K-1) 로 정의된다. 다음 식에서 epsilon은 충분히 작은 상수 이다. 
        ![eq](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/Saved%20Pictures/bag%20of%20tricks%20image10.PNG)<br><br>
        - 그리고 이때의 최적화 솔루션은 ![eq](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/Saved%20Pictures/bag%20of%20tricks%20image11.PNG)와 같이 정의된다.
        - epsilon 이 0이면 log((K-1)(1-)/)+alpha 는 무한대가 되고, 의 증가에 따라서는 값이 감소하게 된다.
        - 특히 epsilon이 (K-1)/K 의 값을 가지면 i=y일 때 ![eq](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/Saved%20Pictures/bag%20of%20tricks%20image12.PNG) = log(1)+alpha 이고, 그 외의 경우에도 alpha 이기 때문에 항등식이 성립한다.
        - 본 논문의 figure4-(a)는 imagenet data set에서 K=1000일 때  값의 변화에 따른 ![eq](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/Saved%20Pictures/bag%20of%20tricks%20image12.PNG)(GAP)의 변화를 보여준다.
        ![Theoretical_gap](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/Saved%20Pictures/bag%20of%20tricks%20image13.PNG)<br>
        According to Tong He et al., [Bag of Tricks for Image Classification with Convolutional Neural Networks], Amazon Web Services, 2019, p7<br><br>
        - 본 논문에서는 경험적으로 label smoothing을 한 ResNet50-D 와 하지 않은 ResNet50-D를 비교하여 Gap의 값을 비교하였는데, label smoothing을 한 결과의 center가 figure 4-(a)의 center에 위치하며, 양극단에 있는 값도 더 적다는 것이 확인되었다.        
        ![Empirical_gap_from_ImageNet_validation_set](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/Saved%20Pictures/bag%20of%20tricks%20image14.PNG)<br>
        According to Tong He et al., [Bag of Tricks for Image Classification with Convolutional Neural Networks], Amazon Web Services, 2019, p7<br><br>           
        
    1. Knowledge Distillation
        - Knowledge Distillation에서 Teacher model 은 현재 model(student model)의 training을 돕기 위해서 사용된다.
        - Teacher model은 종종 높은 정확도로 pre-trained 되고, 따라서 모방에 의해서 student model은 model의 복잡성을 유지하면서도 정확도를 향상시킬 수 있다.
        - 예를 들면 ResNet-152는 ResNet50의 train을 돕기 위한 Teacher model이다. <br><br>
        - Training 동안 본 논문의 저자들은 teacher model 과 student model의 softmax 출력에 distillation loss를 추가하여 training을 불리하도록 만들었다.

        - 입력에 대해서 p를 실제를 추정한 확률분포라고 하고, z 와 r을 각각 student model 과 teacher model의 fully connected layer라고 할 때
        - 이전에 p 와 z 사이의 차를 구하기 위해서 cross entropy loss를 구했던 것처럼 여기서도 같은 방식으로 distillation의 loss를 구한다.
        - 그러므로 loss는 다음과 같이 변화한다.
        ![eq](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/Saved%20Pictures/bag%20of%20tricks%20image15.PNG)
        T는 softmax 출력을 smooth하게 하기 위한 parameter. 그러므로 teacher의 prediction으로부터 label 분포에 대한 지식이 서서히 특징을 나타내게 된다.<br><br>
        
    1. Mixup Training
        - 본 논문의 2.1에서 논문 저자들은 어떻게 training 전에 이미지 augmentation을 하는지 묘사했다. 본 장에서는 다른 augmentation 방법을 소개한다 ; Mixup Training.
        - mixup을 위해서 매 시간 (xi, yi) 와 (xj, yj)의 랜덤 샘플을 구한다. 이후 각 샘플에 weighted linear interpolation으로부터 새로운 두 개의 데이터를 만든다.
        ![eq](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/Saved%20Pictures/bag%20of%20tricks%20image16.PNG)
        
    1. Experiment Results
        - 실험 조건
            - label smoothing 에서 epsilon = 0.1
            - model distillation 에서 T = 20, Teacher model 은 ResNet-152-D 이며, cosine decay 와 label smoothing 적용함
            - mixup training 에서 alpha = 0.2, epoch 수를 120에서 200 으로 증가 (mixed 데이터들은 수렴이 잘 되도록 하기 위해서 더 긴 training progress 필요)
            - mixup training 과 distillation 을 결합하기 위해서 teacher model 역시 mixup training 실시함<br><br>
        - 논문 저자들은 이러한 기법들이 imageNet data set 이나 ResNet 구조에만 한정된 것은 아니라고 설명한다. ResNet-50-D, Inception-V3, MobileNet을 ImageNet data set에 각각 기법들을 적용한 결과를 표6에 제시했다.<br>
        ![table](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/Saved%20Pictures/bag%20of%20tricks%20image17.PNG)<br>
        - 위 표를 보면 distillation이 ResNet에서는 좋은 효과를 보이지만 InceptionV3 와 MobileNet에서는 그렇지 못한 것을 확인할 수 있다. 논문 저자들은 teacher model이 같은 군의 student model이 아니기 때문에 확률 분포가 다르고, 이것이 model에 부정적인 영향을 준 것으로 해석한다.
        - 게다가 저자들은 trainig을 할 때 200epoch mixup을 제외시켰고, 그것이 Top-1 accuracy를 약 0.15% 증가시켰으므로 mixup training은 여전히 중요하다.
        - 본 논문의 기법들을 다른 dataset에도 전이 가능하게 지원하기 위해서 저자들은 ResNet-50-D model에 MIT Place 365 dataset을 정제 없이 훈련시켰다.
        - 그 결과는 table7에 있으며, top5 accuracy를 validation과 test set에서 모두 일관되게 증가하였음을 확인할 수 있다.
        ![table](file:///C:/Users/%EC%82%AC%EC%9A%A9%EC%9E%90/Pictures/Saved%20Pictures/bag%20of%20tricks%20image18.PNG)<br><br>


