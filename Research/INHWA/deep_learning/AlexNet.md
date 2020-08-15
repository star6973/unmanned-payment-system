## AlexNet 


AlexNet의 기본구조는 LeNet-5와 크게 다르지 않다. 2개의 GPU로 병렬연산을 수행하기 위해서 병렬적인 구조로 설계되었다는 점이 가장 큰 변화이다

그 당시에 사용한 GPU인 GTX 580이 3GB의 VRAM을 가지고 있는데, 하나의 GPU로 실험을 하기엔 memory가 부족하여 위와 같은 구조를 가지게 되었다고 합니다.


![alexnet](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F99FEB93C5C80B5192E)

AlexNet은 8개의 레이어로 구성되어 있다. 5개의 컨볼루션 레이어와 3개의 full-connected 레이어로 구성되어 있다. 두번째, 네번째, 다섯번째 컨볼루션 레이어들은 전 단계의 같은 채널의 특성맵들과만 연결되어 있는 반면, 세번째 컨볼루션 레이어는 전 단계의 두 채널의 특성맵들과 모두 연결되어 있다

alexNet의 input = 244 x 244 x 3 이미지이다 [ 3 chanel ]

 
- 첫번째 레이어에 96개의 11x11x3 사이즈를  zero padding을 사용하지 않은체 stride = 4로 설정하여 결과적으로 55x55x96특성맵을 갖게 된다
그 뒤 Relu를 적용하고 다시 maxpooling = 3x3 , stride = 2로 subsampling을 시행하여 27x27x96 특성맵으로 갖게된다.
그 다음 수렴 속도를 높이기 위해 local response normalization을 시행한다. feature map의 크기는 변하지 않는다

- local response normalization이란 : 너무 강한 feature를 갖고있는 뉴런이 컨볼루션으로 섞이면 결국 그 강한 feature가 간섭을 하여 overfitting을 야기한다.
그래서 filter들을 square sum하여 한 filter에서만 과도하게 activation하는것을 막습니다

  
- 두번째 레이어는 zero padding을 하여 5x5x48,256개 커널을 사용해 convolution, 결과는 13x13x256으로 특성맵을 얻게된다

- 세번쨰 레이어는 두개의 특성맵을 섞는다. 3x3x256커널을 사용하여 384개의 커널을 사용하여 stride = 1 , padding = 1로 하여 13x13x384의 특성맵을 갖는다

- 네번째 레이어는 3x3x192커널을 사용하여 stride = 1, zero padding =1 을 사용하여 13x13x384 특성맵을 얻는다 

- 다섯번째 레이어는 3x3x192의 256개의 커널을 사용하며 3x3 maxpooling stride = 2 을 사용하여 6x6x256의 특성맵을 갖는다. 

- 여섯번째 레이어(Fully connected layer): 6 x 6 x 256 특성맵을 flatten해줘서 6 x 6 x 256 = 9216차원의 벡터로 만들어준다. 그것을 여섯번째 레이어의 4096개의 뉴런과 fully connected 해준다. 그 결과를 ReLU 함수로 활성화한다. 

- 일곱번째 레이어(Fully connected layer): 4096개의 뉴런으로 구성되어 있다. 전 단계의 4096개 뉴런과 fully connected되어 있다. 출력 값은 ReLU 함수로 활성화된다. 

- 여덟번째 레이어(Fully connected layer): 1000개의 뉴런으로 구성되어 있다. 전 단계의 4096개 뉴런과 fully connected되어 있다. 1000개 뉴런의 출력값에 softmax 함수를 적용해 1000개 클래스 각각에 속할 확률을 나타낸다. 


총, 약 6천만개의 파라미터가 훈련되어야 한다. LeNet-5에서 6만개의 파라미터가 훈련되야했던 것과 비교하면 천배나 많아졌다. 하지만 그만큼 컴퓨팅 기술도 좋아졌고 훈련기간이 줄이기 위한 방법을 사용되었기 떄문에 가능해졌다

