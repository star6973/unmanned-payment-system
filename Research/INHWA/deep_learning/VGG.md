### VGG-F, VGG-M, VGG-S의 구조

- VGG-F, VGG-M, VGG-S는 속도와 정확도의 트레이드오프를 고려한 모델들이다. VGG-F는 fast, 즉 빠름에 초점을 맞춘 모델이고, VGG-S는 slow, 즉 느리지만 정확도에 초점을 맞춘 모델이다. VGG-M은 medium으로 이 둘의 중간 정도라고 보면 된다

![VGG](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbOgKcE%2FbtqwkWmeIZS%2F2WpkIhZ3pDWqw3CtXseK31%2Fimg.png)

- 0: VGG-F는 224x224x3사이즈의 이미지를 입력받는다 

- 첫번째 레이어[convolution] : 64개의 11x11x3사이즈 필터커널로 stride =4, zero-padding = none으로 입력영상을 컨볼루션해준다 그 뒤 LRN[local response nomalization - 수렴속도 향상 ]을 시행한다. 그 뒤 max-pooling으로 2만큼 다운 샘플링을 한다 
결국 27x27사이즈의 특성맵에 64장으로 된다 

- 두번째 레이어(컨볼루션 레이어, conv2): 256개의 5x5x64 사이즈의 필터커널로 특성맵을 컨볼루션 해준다. stride = 1 , zero-padding = 2로 설정하며 max-pooling =2 로 하여 
256개의 13x13의 특성맵을 얻는다 


- 세번째 레이어(컨볼루션 레이어, conv3): 256개의 3 x 3 x 256 사이즈 필터커널로 특성맵을 컨볼루션해준다. 컨볼루션 보폭은 1로 설정하고, zero-padding은 1만큼 해준다. 결과적으로 13 x 13 사이즈의 256장의 특성맵이 산출된다. ReLU활성화 함수를 적용한다. 

 

- 네번째 레이어(컨볼루션 레이어, conv4): 256개의 3 x 3 x 256 사이즈 필터커널로 특성맵을 컨볼루션해준다. 컨볼루션 보폭은 1로 설정하고, zero-padding은 1만큼 해준다. 결과적으로 13 x 13 사이즈의 256장의 특성맵이 산출된다. ReLU활성화 함수를 적용한다. 

- 다섯번째 레이어[컨볼루션] : 256개의 3x3x256 사이즈 필터커널을 특성맵을 만들어준다. stride =1, zero-padding = 1로 하며 max-pooling = 2로 하여 
256장의 6x6사이즈의 특성맵을 갖는다.

- 여섯번째 레이어(fully-connected 레이어, full6): 전 층에서 생성된 6 x 6 x 256의 특성맵을 flatten 해준다. flatten이란 전 층의 아웃풋을 1차원의 벡터로 펼쳐주는 것을 의미한다. 따라서 6 x 6 x 256 = 9216개의 뉴런이 되고 full6층의 4096개 뉴런과 fully connected된다. 4096개의 뉴런으로 출력된 것을 ReLU 함수로 활성화해준다. 과적합을 막기 위해 full6에는 훈련시 dropout이 적용된다


- 일곱번째 레이어(fully-connected 레이어, full7): 4096개의 뉴런으로 구성되어 있다. full6의 4096개 뉴런과 fully-connected(전 층과 이번 층의 모든 뉴런이 서로 다 연결)되어 있다. 출력된 값을 역시 ReLU 함수로 활성화해준다. full7에는 훈련시 dropout이 적용된다.

- 여덟번째 레이어(fully-connected 레이어, full8): 1000개의 뉴런으로 구성되어 있다. full7의 4096개 뉴런과 fully-connected 되어 있다. 출력된 값은 softmax 함수로 활성화되어 입력된 이미지가 어느 클래스에 속하는지 알 수 있게 해준다. 


### 왜 VGG-F가 가장 빠른가 

- VGG-F는 첫번째 레이어에서 입력영상을 64개의 11 x 11 x 3 사이즈 필터커널로 보폭 4로 컨볼루션해주는 반면, VGG-M은 첫번째 레이어에서 입력영상을 96개의 7 x 7 x 3 사이즈 필터커널로 보폭 2로 컨볼루션해준다. 필터커널의 사이즈가 작을수록, 또한 보폭이 작을수록 연산량은 많아진다. 

- 이번에는 VGG-M과 VGG-S의 연산량을 비교해보자. 연산량에 가장 큰 차이를 유발하는 곳은 바로 두번째 레이어다. 두번째 레이어에서 VGG-S는 VGG-M이 보폭2로 컨볼루션을 진행하는 것과 달리 보폭1로 컨볼루션을 진행한다. 













