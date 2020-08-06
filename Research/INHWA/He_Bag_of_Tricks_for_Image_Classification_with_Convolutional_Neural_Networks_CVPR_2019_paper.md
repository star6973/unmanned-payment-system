##내용 논문 정리 

1.최근 이미지 분류 연구에서 이루어진 많은 발전은 데이터 확대 및 최적화 방법의 변경과 같은 훈련 절차 개선의 결과일 수 있다.
2.본 논문에서는 그러한 개선사항의 집합을 검토하고 절제 연구를 통해 최종 모델 정확도에 미치는 영향을 경험적으로 평가할 것이다.
3.예를 들어, 이미지넷에서 ResNet-50의 상위 1 검증 정확도를 75.3%에서 79.29%로 높인다. 또한 이미지 분류 정확도의 향상이 객체 감지 및 의미 분할과 같은 다른 응용 분야에서의 전송 학습 성과 향상으로 이어진다는 것을 증명할 것이다.


###  Training Procedures

-랜덤한 이미지셋으로 그래디언트를 구한 뒤 네트워크 매개 변수를 업데이트 합니다다

###  Baseline Training Produre

1. 이미지 무작위로 샘플링하여 [0,255]의 픽셀값으로 디코딩합니다 
2. 랜덤비율 3:4, 4:3 로 짜른뒤에 [8%~100%]의 영역을 자른 뒤에 224x224정사각형 이미지로 조정합니다. 
3. 0.5 확률로 flip 한다
4. 색, 채, 명으로 변환이후에 [0.6~1.4]사이로 조정합니다
5. N(0,0.1)에 있는 데이터들을  PCA노이즈를 시킨다 

### Experiment Results
-1.3 1,200,000개의 이미지를 1000개의 클래스로 트레이닝한 뒤 결과를 보니 ResNet-50은 성능이 올라갔지만 Inception-V3
은 정확도가 조금 떨어졌다.


### Efficient Training 
-요즘은 GPU성능이 좋기 떄문에 배치 사이즈를 크게 돌리는것이 효과적이다

### Large-Batch training 

-배치사이즈가 크면 연산량이 많아 트레이닝 시간은 늘어나지만 정확도는 좋다. 
 연산량을 줄이려 배치사이즈를 줄이면 epochs를 늘릴때마다 효능이 떨어진다. 
 그래서 휴리스틱하게 잡는것이 중요하다.
 
a. Linear scaling learning rate
- 배치 크기에 따라 학습률을 선형적으로 증가시키는 것이 ResNet-50 교육에 경험적으로 효과가 있다고 보고한다. 
- 배치 크기 256에 대한 초기 학습 속도로 0.1을 선택한 다음, 더 큰 배치 크기 b로 변경할 때 초기 학습 속도를 0.1 × b/256으로 증가시킬 것이다.

b. Learning rate warmup.
-초기에 랜덤한 데이터를 사용하여  learning rate값을 작게하여 뺑뻉이 돌린뒤 weight가 안정적으로 바뀐뒤에는 lr 값을 높여서 학습시작을 한다 .

c. Zero 
- BN이 있는 모델에 있는 인수이다.Zero가 0이 아니면 역전파가되서 효율성이 떨어진다 

d.  no bias decay
- weight decay를 줄여 오버피팅을 예방한다. 

###Low-precision training
-소수점 float32자리를 float16으로 하면 속도가 더 빠르고 경우에 따라 정확도도 올라간다.

### Model Tweak
- 모델 변경은 convolution layer의 stride를 건드는 일이다. 
기존의 복잡도는 바꿀수 있으나 유의미한 성능 향상을 얻을 수도 있다.

### ResNet Architecture

- 컨볼루션 stride =2 , 풀링 stride = 2, 총 4배가 줄어든다. 그래서 인풋모델을 4배 줄이고 채널크기를 64까지 키운다
- 입력보다 4배가 더 커서 병목구조라고 한다 .

### ResNet Tweaks
- ResNet-b : downsampling의 기존 ResNet은 1x1의 stride가 2라 input데이터가 마니 훼손되는데 위치를 바꾸어 정의 훼손을 막는다 

- ResNet-D : ResNet-D는 ResNet-B의 영감을 받아 Path B의 1x1 conv , stride =2 로 손실되는걸 막는다 

- Result : ResNet-50 에 ResNet-B를 사용했을떄 정확도 0.5%가 증가했다.  
7x7conv를 3x3x3으로 바꿧더니 0.2% 더 좋아지고 ResNet-50 을 ResNet-D를 적용 시키면 1% 정확도 향상이있다. 하지만 ResNet-D는 연산처리량이 많아 속도가 3%가 느려졌다. 

### Label Smoothing 
- label에 0,1 classification이 아닌 확률로 1-e로 임의의값으로 뺴서 레이블에 노이즈를 넣는다 

###  Knowledge Distillation
- 미리 잘 학습된 큰 네트워크(Teacher network) 의 지식을 실제로 사용하고자 하는 작은 네트워크(Student network) 에게 전달하는 것
- 코스트는 (티처모델과 스튜던트 모델의 결과차이 ) + ( 실제물체와 스튜던트 모델의 차이)

