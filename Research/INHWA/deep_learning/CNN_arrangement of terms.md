### 1.2 채널, Channel
- 이미지 픽셀 하나하나는 실수입니다. 컬러 사진은 천연색을 표현하기 위해서, 각 픽셀을 RGB 3개의 실수로 표현한 3차원 데이터입니다.(<그림 2> 참조) 컬러 이미지는 3개의 채널로 구성됩니다. 반면에 흑백 명암만을 표현하는 흑백 사진은 2차원 데이터로 1개 채널로 구성됩니다. 높이가 39 픽셀이고 폭이 31 픽셀인 컬러 사진 데이터의 shape은 (39, 31, 3)3으로 표현합니다. 반면에 높이가 39픽셀이고 폭이 31픽셀인 흑백 사진 데이터의 shape은 (39, 31, 1)입니다.

![channel](https://taewanmerepo.github.io/2018/01/cnn/channel.jpg)

### 1.3 필터(Filter)[= kernel] & Stride
필터는 이미지의 특징을 찾아내기 위한 공용 파라미터입니다. Filter를 Kernel이라고 하기도 합니다. CNN에서 Filter와 Kernel은 같은 의미입니다. 필터는 일반적으로 (4, 4)이나 (3, 3)과 같은 정사각 행렬로 정의됩니다. CNN에서 학습의 대상은 필터 파라미터 입니다. <그림 1>과 같이 입력 데이터를 지정된 간격으로 순회하며 채널별로 합성곱을 하고 모든 채널(컬러의 경우 3개)의 합성곱의 합을 Feature Map로 만듭니다. 필터는 지정된 간격으로 이동하면서 전체 입력데이터와 합성곱하여 Feature Map을 만듭니다. <그림 3>은 채널이 1개인 입력 데이터를 (3, 3) 크기의 필터로 합성곱하는 과정을 설명합니다.

![filter](https://taewanmerepo.github.io/2018/01/cnn/conv.png)

필터는 입력 데이터를 지정한 간격으로 순회하면서 합성곱을 계산합니다. 여기서 지정된 간격으로 필터를 순회하는 간격을 Stride라고 합니다. <그림 4>는 strid가 1로 필터를 입력 데이터에 순회하는 예제입니다. strid가 2로 설정되면 필터는 2칸씩 이동하면서 합성곱을 계산합니다.

![filter2](https://taewanmerepo.github.io/2018/01/cnn/filter.jpg)

입력 데이터가 여러 채널을 갖을 경우 필터는 각 채널을 순회하며 합성곱을 계산한 후, 채널별 피처 맵을 만듭니다. 그리고 각 채널의 피처 맵을 합산하여 최종 피처 맵으로 반환합니다. 입력 데이터는 채널 수와 상관없이 필터 별로 1개의 피처 맵이 만들어 집니다. <그림 5 참조>

![filter3](https://taewanmerepo.github.io/2018/01/cnn/conv2.jpg)

하나의 Convolution Layer에 크기가 같은 여러 개의 필터를 적용할 수 있습니다. 이 경우에 Feature Map에는 필터 갯수 만큼의 채널이 만들어집니다. 입력데이터에 적용한 필터의 개수는 출력 데이터인 Feature Map의 채널이 됩니다.

Convolution Layer의 입력 데이터를 필터가 순회하며 합성곱을 통해서 만든 출력을 Feature Map 또는 Activation Map이라고 합니다. Feature Map은 합성곱 계산으로 만들어진 행렬입니다. Activation Map은 Feature Map 행렬에 활성 함수를 적용한 결과입니다. 즉 Convolution 레이어의 최종 출력 결과가 Activation Map입니다.

### 1.4 패딩(Padding)
Convolution 레이어에서 Filter와 Stride에 작용으로 Feature Map 크기는 입력데이터 보다 작습니다. Convolution 레이어의 출력 데이터가 줄어드는 것을 방지하는 방법이 패딩입니다. 패딩은 입력 데이터의 외각에 지정된 픽셀만큼 특정 값으로 채워 넣는 것을 의미합니다. 보통 패딩 값으로 0으로 채워 넣습니다.

![padding](https://taewanmerepo.github.io/2018/01/cnn/padding.png)

### 1.5 Pooling 레이어
풀링 레이어는 컨볼류션 레이어의 출력 데이터를 입력으로 받아서 출력 데이터(Activation Map)의 크기를 줄이거나 특정 데이터를 강조하는 용도로 사용됩니다. 플링 레이어를 처리하는 방법으로는 Max Pooling과 Average Pooning, Min Pooling이 있습니다. 정사각 행렬의 특정 영역 안에 값의 최댓값을 모으거나 특정 영역의 평균을 구하는 방식으로 동작합니다. <그림 7>은 Max pooling과 Average Pooling의 동작 방식을 설명합니다. 일반적으로 Pooing 크기와 Stride를 같은 크기로 설정하여 모든 원소가 한 번씩 처리 되도록 설정합니다.


![pooling](https://taewanmerepo.github.io/2018/02/cnn/maxpulling.png)



### 1.6ReLU 함수
- 활성화 함수로는 LeNet-5에서 사용되었던 Tanh 함수 대신에 ReLU 함수가 사용되었다. ReLU는 rectified linear unit의 약자이다. 같은 정확도를 유지하면서 Tanh을 사용하는 것보다 6배나 빠르다고 한다. AlexNet 이후에는 활성화함수로 ReLU 함수를 사용하는 것이 선호되고 있다. 

 
![relu](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcexrVz%2FbtqBFwoUz96%2F6E1W6ALGpm3EfkJykHPFak%2Fimg.jpg)


### 1.7 dropout
- 과적합(over-fitting)을 막기 위해서 규제 기술의 일종인 dropout을 사용했다. dropout이란 fully-connected layer의 뉴런 중 일부를 생략하면서 학습을 진행하는 것이다[5]. 몇몇 뉴런의 값을 0으로 바꿔버린다. 따라서 그 뉴런들은 forward pass와 back propagation에 아무런 영향을 미치지 않는다. dropout은 훈련시에 적용되는 것이고, 테스트시에는 모든 뉴런을 사용한다. 

![dropout]( https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcMcWkE%2FbtqBFNcRhiv%2FjJyZWvbf9uQLmKJG3pQAK1%2Fimg.jpg)



### 1.7 overlapping pooling
- CNN에서 pooling의 역할은 컨볼루션을 통해 얻은 특성맵의 크기를 줄이기 위함이다. LeNet-5의 경우 평균 풀링(average pooling)이 사용된 반면, AlexNet에서는 최대 풀링(max pooling)이 사용되었다. 또한 LeNet-5의 경우 풀링 커널이 움직이는 보폭인 stride를 커널 사이즈보다 작게 하는 overlapping pooling을 적용했다. 따라서 정확히 말하면 LeNet-5는 non-overlapping 평균 풀링을 사용한 것이고, AlexNet은 overlapping 최대 풀링을 사용한 것이다. 

![pooling](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb5hfOx%2FbtqBCUY3kpE%2FCKcK19bmDgtkSkWS5GPkBk%2Fimg.png)


### 1.8 local response normalization (LRN)
- 신경생물학에는 lateral inhibition이라고 불리는 개념이 있다. 활성화된 뉴런이 주변 이웃 뉴런들을 억누르는 현상을 의미한다. lateral inhibition 현상을 모델링한 것이 바로 local response normalization이다. 강하게 활성화된 뉴런의 주변 이웃들에 대해서 normalization을 실행한다. 주변에 비해 어떤 뉴런이 비교적 강하게 활성화되어 있다면, 그 뉴런의 반응은 더욱더 돋보이게 될 것이다. 반면 강하게 활성화된 뉴런 주변도 모두 강하게 활성화되어 있다면, local response normalization 이후에는 모두 값이 작아질 것이다. 


### 1.9 data augmentation
- 과적합을 막기 위해 dropout 말고도 또다른 방법이 사용되었다. 과적합을 막는 가장 좋은 방법 중 하나는 데이터의 양을 늘리는 것이다. 훈련시킬 때 적은 양의 데이터를 가지고 훈련시킬 경우 과적합될 가능성이 크기 때문이다. 따라서 AlexNet의 개발자들은 data augmentation이란 방법을 통해 데이터의 양을 늘렸다. 
쉽게 말해서 하나의 이미지를 가지고 여러 장의 비슷한 이미지를 만들어 내는 것이다. 이미지를 좌우 반전시키거나, AlexNet이 허용하는 입력 이미지 크기인 224 x 224 x 3보다 좀 더 큰 이미지를 조금씩 다르게 잘라서 224 x 224 x 3으로 만들어줘서 여러 장을 만들어낸다. 같은 내용을 담고 있지만 위치가 살짝 다른 이미지들이 생산된다. 




## 1.10 Batch Nomalization 

- 활성화 함수로 활성시킬때 출력값을 정규분포로 만들기위해 사용 
- 각 차원 단위로 평균과 분산을 구함 
- 각각 차원은 독립이다
- 배치 단위로 정규화 진행 

- 장점

1) gradient vanishing 문제를 매우 잘 해결한다.

2) Learning rate를 높게 주어도 학습을 잘하기 때문에 수렴속도가 빠름

3) 학습을 할대마다 출력값을 정규화로 만들어 주기 때문에, 초기화 전략에 영향을 덜 받음 (물론 어느정도 중요하지만)

4) 배치정규화(BN)이 regularizer의 역할을 한다.

+ 항상 그렇지는 않지만 Dropout을 제거해도 비슷한 성능을 보이고, 일반적으로 성능이 상승한다.

+ 배치 단위로 정규화를 하면서 모델에 노이즈를 주는 것으로 해석 가능. 이 노이즈로 인해 성능이 좋아짐

   (배치단위로 정규화를 하면 평균과 분산이 전체 평균의 분산과 값이 약간 다를 수 있다. 이는 일종의 노이즈가 됨)

- 단점
   테스트 단계에서는 배치 단위로 평균/분산을 구하기가 어려움

+ 이를 해결하기 위해서는?

1) 학습단계에서 배치단위의 평균/분산을 저장해놓기 (이동평균)

2) 테스트시에는 구해진 이동평균을 사용하여 정규화