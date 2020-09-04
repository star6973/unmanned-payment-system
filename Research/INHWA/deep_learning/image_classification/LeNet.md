### LeNet-5 , 1998

- CNN의 최초로 제안한 논문인 LeNet-5입니다

![LeNet](https://hoya012.github.io/assets/img/image_classification_guidebook/1.PNG)

- 1. 인풋 이미지를 convolution 으로 feature map을 만든다 [32x32]
[ convolution의 훈련해야할 파라미터 개수: (가중치*입력맵개수 + 바이어스)*특성맵개수 =  (5*5*1 + 1)*6 = 156]
- 2. feature map을 subsampling(maxpooling)을 한다.  [ feature map의 모든값을 알필요는 없다. 그래서 마스크에 가장 특징이 되는 값을 빼오는것을 subsampling이라 한다.] [ 28x28]
[resampling의 훈련해야할 파라미터 개수: (가중치 + 바이어스)*특성맵개수 = (1 + 1)*6 = 12]
- 3. subsampling을 한 map을 컨볼루션을 한다. [14x14]
- 4. 다시 convolution [10x10]
- 5. 다시 subsampling [5x5]
- 6. 특징점들의 사이즈를 줄이며 샘플링한 값들을 1자로 배열시킨다 [ full connection ] [ 1x120] 
- 7. 특징점들의 사이즈를 줄이며 샘플링한 값들을 1자로 배열시킨다 [ full connection ] [ 1x120] 
[full connection의 훈련해야할 파라미터 개수: 연결개수 = (입력개수 + 바이어스)*출력개수 = (120 + 1)*84 = 10164]
- 8. output layer 

- LeNet-5를 제대로 가동하기 위해 훈련해야할 파라미터는 총 156 + 12 + 1516 + 32 + 48120 + 10164 = 60000개다. 

>> CNN의 특징 = input을 1차원적으로 바라보던 관점에서 2차원으로 확장하였고, parameter sharing 을 통해 input의 픽셀수가 증가해도 파라미터가 변하지 않는다.

- 1990년대 당시에는 지금 수준의 컴퓨팅 파워에 비하면 현저히 연산 자원이 부족했기 때문에 32x32라는 작은 size의 image를 input으로 사용하고 있습니다. Layer의 개수도 Conv layer 2개, FC layer 3개로 굉장히 적은 개수의 layer를 사용하고 있습니다.

