## 알파벳에 숨겨져 있는 숫자 찾기

#### 1일차
pytorch로 CNN 사용해서 훈련해보기 --> 0.3

#### 2일차
train, val으로 나눠서 훈련해보기 --> 0.5

#### 3일차
alphabet도 쪼개서 훈련해보기 --> 0.65, 0.71

#### 4, 5일차
Data Augmentation 적용해보기 --> 0.72
Convolution 쌓기 --> 0.86, 0.87, 0.88

- 쌓는 방법
    <center><img src="/Daicon/MH.Ji/1.PNG" width="70%"></center><br>

    + 위의 그림을 예시로 들며, EMNIST의 이미지의 크기는 28x28이다.
    + layer를 쌓을 때마다, input과 output layer의 계산은 비교적 쉬운 방법이 있다. 아래의 pytorch documentation에 들어가보면 자세히 설명되어 있다.

    [CONV1D](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)  
    [CONV2D](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

    + 간단히 식을 설명하자면, 다음 노드의 입장에선 이전 노드의 output이 들어오는 채널값이다.

    + (channel + 2 x padding - dilation x (kernel size - 1) - 1) / stride + 1

    + channel과 padding, kernel size는 보통 사용자가 지정해주기 때문에 대입만 하면 된다.

    + dilation은 default가 1이다.

    + stride가 어떻게 작용하는지 CNN을 이해해보면 왜 나누는지 알 수 있다.

    + 따라서 위의 그림으로 다시 돌아가서, Letter의 Convolution만 계산해보자.
        + Letter는 A-Z까지의 문자이므로, 26의 크기인 1채널이다. 따라서 첫 번째 conv 레이어는 입력으로 1을 받게 되고, 16 채널로 만들어준다. 그리고 kernel 사이즈는 3이고, padding은 1로 주어진다.
        
        + 이 conv 레이어를 통과하면(출력되는 채널만 계산해주면 됨), (26 + 2 x 1 - 1 x (3 - 1) - 1) / 1 + 1 = 26이 된다. 다음 conv 레이어를 계속 통과해보면 각각 25, 25, 26, 24로 줄어드는 것을 확인할 수 있다.

    + 이번에는 Image의 Convolution을 계산해보자.
        + Image는 28x28 사이즈의 흑백 채널인 1채널이다.

        + 마찬가지고 conv 레이어를 계속 통과해보면 각각 28, 28, 28, 26, 26, 26, 26, 26이 될 것이다.

    + 위에서 구한 채널들을 이제 Fully Connected Layer에 대입해줘야 하기 때문에 1차원으로 만들어준다. Letter와 Image는 개별적인 convolution block이기 때문에, 각각 16@24, 32@26x26이 된다.

    + 즉, 16x24 + 32x26x26 = 22016 채널이 된다.



#### 5일차
ResNet으로 훈련해보기