##ResNet

![ResNet](https://hoya012.github.io/assets/img/image_classification_guidebook/10.PNG)


- 위의 그림의 왼쪽 architecture는 ResNet-34의 구조를 나타내고 있습니다. ResNet은 3x3 conv가 반복된다는 점에서 VGG와 유사한 구조를 가지고 있습니다. Layer의 개수에 따라 ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152 등 5가지 버전으로 나타낼 수 있다.
Layer 개수를 많이 사용할수록 연산량과 parameter 개수는 커지지만 정확도도 좋아지는 효과를 얻을 수 있습니다.


![ResNet1](https://hoya012.github.io/assets/img/image_classification_guidebook/11.PNG)

- 다시 ResNet-34의 그림으로 돌아가면, 2개의 conv layer마다 옆으로 화살표가 빠져나간 뒤 합쳐지는 식으로 그려져 있습니다. 이러한 구조를 Shortcut 이라 부릅니다. 일반적으로 Shortcut으로는 identity shortcut, 즉 input feature map x를 그대로 output에 더해주는 방식을 사용합니다.

- ResNet으로 돌아오자. 층수에 있어서 ResNet은 급속도로 깊어진다. 2014년의 GoogLeNet이 22개 층으로 구성된 것에 비해, ResNet은 152개 층을 갖는다. 약 7배나 깊어졌다!

![ResNet](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fcx1l7G%2FbtqzR2RurjQ%2FuRBKXJoxhDZdBjqI2BqWnK%2Fimg.png)

### 1. residual block

- ResNet의 핵심인 Residual Block이 깊은 레이어층의 출현을 가능케 했다. 아래 그림에서 오른쪽이 Residual Block을 나타낸다. 기존의 망과 차이가 있다면 입력값을 출력값에 더해줄 수 있도록 지름길(shortcut)을 하나 만들어준 것 뿐이다. 

![ResNet3](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbFPOry%2FbtqzR2En9ry%2F2DTETgT1BkCrW74hKQCsrk%2Fimg.png)

- 기존의 신경망은 입력값 x를 타겟값 y로 매핑하는 함수 H(x)를 얻는 것이 목적이었다. 그러나 ResNet은 F(x) + x를 최소화하는 것을 목적으로 한다. x는 현시점에서 변할 수 없는 값이므로 F(x)를 0에 가깝게 만드는 것이 목적이 된다. F(x)가 0이 되면 출력과 입력이 모두 x로 같아지게 된다. F(x) = H(x) - x이므로 F(x)를 최소로 해준다는 것은 H(x) - x를 최소로 해주는 것과 동일한 의미를 지닌다. 여기서 H(x) - x를 잔차(residual)라고 한다. 즉, 잔차를 최소로 해주는 것이므로 ResNet이란 이름이 붙게 된다. 

#### Residual Network

- Residual learning은 기존의 stacking된 네트워크에 일종의 skip connection을 추가한 것을 말합니다. 위 그림에서 2개의 weight layer를 거친 것에 input을 그대로 더해주는 형태로 구성되어 있는데, 이를 Residual learning block이라고 부릅니다.
최종 배워야하는 것을 H(x) , Stacking된 Layer의 Output을 F(x), 그리고 Input을x 라고 하면
>> H(x) = F(x) + x

형태로 Block을 만든 것입니다. 즉 Skip Connection이 없었던 것에 비해서 F(x) 는 Input과의 '차이’만을 학습하면 되는 것이므로 이 때문에 Residual learning이라고 부르는 것 같습니다.

>>F(x) = H(x) − x : 이것이 잔차(Residual)이라한다. 


### ResNet의 구조 


