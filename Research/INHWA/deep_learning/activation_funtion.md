##액티베이션 함수


###linear Separable[선형 분리]
 - linear Separable 이 선이 선형분리로 나눌수있다 하지만 여기서 linear라는 개념은 일직선이라서 리니어지만 그 기준은 WX + b 에서 W이다. 
 w tern이 리니어하다 = 기울기가 w이다. 만약 wx^2+b이면 그 그래프의 x축은 x^2이다.  그래서 무작위로 있는 데이터들을 비선형으로 데이터들을 마음대로 내 입맛대로 움직이게한다. 
 공간을 접는다라고 말하기도 한다 . 
 
- 선형 분리만으로는 데이터의 분리에 한계를 느껴 비선형으로 액티베이션 함수를 사용한다.

 

###시그모이드 함수 (Sigmoid)

- 시그모이드 함수는 Logistic 함수라 불리기도한다. 선형인 멀티퍼셉트론에서 비선형 값을 얻기 위해 사용하기 시작했다. 


![sig](https://mlnotebook.github.io/img/transferFunctions/sigmoid.png) 

- 우선 함수값이 (0, 1)로 제한된다.
- 중간 값은 12이다.
- 매우 큰 값을 가지면 함수값은 거의 1이며, 매우 작은 값을 가지면 거의 0이다.

###Gradient Vanishing
- 현상이 발생한다. 미분함수에 대해 x=0에서 최대값 1/4 을 가지고, input값이 일정이상 올라가면 미분값이 거의 0에 수렴하게된다. 이는 |x|값이 커질 수록 Gradient Backpropagation시 미분값이 소실될 가능성이 크다.



###ReLU 함수 (Rectified Linear Unit)


![sig](https://mlnotebook.github.io/img/transferFunctions/relu.png) 

ReLU 함수의 특징을 살펴보자.

- x>0 이면 기울기가 1인 직선이고, x<0이면 함수값이 0이된다.
- sigmoid, tanh 함수와 비교시 학습이 훨씬 빨라진다.
- 연산 비용이 크지않고, 구현이 매우 간단하다.
- x<0인 값들에 대해서는 기울기가 0이기 때문에 뉴런이 죽을 수 있는 단점이 존재한다.