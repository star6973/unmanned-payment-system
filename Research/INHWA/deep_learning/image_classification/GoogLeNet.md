## GoogLeNet

![Goog:LeNet](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FIq9NO%2FbtqyPWk5PBX%2FK2JicGjIjj5w0eFIbhx4bK%2Fimg.png)

- GoogLeNet은 상술한 바와 같이 22개 층으로 구성되어 있다. 파란색 블럭의 층수를 세보면 22개 층임을 알 수 있다. 이제 GoogLeNet의 특징들을 하나하나 살펴보자. 

### 1 x 1 컨볼루션
- 먼저 주목해야할 것은 1 x 1 사이즈의 필터로 컨볼루션해주는 것이다. 구조도를 보면 곳곳에 1 x 1 컨볼루션 연산이 있음을 확인할 수 있다.

![1x1](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FMzPze%2FbtqyQy5e3NM%2F5HPtmAwVQzKJTj6wgWautk%2Fimg.png)


- 1 x 1 컨볼루션은 어떤 의미를 갖는 것일까? 왜 해주는 것일까? GoogLeNet에서 1 x 1 컨볼루션은 특성맵의 갯수를 줄이는 목적으로 사용된다. 특성맵의 갯수가 줄어들면 그만큼 연산량이 줄어든다.

 

- 예를 들어, 480장의 14 x 14 사이즈의 특성맵(14 x 14 x 480)이 있다고 가정해보자. 이것을 48개의 5 x 5 x 480의 필터커널로 컨볼루션을 해주면 48장의 14 x 14의 특성맵(14 x 14 x 48)이 생성된다. (zero padding을 2로, 컨볼루션 보폭은 1로 설정했다고 가정했다.) 이때 필요한 연산횟수는 얼마나 될까? 바로 (14 x 14 x 48) x (5 x 5 x 480) = 약 112.9M이 된다. 

 

- 이번에는 480장의 14 x 14 특성맵(14 x 14 x 480)을 먼저 16개의 1 x 1 x 480의 필터커널로 컨볼루션을 해줘 특성맵의 갯수를 줄여보자. 결과적으로 16장의 14 x 14의 특성맵(14 x 14 x 16)이 생성된다. 480장의 특성맵이 16장의 특성맵으로 줄어든 것에 주목하자. 이 14 x 14 x 16 특성맵을 48개의 5 x 5 x 16의 필터커널로 컨볼루션을 해주면 48장의 14 x 14의 특성맵(14 x 14 x 48)이 생성된다. 위에서 1 x 1 컨볼루션이 없을 때와 결과적으로 산출된 특성맵의 크기와 깊이는 같다는 것을 확인하자. 그럼 이때 필요한 연산횟수는 얼마일까? (14 x 14 x 16)*(1 x 1 x 480) + (14 x 14 x 48)*(5 x 5 x 16) = 약 5.3M이다. 112.9M에 비해 훨씬 더 적은 연산량을 가짐을 확인할 수 있다. 연산량을 줄일 수 있다는 점은 네트워크를 더 깊이 만들수 있게 도와준다는 점에서 중요하다. 

![1x1conv](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbt4AxN%2FbtqyQ6NHO6u%2FezqfgBmWfkN5N2C49icbR1%2Fimg.png)


### Inception 모듈

![inception](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbHZHKC%2FbtqyQ5aekdF%2F3rkScmoIxS4P4fia96lQwk%2Fimg.png)

GoogLeNet은 총 9개의 인셉션 모듈을 포함하고 있다. 

![inception1](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F14Um2%2FbtqyQ5nKlEA%2FhjSsZaYiBukseySytXWFCK%2Fimg.png)

- GoogLeNet에 실제로 사용된 모듈은 1x1 컨볼루션이 포함된 (b) 모델이다. 아까 살펴봤듯이 1x1 컨볼루션은 특성맵의 장수를 줄여주는 역할을 한다. 노란색 블럭으로 표현된 1x1 컨볼루션을 제외한 나이브(naive) 버전을 살펴보면, 이전 층에서 생성된 특성맵을 1x1 컨볼루션, 3x3 컨볼루션, 5x5 컨볼루션, 3x3 최대풀링해준 결과 얻은 특성맵들을 모두 함께 쌓아준다. AlexNet, VGGNet 등의 이전 CNN 모델들은 한 층에서 동일한 사이즈의 필터커널을 이용해서 컨볼루션을 해줬던 것과 차이가 있다. 따라서 좀 더 다양한 종류의 특성이 도출된다. 여기에 1x1 컨볼루션이 포함되었으니 당연히 연산량은 많이 줄어들었을 것이다. 

###  global average pooling

- AlexNet, VGGNet 등에서는 fully connected (FC) 층들이 망의 후반부에 연결되어 있다. 그러나 GoogLeNet은 FC 방식 대신에 global average pooling이란 방식을 사용한다. global average pooling은 전 층에서 산출된 특성맵들을 각각 평균낸 것을 이어서 1차원 벡터를 만들어주는 것이다. 1차원 벡터를 만들어줘야 최종적으로 이미지 분류를 위한 softmax 층을 연결해줄 수 있기 때문이다. 만약 전 층에서 1024장의 7 x 7의 특성맵이 생성되었다면, 1024장의 7 x 7 특성맵 각각 평균내주어 얻은 1024개의 값을 하나의 벡터로 연결해주는 것이다.

![global_average_pooling](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbwTHh0%2FbtqB2uyArWI%2FqBr48Ik8bl4bK1oOEJa3bk%2Fimg.png)

- 이렇게 해줌으로 얻을 수 있는 장점은 가중치의 갯수를 상당히 많이 없애준다는 것이다. 만약 FC 방식을 사용한다면 훈련이 필요한 가중치의 갯수가 7 x 7 x 1024 x 1024 = 51.3M이지만 global average pooling을 사용하면 가중치가 단 한개도 필요하지 않다.

### auxiliary classifier

- 네트워크의 깊이가 깊어지면 깊어질수록 vanishing gradient 문제를 피하기 어려워진다. 그러니까 가중치를 훈련하는 과정에 역전파(back propagation)를 주로 활용하는데, 역전파과정에서 가중치를 sigmoid로 업데이트하다보면 사용되는 gradient가 점점 작아져서 0이 되어버리는 것이다. 따라서 네트워크 내의 가중치들이 제대로 훈련되지 않는다. 이 문제를 극복하기 위해서 GoogLeNet에서는 네트워크 중간에 두 개의 보조 분류기(auxiliary classifier)를 달아주었다. 

![auxiliary](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbD5poT%2FbtqyQM98EkX%2FnbxasUSmCO1WnaIyIsvUD0%2Fimg.png)

-보조 분류기의 구성을 살펴보면, 5 x 5 평균 풀링(stride 3) -> 128개 1x1 필터커널로 컨볼루션 -> 1024 FC 층 -> 1000 FC 층 -> softmax 순이다. 이 보조 분류기들은 훈련시에만 활용되고 사용할 때는 제거해준다. 