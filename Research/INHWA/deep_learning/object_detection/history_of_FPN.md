## FPN의 역사

#### 1.  Featurized Image Pyramid

- 이 방법은 각 레벨에서 독립적으로 특징을 추출하여 객체를 탐지하는 방법이다. 
- 연산량과 시간 관점에서 비효율적이며 현실에 적용하기 어렵다는 단점이 있다

![image1](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc9fbUC%2FbtquatVdynO%2Fy0sHZPytUMY1aMmQ2mZBE0%2Fimg.png)

#### 2. Single Feature Map

- 이 방법은 컨볼루션 레이어가 스케일 변화에 "로버스트" 하기 때문에 컨볼루션 레이어를 통해서 특징을 압축하는 방식이다. 하지만 멀티 스케일을 사용하지 않고 한번에 특징을 압축하여 마지막에 압축된 특징만을 사용하기 때문에 성능이 떨어진다는 평가가 있다. 

![image2](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCIL62%2Fbtquc2BRy6m%2FLHHSDkG2yTcJJShJxQq9h1%2Fimg.png)

#### 3. Pyramidal Feature Hierarchy

- 이는 서로 다른 스케일의 특징 맵을 이용하여 멀티 스케일 특징을 추출하는 방식이다.

- 각 레벨에서 독립적으로 특징을 추출하여 객체를 탐지하게 되는데,이는 이미 계산 되어 있는 상위 레벨의 특징을 재사용 하지 않는다는 특징이 있다. 이는 SSD에서 사용된 방식이다.

>> Feature Scaling 
>feature을 0~1 사이갑승로 정규화 시켜 정확한 분석이 가능하게 해주는것!

![image3](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdgMQdR%2FbtqubTSVsGf%2FUb7AYjHyJWz39ou6bISqhK%2Fimg.png)


#### 4. Feature Pyramid Network

- Top-down 방식으로 특징을 추출하며, 각 추출된 결과들인 low-resolution 및 high-resolution 들을 묶는 방식이다. 
- 각 레벨에서 독립적으로 특징을 추출하여 객체를 탐지하게 되는데 상위 레벨의 이미 계산 된 특징을 재사용 하므로 멀티 스케일 특징들을 효율적으로 사용 할 수 있다. 
- CNN 자체가 레이어를 거치면서 피라미드 구조를 만들고 forward 를 거치면서 더 많은 의미(Semantic)를 가지게 된다. 
- 각 레이어마다 예측 과정을 넣어서 Scale 변화에 더 강한 모델이 되는 것이다. 
- 이는 skip connection, top-down, cnn forward 에서 생성되는 피라미드 구조를 합친 형태이다. 
- forward 에서 추출된 의미 정보들을 top-down 과정에서 업샘플링하여 해상도를 올리고 forward에서 손실된 지역적인 정보들을 skip connection 으로 보충해서 스케일 변화에 강인하게 되는 것이다.

![image4](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F16xz2%2FbtqubeXA8WS%2FmQUOaaqCKPwUL5cVYDMl8k%2Fimg.png)

## Bi=FPN 

- BiFPN은 FPN에 레이어마다 가중치를 주어 좀더 각각의 층에 대한 해상도 정보가 잘 녹을 수 있도록 하는 장치이다

- 예를들어 컨볼루션 네트워크는 각각의 단계마다 ㅏ른 특징을 추출하게 되는데 FPN은 여러 해상도를 가지는 특징들을 전부 예측에 이용함으로써 정확도를 늘림 

- 모두 서로 다른 input feature들을 합칠때 구분없이 단순히 더하는 방식을 사용하고 있는데 그것을 지적함

- 서로 다른 input feature들은 해상도가 다르기 때문에 output feature에 기여하는 정도를 다르게 가져가야 험을 주장함  이것이 Bi-FPN 이다


 


 
 

