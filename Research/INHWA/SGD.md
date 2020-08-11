##Stochastic gradient descent(SGD)


추출된 데이터 한 개에 대해서 error gradient 를 계산하고, Gradient descent 알고리즘을 적용하는 방법.

모델의 레이어 층은 하나의 행렬곱으로 생각할 수 있고, 여러개의 묶음 데이터는 행렬이라고 생각 할 수 있다.

즉, 여러개의 묶음 데이터를 특정 레이어 층에 입력하는 것은 행렬 x 행렬로 이해할 수 있는데,

SGD는 입력 데이터 한 개만을 사용하기 때문에 한 개의 데이터를 '벡터' 로 표현하여 특정 레이어 층에 입력하는 것으로 이해할 수 있고 이는 벡터 x 행렬 연산이 된다.


![momentum](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile3.uf.tistory.com%2Fimage%2F996A6C395D0E18BB21AF72) 


##Mini-batch gradient descent(MSGD)

전체 데이터셋에서 뽑은 Mini-batch 안의 데이터 m 개에 대해서 각 데이터에 대한 기울기를 m 개 구한 뒤, 그것의 평균 기울기를 통해 모델을 업데이트 하는 방법이다.



다시 간단하게 요약하면, BGD 와 SGD 의 장점만 빼먹고 싶은 알고리즘. 전체 데이터 셋을 여러개의 mini-batch 로 나누어, 한 개의 mini-batch 마다 기울기를 구하고 모델을 업데이트 하는 것.



예를들어, 전체 데이터가 1000개인데 batch size 를 10으로 하면 100개의 mini-batch 가 생성되는 것으로, 이 경우 100 iteration 동안 모델이 업데이트 되며 1 epoch 이 끝난다.



![momentum](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile26.uf.tistory.com%2Fimage%2F99BE27485D0F73FD2BAD11)

 
