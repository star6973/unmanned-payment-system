
###Stochastic gradient descent(SGD)

- 엄청나게 많은 모든 데이터의 가중치을 조절하며 다보고 가는것보다 미니배치의 cost를 구하고 learning rate를 곱해 그 방향으로 가는 것을 반복하여 계산횟수가 줄어든다

![SGD](https://t1.daumcdn.net/cfile/tistory/996AFC3C5B0CF0C901) 

###Momentum(모멘텀)

- 이전에 업데이트된 량을 남겨두고 모멘텀을 더해준다 그래서 연산 수행 반복 횟수를 줄어든다 . 
경사의 로스는 로컬 미니멈에 멀수록 크지만 가까워질수록 엄청나게 작아진다 그래서 미니배치를 통해 빠르게 다가가서 연산 수행 횟수를 줄어들게한다.


![SGD](https://t1.daumcdn.net/cfile/tistory/99A14F455B0CF54C21) 
 
m의 정확한 용어는 아니지만 저희는 그냥 모멘텀(운동량) 또는 모멘텀 계수라고 부릅니다. 보통 0.9로 설정하며 교차 검증을 한다면 0.5에서 시작하여 0.9, 0.95, 0.99 순서로 증가시켜 검증합니다

### Nesterov Accelrated Gradient(NAG, 네스테로프 모멘텀)

- 모멘텀 방향과 이전의 그래디언트 방향 , 현재의 그래디언트 방향 + 모멘트 방향을 더해서 빠르게 다가간다.
미리 예측을 해서 더 미리 빠른 효과를 얻는다

![NAG](https://t1.daumcdn.net/cfile/tistory/996E494B5B0D03A003)
![NAG-equation](https://t1.daumcdn.net/cfile/tistory/99527A335B0CFF2E2B)

