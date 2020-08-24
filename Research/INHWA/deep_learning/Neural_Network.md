
ex) 이미지

- 가중치 = 레이어가 읽으려는 패턴에 얼마나 잘 부합하냐에 따라서 값이 결정된다.
 당연히 그 패턴에 잘 부합하면 weight값이 커지고 패턴에 맞지 않는 입력이면 weight값이 작다
 
 - bias = 이것은 Wx의 threshold의 역할을 한다. 뉴런이 활성화가 되기위한 가장 작은 값이다. 어느 임계부터 값이 나오도록 도와주는 값이다 .
 
 - ex)MNIST에서  hidden layer의 갯수가 2개면 weight와 bias는 784*16 + 16*16 + 16*10 = 13000여개가 있다
  
- a1 = Wa0 + b 
- W = 다음 레이어에 가기위한 모든 웨이트 나뭇가지(784*16)
- a0 = 입력단의 갯수 (784개)
- b = 다음 레이어의 뉴런이 활성화할지 판단하는 쓰레숄드값(784개)


 결론은 컴퓨터가 실제로 해당 문제를 스스로 해결하기 위해서는 수많은 수치들을 얻는것입니다 
 
 
### 신경망 
- input = 784 [ 28 x 28]
- output = 10 number
- parameter = 13000 [weight/ bias]

### Cost function
- input = 13000 [weight/ bias]
- output = 1 number [ the cost ]
- parameter[매개변수] = many many many training example


- 