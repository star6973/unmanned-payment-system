## fast R-CNN
>> CNN 특징 추출부터 classification, bounding box regression 까지 모두 하나의 모델에서 학습시키자!

>![image1](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbDeWQr%2FbtqATgtSbW1%2FEUrHGvrUOFMa6rbQvrbkW1%2Fimg.png)

1. 먼저 전체 이미지를 미리 학습된 CNN을 통과시켜 피쳐맵을 추출합니다.

2. Selective Search를 통해서 찾은 각각의 RoI에 대하여 RoI Pooling을 진행합니다. 그 결과로 고정된 크기의 feature vector를 얻습니다.

3. feature vector는 fully connected layer들을 통과한 뒤, 두 개의 브랜치로 나뉘게 됩니다.

4-1. 하나의 브랜치는 softmax를 통과하여 해당 RoI가 어떤 물체인지 클래시피케이션 합니다. 더 이상 SVM은 사용되지 않습니다.

4-2. bouding box regression을 통해서 selective search로 찾은 박스의 위치를 조정합니다.
