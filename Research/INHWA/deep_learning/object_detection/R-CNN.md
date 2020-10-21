## R-CNN model 

![image1](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbJaTYc%2FbtqANCZbqeK%2FYilKOm42aNYvPcWIjYxCdK%2Fimg.png)

![image2](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbdmFi2%2FbtqAQ38E2v3%2FJMXznsWZsX3YQAuTkKtpWK%2Fimg.png)

R-CNN이 Object Detection을 수행하는 알고리즘은 다음과 같습니다.

1. 입력 이미지에 Selective Search 알고리즘을 적용하여 물체가 있을만한 박스 2천개를 추출한다.

2. 모든 박스를 227 x 227 크기로 리사이즈(warp) 한다. 이 때 박스의 비율 등은 고려하지 않는다.

3. 미리 이미지 넷 데이터를 통해 학습시켜놓은 CNN을 통과시켜 4096 차원의 특징 벡터를 추출한다.

4. 이 추출된 벡터를 가지고 각각의 클래스(Object의 종류) 마다 학습시켜놓은 SVM Classifier를 통과한다.

5. 바운딩 박스 리그레션을 적용하여 박스의 위치를 조정한다.


#### Region Proposal

- Region Proposal이란 주어진 이미지에서 물체가 있을법한 위치를 찾는 것입니다.

- R-CNN은 Selective Search라는 룰 베이스 알고리즘을 적용하여 2천개의 물체가 있을법한 박스를 찾습니다.

- Selective Search는 주변 픽셀 간의 유사도를 기준으로 Segmentation을 만들고, 이를 기준으로 물체가 있을법한 박스를 추론합니다.
![image3](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FSRNtz%2FbtqAPeQCKIU%2F1JsEHoX4e2bSAgzrgQQCD1%2Fimg.png)

하지만 R-CNN 이후 Region Proposal 과정 역시 뉴럴 네트워크가 수행하도록 발전하였습니다.


#### Feature Extraction


- Selective Search를 통해서 찾아낸 2천개의 박스 영역은 227 x 227 크기로 리사이즈 됩니다. (warp)

- 그리고 Image Classification으로 미리 학습되어 있는 CNN 모델을 통과하여 4096 크기의 특징 벡터를 추출합니다. 

저자들은 이미지넷 데이터(ILSVRC2012 classification)로 미리 학습된 CNN 모델을 가져온 다음, fine tune하는 방식을 취했습니다.

fine tune 시에는 실제 Object Detection을 적용할 데이터 셋에서 ground truth에 해당하는 이미지들을 가져와 학습시켰습니다.

그리고 Classification의 마지막 레이어를 Object Detection의 클래스 수 N과 아무 물체도 없는 배경까지 포함한 N+1로 맞춰주었습니다.

fine tune을 적용했을 떄와 하지 않았을 때의 성능을 비교해보면 아래와 같습니다.
![image4](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbklvjP%2FbtqAQl2Z3K3%2FaltDKimUjrdIaMiXocinv1%2Fimg.png)


>> 정리하자면, 미리 이미지 넷으로 학습된 CNN을 가져와서, Object Detection용 데이터 셋으로 fine tuning 한 뒤,selective search 결과로 뽑힌 이미지들로부터 특징 벡터를 추출합니다.


#### Classification

- CNN을 통해 추출한 벡터를 가지고 각각의 클래스 별로 SVM Classifier를 학습시킵니다. 주어진 벡터를 놓고 이것이 해당 물체가 맞는지 아닌지를 구분하는 Classifier 모델을 학습시키는 것입니다.


#### Non-Maximum Suppression


- SVM을 통과하여 이제 각각의 박스들은 어떤 물체일 확률 값 (Score) 값을 가지게 되었습니다. 그런데 2천개 박스가 모두 필요한 것일까요?

- 동일한 물체에 여러 개의 박스가 쳐져있는 것이라면,  가장 스코어가 높은 박스만 남기고 나머지는 제거해야합니다.

- 이 과정을 Non-Maximum Supperssion이라 합니다.
![image5](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fpu1Jo%2FbtqANDX2WUQ%2FdB9pDakTtO57zjZa0CLsa1%2Fimg.png)
![image6](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FJGbNK%2FbtqAQ2Pr0yo%2FJ5zFOBxdpX1lZBxKuWk8mk%2Fimg.png)
>> 논문에서는 IoU가 0.5 보다 크면 동일한 물체를 대상으로 한 박스로 판단하고 Non-Maximum Suppression을 적용합니다.