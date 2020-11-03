#### 전처리 

- face detection 다음 crop, 그리고 정면을 보게하기위해 face alignment을 하여 프로세싱합니다. 

- the gallery = train dataset 

- the proble = test dataset 

- identification = One-to-many , ID

- Verification = one-to-one , Yes/No

- Closed-set : 입력된 인풋 데이터가 라벨에 있으면 라벨로 추측

- Opened-set : 입력된 인풋 데이터가 라벨에  없으면 feature extractor로 특징 추출을 하여 유클리디언 거리로 참과 거짓을 판별한다. 

## summary 
- faceNet은 align을 하지 않는다. 

- identification loss 를 사용하지 않는다. 100% verification loss로 metric loss[거리]를 구한다. (metric learning은 특징 추출로 similarity를 비교한다 . 그것에 가장 대표적인 loss는 유클리드 loss이다.)

- 같은 카테로리의 사진들의 거리가 작았으면 좋겟다는것이 metric loss로 구현한다. 

- L2 imbedding = 마지막에 나오는 특징 벡터가  1로 나오게 하는것 

-  임베딩을 하여 verification 과 클러스터를 한번에 할수있다 .

- verification 문제는 threshould로 문제를 해결한다. 같은 사람인지 판별이기에 그 카테고리의 사진의 거리를 지정해줘서 원 안에 들어가있는 사람인지 아닌지를 판별한다.

- identification 문제는 새로 들어온 얼굴이 누군지를 판별하기위해 k-최근접 이웃(k nearest neighbors)을 사용해서 문제를 해결한다. 


# Background
#### Triplet network
Triplet Network는 동일한 네트워크 3개와, 기준이 되는 클래스에 속하는 샘플 x와, 다른 클래스에 속하는 negative sample, 같은 클래스에 속하는 positive smaple로 구성된다.
일반적은 손실함수가 한 개의 input sample을 받는데 비해 삼중항 손실 함수는 위의 3개의 입력을 받으며, loss또한 3개의 입력에 대해 계산한다.
![omg3](https://media.vlpt.us/images/chy0428/post/8e358db8-2793-43f7-9a40-dc77b9af6c17/image.png)

#### Triplet loss

Triplet loss는 anchor sample, positive sample, negative sample, 3개의 샘플에 대해 loss 계산을 수행한다.
앞에서 말했던 저차원의 벡터를 학습 시키는 방법으로 사용한다. 같은 사진은 가까운 거리에, 다른 사진 먼 거리에 있도록 유사도를 벡터 사이의 거리와 같아지게 하려는 목적이다. 이를 loss로 나타낸다.

![omg5](https://media.vlpt.us/images/chy0428/post/76de7aa3-3f0c-4aea-ba75-bb4146643575/image.png)

같은 사진을 가깝게, 다른 사진을 멀게 하는 것을 표현한 식이 다음과 같다.


![omg6](https://media.vlpt.us/images/chy0428/post/5da861f8-3f4d-44de-b211-cf7d55d17d39/image.png)

>> α는 positive와 negative 사이에 주고 싶은 margin을 의미한다고 생각하면 된다.

#### Triplet selection


![omg7](https://media.vlpt.us/images/chy0428/post/5da861f8-3f4d-44de-b211-cf7d55d17d39/image.png)

위의 식을 만족하는 triplet을 만들면서 학습을 진행할 때, 완전 다른 사진일 때는 너무 쉽게 만족하는 경우가 많을 것이다. 이런 경우, 학습이 제대로 되지 않는 문제가 발생한다. 따라서 잘 구분하지 못하는 사진을 넣어, 위의 식을 만족하지 않는 triplet을 만들어야 한다.
따라서 최대한 먼 거리에 있는 Positive를 고르고, 최대한 가까운 거리에 있는 negative를 골라야 한다.

![omg9](https://media.vlpt.us/images/chy0428/post/0f205412-1053-4e5c-94b3-955a68be19b4/image.png)

이를 각각 hard positive, hard negative로 표현한다. 하지만, 전체 데이터에서 각각 hard point들을 찾아야 한다고 할 때, 계산량이 많아져 시간이 많이 필요하며, 비효율적이고, 오버피팅이 생길 수 있다는 문제가 발생한다. 따라서, 이것을 해결하기 위해 이 논문에서 mini batch 안에서 hard point를 찾도록 하는 방법을 제시한다. 이 때, hard positive를 뽑는 것보다 모든 anchor-positive쌍을 학습에 사용하며, hard negative를 뽑을 때는 다음과 같은 식을 만족하는 x중에 뽑는 것이 좋은 성능을 보였다고 한다.

>> 이러한 negative examplars를 semi-hard라고 부른다.
이는 anchor-positive보다 anchor-nagative간 거리가 더 멀긴 가지만 margin이 충분히 크지 않은 nagative를 뽑는 것이다.




#### Metric Learning
metric learning은 Input image를 이용하여 feature extraction을 학습함으로 기하학적 거리가 시맨틱 거리에 근접한 새로운 feature space로 바꾼다. 즉, input image space에서 data들에 가장 적합한 형태의 어떤 metric을 learning하는 알고리즘이다. 여기에서 data는 각 pair 별로 유사도가 정의되어 있는 형태의 데이터이다. metric learning은 유사한 데이터들끼리는 가까운 거리로, 유사하지 않은 데이터들끼리 더 먼 거리로 판단하게 하는 어떤 metric을 학습하는 것이다.

![omg](https://media.vlpt.us/images/chy0428/post/48090770-d7c1-4fe0-b10a-422b9f1b6040/image.png)

>>기존의 Image space에서는 pixel by pixel로 비교하기 때문에, 기하학적 거리와 시맨틱 거리가 일치하지 않는다. 예를 들어, 인물 사진 2장을 비교했을 때, 동일 인물이면 거리가 가까워야 하는데, 인물이 입고 있는 옷이나 배경에 따라 그 거리가 달라지기 때문에 두 공간상의 거리가 일치하지 않게된다.
하지만, Fearture extraction을 학습시키면, feature space에서의 기하학적 거리는 시맨틱 거리에 근접하게 되어, 굉장히 좋은 성질을 갖게 된다. 이는 하나의 sample을 골라, 대표 sample과의 거리만 재면 어떤 클래스인지 파악 할 수 있게 되는 것을 의미한다.

Feature Extraction이란, 고차원의 원본 feature 공간을 저차원의 새로운 feature 공간으로 투영시킨다. 새롭게 구성된 feature 공간은 보통은 원본 feature 공간의 선형 또는 비선형 결합이다.

특징 추출기가 추출한 특징 벡터 간 거리가 입력 패턴 간의 의미적 거리와 비례하도록 학습하면, 동일 클래스의 샘플들이 특징 공간 상에서 유사한 위치에 배치되도록 할 수 있다.

![omg1](https://media.vlpt.us/images/chy0428/post/bdef24e5-79e7-4ba5-a05f-26cd2ff5cb29/image.png)


# FaceNet 


![img](https://github.com/HwangToeMat/HwangToeMat.github.io/blob/master/Paper-Review/image/FaceNet/img6.jpg?raw=true)

FaceNet은 각각의 얼굴 이미지를 128차원으로 임베딩하여 유클리드 공간에서 이미지간의 거리(까울 수록 유사도가 높음)를 통해 분류하는 모델이다.

간단히 말하면 얼굴 사진에서 그 사람에 대한 특징값을 구해주는 모델이고, 그 값을 활용하여 값들간의 거리를 통해 이미지에 대한 identification, verification, clustering을 할 수 있게 되는 것이다.

![img2](https://github.com/HwangToeMat/HwangToeMat.github.io/blob/master/Paper-Review/image/FaceNet/img7.jpg?raw=true)

이때 triplet loss를 사용한 Metric Learning으로 모델을 학습하였는데 이는 매우 중요하기 때문에 아래에서 더 자세히 다루어 보도록 하겠다.

추가적으로 기존의 모델들이 2D나 3D로 aligned된 얼굴 이미지를 필요로 하였던 것에 비해 FaceNet은 그러한 과정없이 높은 성능이 나왔다는 점 또한 모델 설계가 얼마나 정교하게 이루어졌는지 나타내는 요소 중 하나라고 생각한다

## Triplet Loss를 사용한 Metric Learning

![img3](https://github.com/HwangToeMat/HwangToeMat.github.io/blob/master/Paper-Review/image/FaceNet/img1.jpg?raw=true)

FaceNet은 학습과정에서 Meric Learning[유사도]을 하기위해 Triplet Loss를 사용했다. 학습시 미니배치안에서 어떠한 사람(Anchor)에 대해 같은 사람(Positive)와 다른 사람(Negative)를 지정해 놓는다. 그리고 임베딩된 값들의 유클리드 거리를 구해 그림과 같이 Anchor와 Positive의 거리는 가까워지고 Negative와의 거리는 멀어지도록 학습을 시켰고 이러한 과정을 triplet loss로 구현하였다.

![img4](https://github.com/HwangToeMat/HwangToeMat.github.io/blob/master/Paper-Review/image/FaceNet/img2.jpg?raw=true)

즉, 앞서말한 과정을 식으로 나타내면 위와 같다. 대괄호 안의 첫 번째 항이 의미하는 것은 Positive와의 거리이고 두 번째 항은 Negative와의 거리이며 alpha는 마진을 의미한다. 따라서 L을 최소화한다는 의미는 Positive의 거리는 가까워지도록 하고 Negative와의 거리는 멀어지도록 한다는 것이다.

하지만 모델의 성능을 높이기 위해서는 Hard Positive(같은 사람이지만 다르게 보이는 사람)과 Hard Negative(다른 사람이지만 닮은 사람)와 같이 학습을 방해하는 요소를 제어할 필요가 있었고, 이러한 문제를 해결하기위해 아래와 같은 식을 사용하였다.

![img5](https://github.com/HwangToeMat/HwangToeMat.github.io/blob/master/Paper-Review/image/FaceNet/img3.jpg?raw=true)

Hard Positive는 위의 첫 번째 식과같이 나타낼 수 있고, Hard Negative는 두 번째 식과 같이 나타낼 수 있는데, 이 모델에서는 Hard Positive는 전부 학습을 진행하였지만 Hard Negative는 세 번째 식을 만족할 경우에만 학습을 진행하였다.


## 모델 구조

![img6](https://github.com/HwangToeMat/HwangToeMat.github.io/blob/master/Paper-Review/image/FaceNet/img8.jpg?raw=true)

위의 그래프는 FaceNet의 여러 구조들이 갖는 성능을 비교한 그래프이다. 각각의 특징은 오른쪽에 나와있는 표와 같다. 그 중 가장 성능이 좋아 보이는 NN2(Inception 224x224)의 자세한 구조를 살펴보면 아래와 같다.

![img7](https://github.com/HwangToeMat/HwangToeMat.github.io/blob/master/Paper-Review/image/FaceNet/img9.jpg?raw=true)



