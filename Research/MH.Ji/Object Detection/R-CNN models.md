# R-CNN부터 Mask R-CNN까지..
## 1. R-CNN
- 구조
    <center><img src="/reference_image/MH.Ji/R-CNN models/1.PNG" width="70%"></center><br>

- 동작 방식
    + input image에 bounding box를 그리고, selective search 기법을 통해 RoI(Region Of Interest)를 약 2,000개 정도 추출한다.
    + 추출된 RoI 조각들을 동일한 크기로 만들어서, 각각을 CNN(pretrained된 AlexNet)의 끝 단을 object detection class로 fine tuning시켜서 훈련을 한다.
    + 훈련 결과 나온 feature들을 SVM에서 classification, bounding regression으로 back propagation 업데이트를 해준다.

    <center><img src="/reference_image/MH.Ji/R-CNN models/2.PNG" width="70%"></center><br>

- Selective Search

    <center><img src="/reference_image/MH.Ji/R-CNN models/3.PNG" width="70%"></center><br>

- Region Proposal
    + 주어진 이미지에서 물체가 있을법한 위치를 찾는 것이다.
    + selective search라는 알고리즘을 적용하여 2,000개의 물체가 있을만한 박스를 찾는다.
    + selective search는 주변 픽셀간의 유사도를 기준으로 merge를 통해 segmentation을 만들고, 이를 기준으로 물체가 있을만한 박스를 추론한다.

- Feature Extraction
    + ImageNet 데이터로 미리 학습된 CNN 모델(AlexNet)을 가져와 fine tune하는 방식을 취했다.
    + fine tune시에는 실제 object detection을 적용할 데이터셋에서 ground truth에 해당하는 이미지들을 가져와 학습시킨다.
    + classification 마지막 레이어를 object detection의 클래스 수 N과 아무 물체도 없는 배경까지 포함한 N+1로 맞춰준다.
    + 이후 selective search 결과로 뽑힌 이미지들로부터 feature vector를 추출한다.

- Classification
    + CNN을 통해 추출한 벡터를 가지고 각각의 클래스별로 SVM classifier를 학습시킨다.
    + SVM을 사용하는 이유는, 단지 CNN classifier를 썼을 때보다 mAP 성능이 4% 더 낮아졌기 때문이다.

- Non-Maximum Suppression
    + OpenCV에서 배웠던 non-maximum suppression으로, 여러 개의 bounding box 중에서 가장 스코어가 높은 박스만 남기고 나머지는 제거한다.
    + 기준을 IoU(Intersection over Union) 개념을 적용한다. 두 박스가 일치할 수록 1에 가까운 값이 나온다.

    <center><img src="/reference_image/MH.Ji/R-CNN models/4.PNG" width="70%"></center><br>

- Bounding Box Regression
    + 지금까지 물체가 있을만한 위치를 추론하고, 해당 물체의 종류를 판별할 수 있는 classification 모델을 학습시켰다. 하지만 selective search를 통해서 찾은 박스가 과연 정확한거라고 보장할 수 있을까?
    + selective search가 찾은 bounding box의 위치를 교정하기 위해 bounding box regression을 도입한다.
    + selective search를 통해 추출된 bounding box에는 다음과 같은 정보들이 포함된다.

    <center><img src="/reference_image/MH.Ji/R-CNN models/5.PNG" width="30%"></center><br>
    <center><img src="/reference_image/MH.Ji/R-CNN models/6.PNG" width="30%"></center><br>

    + 하나의 박스에는 x, y좌표와 width, height의 값이 담기고, Ground Truth(정답)은 G로 표현한다.
    + 우리는 최대한 정답에 가깝게 만들어줘야 하기 떄문에 다음과 같은 수식을 얻을 수 있다.

    <center><img src="/reference_image/MH.Ji/R-CNN models/7.PNG" width="30%"></center><br>

    + G에 가깝게 하도록 d의 값을 찾는 것으로 다음과 같은 함수로 가중치를 부여한다.

    <center><img src="/reference_image/MH.Ji/R-CNN models/8.PNG" width="30%"></center><br>

- R-CNN의 특징
    + multi staged training을 해야 한다(fine tune, svm classifier, bounding box regression).
    + 2,000개의 영역을 모두 CNN으로 학습하기 때문에 속도가 느리다.

<br><br>

## 2. Fast R-CNN
- 구조
    + 이전의 R-CNN이 여러 번 CNN inference하는 방식으로 인해 속도가 저하되었다면, Fast R-CNN은 CNN 특징 추출부터 classification, bounding box regression까지 모두 하나의 모델에 학습시킨다.

    <center><img src="/reference_image/MH.Ji/R-CNN models/9.PNG" width="70%"></center><br>

- 동작 방식
    + 전체 이미지를 미리 학습된 CNN을 통과시켜 feature map을 추출한다.
    + selective search를 통해서 찾은 각각의 RoI에 대해 RoI Pooling을 진행한다. 그 결과 고정된 크기의 feature vector를 얻는다.
    + feature vector는 fully connected layer들을 통과하고, 2개의 브랜치로 나뉜다.
        - 첫 번째는 softmax를 통과해서 해당 RoI가 어떤 물체인지 classification한다(SVM 사용 x).
        - 두 번째는 bounding box regression을 통해 bounding box의 위치를 업데이트한다.

- RoI Pooling
    + 입력된 이미지가 CNN을 통과하여 feature map을 추출한다. 추출된 feature map을 미리 정해놓은 H x W 크기에 맞게끔 grid를 설정한다. 그리고 각각의 칸 별로 가장 큰 값을 추출하는 max pooling을 실시하여 항상 H x W 크기의 feature map이 나오도록 한다.
    + 마지막으로 H x W 크기의 feature map을 1차원으로 늘려서 feature vector를 추출한다.

    <center><img src="/reference_image/MH.Ji/R-CNN models/10.PNG" width="70%"></center><br>

    + input image와 feature map의 크기가 다른 경우, 그 비율을 구해서 RoI를 조절한 다음 RoI Pooling을 진행한다.

- Multi Task Loss
    + 최종적으로 RoI Pooling을 이용하여 feature vector를 구했다. 이제 이 벡터로 classification과 bounding box regression을 적용하여 각각의 loss를 구하고, 이를 back propagation하여 전체 모델을 학습시킨다.
    + 이때, classification loss와 bounding box regression을 적절하게 엮어주는 것을 multi task loss라고 한다.

    <center><img src="/reference_image/MH.Ji/R-CNN models/11.PNG" width="70%"></center><br>

    + p는 softmax를 통해서 얻은 N + 1(N개의 object 배경, 아무것도 아닌 배경)개의 확률값이다.
    + u는 해당 RoI의 ground truth의 라벨값이다.

    + bounding box regression을 적용하면 위의 식에서 각각 x, y, w, h값을 조정하는 tk를 반환한다. 즉, 이 RoI를 통해 어떤 클래스는 어디 위치로 조절해야하는지에 대한 값이다. loss function에서는 이 값들 가운데 ground truth 라벨에 해당하는 값만 가져오며, 이는 tu에 해당한다.
    + v는 ground truth bounding box 조절값에 해당한다.

    <center><img src="/reference_image/MH.Ji/R-CNN models/12.PNG" width="30%"></center><br>

    + 다시 전체 로스에서 앞부분 p와 u를 가지고 classification loss를 구한다. 여기서는 log loss를 사용한다.

    <center><img src="/reference_image/MH.Ji/R-CNN models/13.PNG" width="30%"></center><br>

    + 전체 로스의 뒷부분은 bounding box regression을 통해 얻은 loss로 다음 수식과 같다.

    <center><img src="/reference_image/MH.Ji/R-CNN models/14.PNG" width="30%"></center><br>

    + 위의 수식에서 입력으로는 정답 라벨에 해당하는 BBR 예측값과 ground truth 조절값을 받는다. 그리고 x, y, w, h 각각에 대해서 예측값과 라벨값의 차이를 계산한 다음, smoothL1이라는 함수를 통과시킨 합을 계산한다.

    <center><img src="/reference_image/MH.Ji/R-CNN models/15.PNG" width="50%"></center><br>

    + 예측값과 라벨값의 차이가 1보다 작으면 0.5(x^2)로 L2 distance를 계산해준다. 반면에 1보다 크다면 L1 distance를 계산해준다. 이는 object detection task에 맞춰 loss function을 커스텀하는 것으로 볼 수 있다.

- Backpropagation through RoI Pooling Layer
    + loss function까지 구했고, 네트워크를 학습시키는 일이 남았다. 이전의 SPPNet에서는 feature map을 추출하는 CNN 부분은 놔두고, SPP 이후의 FC들만 fine tune하였다. 하지만 Fast R-CNN에서는 이렇게하면 이미지로부터 feature를 뽑는 가장 중요한 역할을 하는 CNN이 학습될 수 없기 때문에 성능 향상에 제약이 걸린다고 주장한다.

    + 과연 네트워크를 어디까지 학습시키는 것이 성능에 가장 좋은지를 검증해보자.

    <center><img src="/reference_image/MH.Ji/R-CNN models/16.PNG" width="30%"></center><br>

    + xi는 CNN을 통해 추출된 feature map에서 하나의 feature를 의미한다. 전체 loss에 대해서 이 feature 값의 편미분을 구하면 그 값이 곧 xi에 대한 loss값이 되며, back propagation을 수행할 수 있다.

    + feature map에서 RoI를 찾고, RoI Pooling을 적용하기 위해서 H x W 크기의 grid로 나눈다. 이 grid들은 sub-window라고 부르며, 위 수식에서 j란 몇 번째 sub-window인지를 나타내는 인덱스이다. yrj란 RoI Pooling을 통과하여 최종적으로 얻어진 output 값이다.

    <center><img src="/reference_image/MH.Ji/R-CNN models/17.PNG" width="70%"></center><br>

    + xi가 최종 predicion 값에 영향을 주려면, xi가 속하는 모든 RoI의 sub-window에서 해당 xi가 최대가 되면 된다. i*(r, j)란 RoI와 sub window index j가 주어졌을 때 최대 feature 값의 index를 의미하며, 이는 곧 RoI Pooling을 통과하는 인덱스값을 말한다. 이 RoI Pooling을 통과한 이후 값에 대한 loss는 이미 전체 loss에 대한 yrj의 편미분값으로, 이미 계산되어 있다. 따라서 이를 중첩시키면 xi에 대한 loss를 구할 수 있다.

    + 논문의 저자들은 실험을 통해 실제로 CNN까지 fine tuning하는 것이 성능 향상에 도움이 되었다는 결과를 보여준다.

    <center><img src="/reference_image/MH.Ji/R-CNN models/18.PNG" width="70%"></center><br>

<br><br>

## 3. Faster R-CNN
- 구조
    + Faster R-CNN의 핵심 아이디어는 Region Proposal Network(RPN)이다. 기존 Fast R-CNN의 구조를 그대로 계승하면서, selective search를 제거하고 RPN을 통해서 RoI를 계산한다. 이를 통해서 GPU를 통한 RoI 계산이 가능해졌다.

    + RPN은 selective search가 2,000개의 RoI를 계산하는 데 반해 800개 정도의 RoI를 계산하면서 더 높은 정확도를 보인다.

    <center><img src="/reference_image/MH.Ji/R-CNN models/19.PNG" width="70%"></center><br>

- Region Proposal Network

    <center><img src="/reference_image/MH.Ji/R-CNN models/20.PNG" width="70%"></center><br>

    1) CNN을 통해 뽑아낸 feature map을 입력으로 받고, 크기를 H x W x C로 잡는다.  

    2) feature map 3x3 convolution을 256 또는 512 채널만큼 수행한다. 위 그림의 Intermediate layer에 해당한다. 이때, padding을 1로 설정해주어, H x W가 보존되게 한다. Intermediate layer 수행 결과, H x W x 256 or H x W x 512의 크기의 두 번째 feature map을 얻는다.  
    
    3) 두 번째 feature map을 입력받아 classification과 bounding box regression 예측값을 계산해준다. 이때 주의해야할 점은 fully connected layer가 아니라, 1x1 convolution을 이용하여 계산하는 fully connected network의 특징을 가진다. 이는 입력 이미지의 크기에 상관없이 동작할 수 있도록 함이다.  

    4) 먼저 classification을 수행하기 위해서 1x1 convolution을 (2x9) 채널 수만큼 수행해주며, 그 결과로 H x W x 18 크기의 feature map을 얻는다. H x W 상의 하나의 인덱스는 feature map 상의 좌표를 의미하고, 그 아래 18개의 채널은 각각 해당 좌표를 anchor로 잡아 k개의 anchor 박스들이 object인지 아닌지에 대한 예측값을 담고 있다. 즉, 한번의 1x1 convolution으로 H x W 개의 anchor 좌표들에 대한 예측을 모두 수행할 수 있다. 이제 이 값들을 적절히 reshape해준 다음 softmax를 적용하여 해당 anchor가 오브젝트일 확률값을 얻는다.

    5) 두 번째로 bounding box regression 예측값을 얻기 위한 1x1 convolution을 (4x9) 채널 수만큼 수행해준다.

    6) 이제 RoI를 구하기 위해, classification을 통해서 얻은 물체일 확률값들을 정렬한 다음, 높은 순으로 K개의 anchor만 추출한다. 그 다음 K개의 anchor들에 각각 bounding box regression을 적용해준다. 그 다음 Non-Maximum-Suppression을 적용하여 RoI를 구해준다.

- Loss Function
    + loss function은 classification과 bounding box regression을 통해 얻은 loss를 엮은 형태를 취한다.

    <center><img src="/reference_image/MH.Ji/R-CNN models/21_1.PNG" width="70%"></center><br>

    + i는 하나의 anchor를 의미하며, pi는 classification을 통해서 얻은 해당 anchor가 물체일 확률을 의미한다. ti는 bounding box regression을 통해서 얻은 박스 조정 값 벡터를 의미한다.

    + pi*와 ti*는 ground truth 라벨에 해당한다.

    + classification은 log loss를 통해서 계산한다. regression loss는 Fast R-CNN에서 나온 smoothL1 함수를 사용한다.

- Training
    + 전체 모델을 학습시키기란 어려운 작업이기 때문에, 4단계에 걸쳐서 모델을 번갈아 학습시키는 Alternating Training 기법을 취한다.

    1) ImageNet pretrained 모델을 불러온 다음, RPN을 학습시킨다.  
    
    2) 1)에서 학습시킨 RPN에서 기본 CNN을 제외한 Region Proposal 레이어만 가져온다. 이를 활용하여 Fast R-CNN을 학습시킨다. 이때, 처음 feature map을 추출하는 CNN까지 fine tune 시킨다.  

    3) 앞서 학습시킨 Fast R-CNN과 RPN을 불러와서 다른 가중치들은 고정하고 RPN에 해당하는 layer들만 fine tune시킨다. 여기서부터 RPN과 Fast R-CNN이 convolution weight을 공유하게 된다.  

    4) 공유하는 CNN과 RPN은 고정시킨채, Fast R-CNN에 해당하는 layer만 fine tune 시킨다.  

<br><br>

## 4. Mask R-CNN
- 개요
    + 지금까지의 R-CNN 계열의 모델을이 object detection을 위한 모델이라면, Mask R-CNN은 Faster R-CNN을 확장하여 instance segmentation을 위한 모델이다.

    + Mask R-CNN = Faster R-CNN with FCN on RoIs

    + Faster R-CNN과 다른 점은 Faster R-CNN에 존재하는 bounding box를 위한 브랜치에 병렬로 object mask predict 브랜치를 추가하였다. 또한, RoI Pooling 대신 RoI Align을 사용하였다.

    + Instance Segmentation을 하기 위해서는 object detection과 semantic segmentation을 동시에 해야 한다. 이를 위해 Mask R-CNN은 기존의 Faster R-CNN을 object detection 역할을 하도록 하고, 각각의 RoI에 mask segmentation을 해주는 작은 fully convolutional network를 추가해주었다.

    + 기존의 Faster R-CNN은 object detection을 위한 모델이었기에, RoI Pooling 과정에서 정확한 위치 정보를 담는 것은 중요하지 않았다. RoI Pooling에서 RoI가 소수점 좌표를 갖고 있을 경우에는 각 좌표를 반올림한 다음에 Pooling을 해준다. 이렇게 되면 input image의 원본 위치 정보가 왜곡되기 때문에 classification task에서는 문제가 발생하지 않지만, segmentation task에서는 문제가 발생한다.

    <center><img src="/reference_image/MH.Ji/R-CNN models/21.PNG" width="70%"></center><br>

    + 위와 같은 문제를 해결하기 위해 RoI Pooling 대신에 RoI Align을 사용한다.

- 구조

    <center><img src="/reference_image/MH.Ji/R-CNN models/25.PNG" width="70%"></center><br>

- RoI Align

    <center><img src="/reference_image/MH.Ji/R-CNN models/22.PNG" width="30%"></center><br>

    + 파란색 점선은 feature map을 나타내고, 실선은 RoI를 나타낸다. RoI에서 얻어내고자 하는 정보는 박스안의 동그라미 점(샘플링 포인트)이다. 하지만 RoI가 정확히 칸에 맞춰져 있지 않기 때문에, 이에 대한 정확한 계산을 위해 bilinear interpolation을 적용하여 각 샘플링 포인트의 값을 계산한다.

    <center><img src="/reference_image/MH.Ji/R-CNN models/26.PNG" width="70%"></center><br>

    + 게산된 값들을 Max Pooling해준다.

    <center><img src="/reference_image/MH.Ji/R-CNN models/23.PNG" width="70%"></center><br>

    <center><img src="/reference_image/MH.Ji/R-CNN models/24.PNG" width="70%"></center><br>

- R-CNN vs Fast/Faster R-CNN vs Mask R-CNN

    <center><img src="/reference_image/MH.Ji/R-CNN models/27.PNG" width="70%"></center><br>

<br><br>

## 참고자료
- [갈아먹는 Object Detection](https://yeomko.tistory.com/)
- [Mask R-CNN 정리](https://mylifemystudy.tistory.com/82)