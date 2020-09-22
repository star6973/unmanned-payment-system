## Faster R-CNN
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

## 참고자료
> [갈아먹는 Object Detection](https://yeomko.tistory.com/)  

> [Mask R-CNN 정리](https://mylifemystudy.tistory.com/82)  

> [Image segmentation with Mask R-CNN](https://medium.com/@jonathan_hui/image-segmentation-with-mask-r-cnn-ebe6d793272)