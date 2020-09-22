## Mask R-CNN
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

## 참고자료
> [갈아먹는 Object Detection](https://yeomko.tistory.com/)  

> [Mask R-CNN 정리](https://mylifemystudy.tistory.com/82)  

> [Image segmentation with Mask R-CNN](https://medium.com/@jonathan_hui/image-segmentation-with-mask-r-cnn-ebe6d793272)