# Faster-R-CNN

## introduction 


## Region Proposal Networks


#### anchor Box
- image/feature pyramids는 이미지의 크기를 조정하며 feature을 뽑아낸다

- multuple-scaled slifing window처럼 filter크기를 변경할 필요가없다.

>> Fast-R-CNN은 이미지 크기 바꿀필요없고 필터 사이즈를 바꿀 필요도 없습니다.

![image1](https://curt-park.github.io/images/faster_rcnn/Figure1.png)
![image2](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRbH71hNFmwA0dEjQGKdxNsM4X1yg4XZwb1pQ&usqp=CAU)
![image3](https://cphinf.pstatic.net/mooc/20181025_153/1540427223883R1CjD_PNG/anchorbox_y.PNG)



#### Labeling to each anchor (Object? or Background?)

>>> 특정 anchor에 posive label이 할당되는데에는 다음과 같은기준이 있다.

- 가장높은 intersection-over-Union(IoU)을 가지고 있는 anchor.

- IoU > 0.7을 만족하는 anchor = positive anchor
- IoU < 0.3을 만족하는 anchor = negative anchor


#### computation Process

- Shared CNN에서 convolutional feature map(14X14X512 for VGG)을 입력받는다. 여기서는 Shared CNN으로 VGG가 사용되었다고 가정한다. (Figure3는 ZF Net의 예시 - 256d)

- Intermediate Layer: 3X3 filter with 1 stride and 1 padding을 512개 적용하여 14X14X512의 아웃풋을 얻는다.
>>>>Output layer

>cls layer[object 분류]: 1X1 filter with 1 stride and 0 padding을 9*2(=18)개 적용하여 14X14X9X2의 이웃풋을 얻는다. 여기서 filter의 개수는, anchor box의 개수(9개) * score의 개수(2개: object? / non-object?)로 결정된다.

>reg layer[BB regression]: 1X1 filter with 1 stride and 0 padding을 9*4(=36)개 적용하여 14X14X9X4의 아웃풋을 얻는다. 여기서 filter의 개수는, anchor box의 개수(9개) * 각 box의 좌표 표시를 위한 데이터의 개수(4개: dx, dy, w, h)로 결정된다. (코드1: 예측된 bounding box에 대한 정보, 코드2: 사전정의된 anchor box의 정보에 예측된 bounding box에 대한 정보를 반영)

![image4](https://curt-park.github.io/images/faster_rcnn/Figure3.png)
![image6](https://user-images.githubusercontent.com/40360823/44243408-e7133080-a209-11e8-8668-a260ea66f761.png)


## Loss Function

- loss function은 아래 그림과 같다.

![loss](https://curt-park.github.io/images/faster_rcnn/LossFunction.png)

- pi = predicted probability of anchor

- pi* = ground-truth label ( 1: anchor is posive, 0: anchor is negative)

- lambda: Balancing parameter. Ncls와 Nreg 차이로 발생하는 불균형을 방지하기 위해 사용된다. cls에 대한 mini-batch의 크기가 256(=Ncls)이고, 이미지 내부에서 사용된 모든 anchor의 location이 약 2,400(=Nreg)라 하면 lamda 값은 10 정도로 설정한다.

- ti: Predicted Bounding box

- ti*: Ground-truth box

## Training RPNs

- end-to-end로 back-propagation 사용.

- Stochastic gradient descent
- 한 이미지당 랜덤하게 256개의 sample anchor들을 사용. 이때, Sample은 positive anchor:negative anchor = 1:1 비율로 섞는다. 혹시 positive anchor의 개수가 128개보다 낮을 경우, 빈 자리는 negative sample로 채운다. 이미지 내에 negative sample이 positive sample보다 훨씬 많으므로 이런 작업이 필요하다.
- 모든 weight는 랜덤하게 초기화. (from a zero-mean Gaussian distribution with standard deviation 0.01)
- ImageNet classification으로 fine-tuning (ZF는 모든 layer들, VGG는 conv3_1포함 그 위의 layer들만. Fast R-CNN 논문 4.5절 참고.)
- Learning Rate: 0.001 (처음 60k의 mini-batches), 0.0001 (다음 20k의 mini-batches)
- Momentum: 0.9
- Weight decay: 0.0005


## Conclusion

- 실험결과에서 보이는 것처럼 약간의 정확도가 향상되었고, 실행시간이 현격히 줄어들었다. 헌데, 논문에서 이를 ‘object detection system to run at near real-time frame rates’ 라고 표현하는 것으로 보아, 아직 실시간 영상처리 등에서 사용하기에는 다소 부족한 부분이 있는 것으로 보인다.
