average precision (AP) @[ Iou=0.5: 0.95 ] | area=   all | maxDets=100 ] 
average precision (AP) @[ Iou=0.5: 0.95 ] | area=   all | maxDets=100 ] 
average precision (AP) @[ Iou=0.5: 0.95 ] | area=   all | maxDets=100 ] 
average precision (AP) @[ Iou=0.5: 0.95 ] | area=   all | maxDets=100 ] 
average precision (AP) @[ Iou=0.5: 0.95 ] | area=   all | maxDets=100 ] 
average precision (AP) @[ Iou=0.5: 0.95 ] | area=   all | maxDets=100 ] 
average  recall   (AP) @[ Iou=0.5: 0.95 ] | area=   all | maxDets=100 ] 
average  recall   (AP) @[ Iou=0.5: 0.95 ] | area=   all | maxDets=100 ] 
average  recall   (AP) @[ Iou=0.5: 0.95 ] | area=   all | maxDets=100 ] 
average  recall   (AP) @[ Iou=0.5: 0.95 ] | area=   all | maxDets=100 ] 
average  recall   (AP) @[ Iou=0.5: 0.95 ] | area=   all | maxDets=100 ] 
average  recall   (AP) @[ Iou=0.5: 0.95 ] | area=   all | maxDets=100 ] 

![img1](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcsmX3b%2FbtquhjiOyKI%2FoKNhWcBu2S7GY9JdLnyq4k%2Fimg.png)

### average precision 
1. precision (정밀도)
![img](https://hoya012.github.io/assets/img/object_detection_fourth/fig3.PNG)
- 모든 검출 결과 중 옳게 검출한 비율 

- TP는 옳은 검출(true positive), FP는 잘못된 검출(false positive)

- precision은 알고리즘이 검출해낸 것들 중에서 재대로 검출해낸 비율을 의미한다. 

- 즉 Precision을 높이기 위해선 모델이 예측 Box를 신중하게 쳐서 FP를 줄여야 합니다.

### average recall 
1. Recalll(재현율)
![img2](https://hoya012.github.io/assets/img/object_detection_fourth/fig4.PNG)
- 마땅히 검출해내야 하는 물체들중 제대로 검출된 것의 비율을 의미한다. 

- FN은 false negative의 약자로 '검출되었어야 하는 물체인데 검출되지 않은 것'을 의미한다. 

* Precision과 Recall은 항상 0과 1사이의 값으로 나오게 되는데, Precision이 높으면 Recall은 낮은 경향이 있고, Precision이 낮으면 Recall이 높은 경향이 있다는 것이다. 따라서 어느 한 값 만으로 알고리즘의 성능을 평가하는 것은 거의 불가능하고, 두 값을 종합해서 알고리즘의 성능을 평가해야 한다. 그래서 필요한 것이 precision-recall 곡선 및 AP이다.

- TP(옳은 검출)와 FP(틀린 검출)를 결정해주는 기준은 무엇일까? 그 기준은 바로 intersection over union(IoU)이다.

- Recall을 높이기 위해선 모델 입장에서는 되도록 Box를 많이 쳐서 정답을 맞혀서 FN을 줄여야 합니

>> 그래서 Recall 과 Precision은 반비례 관계를 갖게되며 두 값이 모두 높은 모델이 좋은 모델이라 할 수 있습니다
>
### Intersection over union(IoU)
- Object Detection에서 Bounding Box를 얼마나 잘 예측하였는지는 IoU라는 지표를 통해 측정하게 됩니다. IoU(Intersection Over Union)는 Object Detection, Segmentation 등에서 자주 사용되며, 영어 뜻 자체로 이해를 하면 “교집합/합집합” 이라는 뜻을 가지고 있습니다. 실제로 계산도 그러한 방식으로 이루어집니다. Object Detection의 경우 모델이 예측한 결과와 GT, 두 Box 간의 교집합과 합집합을 통해 IoU를 측정합니다.
![img4](https://hoya012.github.io/assets/img/object_detection_fourth/fig1.PNG)

- 처음 보신 분들은 다소 생소할 수 있습니다. IoU는 교집합이 없는 경우에는 0의 값을 가지며, GT와 모델이 예측한 결과가 100% 일치하는 경우에는 1의 값을 가집니다. 일반적으로 IoU가 0.5(threshold)를 넘으면 정답이라고 생각하며, 대표적인 데이터셋들마다 약간씩 다른 threshold를 사용하기도 합니다.

>> PASCAL VOC : 0.5

>>Image Net : min(0.5, wh/(w+10)(h+10))

>>MSCOCO : 0.5, 0.55, 0.6, 0.95


## Precision - Recall 곡선

- PR 곡선은 confidence 레벨에 대한 threshold 값의 변화에 의한 물체 검출기의 성능을 평가하는 방법이다.

- confidence 레벨은 검출한 것에 대해 알고리즘이 얼마나 확신이 있는지를 알려주는 값이다. ex) confidence 레벨 = 0.999이면 굉장히 큰 확신을 가지고 검출한것이다. 

-그래서 confidence레벨에 대해 threshold값을 부여해서 특정값 이상이 되어야 검출된 것을 인정한다. ex) threshold값이 0.4라면 confidence레벨로 0.1또는 0.를 갖고있는 검출은 무시하는 것이다. 

![img6](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc9LaL5%2FbtquhiqWxyj%2FIKuI2gA32HjsF09ythh8UK%2Fimg.png)

>> confidence 레벨에 대한 threshold 값을 아주 엄격하게 적용해서 95%로 했다면, 하나만 검출한 것으로 판단할 것이고, 이때 Precision = 1/1 = 1, Recall = 1/15 = 0.067이 된다. threshold 값을 91%로 했다면, 두 개가 검출된 것으로 판단할 것이고, 이때 Precision = 2/2 = 1, Recall = 2/15 = 0.13이 된다. threshold 값을 검출들의 confidence 레벨에 맞춰 낮춰가면 다음과 같이 precision과 recall이 계산될 것이다. 

![image6](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FccSEQz%2Fbtqufc6vw7L%2FPm8ieSMgK82ICz3cLwLqKK%2Fimg.png)

이 Precision값들과 Recall 값들을 아래와 가이 그래프로 나타내면 그것이 바로 PR곡선이다. 
![PR1](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FqsCQt%2FbtqufbT4BWO%2FRgG7ha2HvpWv52sp4k4sEk%2Fimg.bmp)


![PR2](https://hoya012.github.io/assets/img/object_detection_fourth/fig5.PNG)
- Average Precision의 계산은 Recall을 0부터 0.1 단위로 증가시켜서 1까지(0, 0.1, 0.2, …, 1) 증가시킬 때 필연적으로 Precision이 감소하는데, 각 단위마다 Precision 값을 계산하여 평균을 내어 계산을 합니다. 즉 11가지의 Recall 값에 따른 Precision 값들의 평균이 AP를 의미하며, 하나의 Class마다 하나의 AP 값을 계산할 수 있습니다.

- 이렇게 전체 Class에 대해 AP를 계산하여 평균을 낸 값이 바로 저희가 논문에서 자주 보는 mean Average Precision, 이하 mAP 입니다.



#### Area
- small =  area < 32^2

- medium = 32^2 < area < 96^2
 
- large = area < 96^2


#### maxDets
- [1 10 100] M=3 thresholds on max detections per image