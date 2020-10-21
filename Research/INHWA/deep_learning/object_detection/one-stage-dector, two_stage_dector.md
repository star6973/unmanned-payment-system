## Object Detection

- Input image에 존재하는 여러 object들을 Bounding box를 통해 Localization을 하고, Classification 하는 것을 나타낸다. 

>> 이러한 Deep Learing Object Detection은 크게 one-stage Detector와 two-stage Detector로 나눌 수 있다. 

>>one-stage Detector의 대표적인 모델은 YOLO 계열(v1,v2,v3), SSD 계열(SSD, DSSD, DSOD, RetinaNet, RefineDet ..)이 있으며, 

>>two-stage Detector의 대표적인 모델은 R-CNN 계열(R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN ...)이 있다. 


## one-stage Detector vs two-stage Detector

![image1](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F6590G%2FbtqCWbQVnkx%2FgrxthKJ38iTwEIpqdX2TWk%2Fimg.png)

- Object Detection은 물체의 위치를 찾는 Localization과 물체를 구별하는 Classification이 필요하다.

- One-stage Detector는 이 두 가지(Classification, Localization)를 동시에 수행하여 결과를 얻는 방법이고,

- Two-stage Detector는 이 두 가지(Classification, Localization)를 순차적으로 수행하여 결과를 얻는 방법이다.

 

>> 결과적으로 One-stage Detector는 일반적으로 빠르지만 정확도가 낮고, Two-stage Detector는 일반적으로 느리지만 정확도가 높다.

 
 ### One-Stage Detector 
 
 - Two-stage detector와 반대로 regional proposal과 classification이 동시에 이루어진다. 즉, classification과 localization 문제를 동시에 해결하는 방법이다.
 
 
 