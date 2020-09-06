# What is object detection?

![r-cnn](https://hoya012.github.io/assets/img/object_detection_first/fig1_cv_task.PNG)

#### Object Detection = Multi-labeled Classification + Bounding Box Regression

- 저희가 일반적으로 Object Detection 이라 부르는 문제는 한 이미지에 여러 class의 객체가 동시에 존재할 수 있는 상황을 가정합니다. 즉, multi-labeled classification (한 이미지에 여러 class 존재)과 bounding box regression (box의 좌표 값을 예측) 두 문제가 합쳐져 있다고 생각하시면 됩니다. 

- 하나의 이미지에 여러 객체가 존재하여도 검출이 가능하여야 합니다.

## deep learning에서도 사용이 되는 object detection 대표적인 2가지 방법

#### Sliding Window

- Sliding Window 기법은 딥러닝 이전에 가장 자주 사용되던 방법으로, 다양한 scale의 window를 이미지의 왼쪽 위부터 오른쪽 아래까지 sliding하며 score를 계산하는 방법을 의미합니다.

![slide](https://hoya012.github.io/assets/img/object_detection_first/fig6_sliding_window.PNG)


#### Selective Search (SS)

![ss](https://hoya012.github.io/assets/img/object_detection_first/fig7_selective_search.PNG)

- 영상의 계층적 구조를 활용하여 영역을 탐색하고 그룹화하는 과정을 반복하며 객체의 위치를 proposal 해주는 기능을 수행합니다. 마찬가지로 관련 자료들을 참고하셔서 잘 알아 두시면 좋을 것 같습니다.


#### R-CNN

- input image가 있으면 Selective Search를 통해 Region Proposal을 수행합니다. 그 뒤, Proposal된 영역들을 CNN의 고정 사이즈를 가지는 입력으로 변환시키고, 각 영역마다 Classification을 수행하는 방식으로 되어있습니다. 예를 들어 1000개의 Region이 Proposal된 경우에는 CNN에서 1000번의 Classification이 수행되는 것입니다. 또한 Proposal된 Region에서 실제 object의 위치와 가까워지도록 보정해주는 Regression도 수행이 됩니다.


#### backbone

- 입력 이미지를 특징맵으로 변형시켜주는 부분이 백본이다 .

#### Neck 

- backbone 과 head를 연결짓는 부분으로, 날것의 특징맵에 정제, 재구성을 한다.
