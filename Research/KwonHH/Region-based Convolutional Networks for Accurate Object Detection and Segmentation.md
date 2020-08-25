# Region-based Convolutional Networks for Accurate Object Detection and Segmentation
## 논문 리뷰
- Abstract
    - SOTA의 Object detection net은 object의 위치를 증명하기 위해서 region proposal 알고리즘에 의존한다
    - SPPNet 과 Fast RCNN 처럼 진보된 방식은 bottleneck 등을 통해 running time을 감소시켰다
    - 본 논문에서는 전체 이미지에 대해서 특징을 convolution하는 detection network (Region Proposal Network)를 사용, 따라서 cost-free의 proposal region 가 거의 가능해졌다