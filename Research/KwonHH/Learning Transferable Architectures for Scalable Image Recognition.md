# Learning Transferable Architectures for Scalable Image Recognition
## 논문 리뷰
Abstract<br>
Our model is 1.2% better in top-1 accuracy than the best human-invented architectures while having 9 billion fewer FLOPS – a reduction of 28% in computational demand from the previous state-of-the-art model.
When evaluated at different levels of computational cost, accuracies of NASNets exceed those of the state-of-the-art human-designed models.
For instance, a small version of NASNet also achieves 74% top-1 accuracy, which is 3.1% better than equivalently-sized, state-of-the-art models for mobile platforms.
- Neural network를 설계할 때 기존의 Neural Architecture Search 방식을 사용하면, ImageNet과 같은 큰 size의 DataSet에서는 architecture를 찾는데 긴 시간이 소요
- 논문의 아이디어 : NASNet architecture
    - CIFAR-10 에서 좋은 성능을 보이는 Cells(layers)를 찾음
    - Cell을 여러층으로 쌓아서 ImageNet DataSet에 적용
    - NASNet architecture를 적용하여 설계한 model을 이용해 SOTA에 등재된 사람이 설계한 model보다 더 좋은 성능의 model을 설계하는데 성공<br>
    <br>
    Our model is 1.2% better in top-1 accuracy than the best human-invented architectures while having 9 billion fewer FLOPS – a reduction of 28% in computational demand from the previous state-of-the-art model.
    When evaluated at different levels of computational cost, accuracies of NASNets exceed those of the state-of-the-art human-designed models.
    For instance, a small version of NASNet also achieves 74% top-1 accuracy, which is 3.1% better than equivalently-sized, state-of-the-art models for mobile platforms.
<br>
- 또 다른 장점으로는 다른 문제로 transferrable
   - NASNet를 적용했을 때 COCO DataSet에 대해서 Object Detection 분야에서 SOTA에 등재된 Faster-RCNN framework를 4% 향상시킴<br><br>
1. Method
    1. NAS ; Neural Architecture Search
    ![Overview of NAS](file:///C:/Users/zkzh6/OneDrive/Pictures/Learmomg%20Transferrable%20Arcitectures%20for%20Scalable%20Image%20Recognition/01.JPG)<br>
    (Barret Zoph et al. Transferable Architectures for Scalable Image Recognition, 2018)<br>
        - The controller(RNN)
            - Child Network에 대해 validation 과정을 통해서 특정 정확도 R이 나오도록 학습시킴
            - 이때의 R에 의해서 child architecture에 대한 확률p의 gradient의 크기 조정되며 Controller가 update
            - Controller는 더 좋은 architecture를 생성함
    - 전체적인 architecture는 수동으로 미리 결정되어 있음
    - 
            
   