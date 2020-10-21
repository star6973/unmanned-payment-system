# EfficientDet

- 객체탐지는 요즘 최첨단 기술로 넘어갈수록 가격이 비싸지고 있어 실제 리소스의 제약을 감안할때 효율적인 모델이 중요해졌습니다

- 그래서 EfficientDet은 리소스의 제약(모바일네트워크의 작은크기의 리소스 + 데이터센터의 큰 리소스)에 효율적인 모델입니다

### challenge 1. Efficient multi-scale feature fusion
- input 해상도가 다르면 그 각각 인풋의 feature가 합쳐지긴 힘들다. output에 기여하는 정도가 달라서 feature마다 weight를 부여하는  weighted bi-directional FPN(BiFPN) 방법을 제안 


### challenge 2.Model scaling 

- EfficientNet에서 제안한 Compound Scaling 기법은 모델의 크기와 연산량를 결정하는 요소들(input resolution, depth, width)을 동시에 고려하여 증가시키는 방법을 의미하며, 이 방법을 통해 높은 성능을 달성할 수 있었습니다. 이러한 아이디어를 Object Detection에도 적용을 할 수 있으며, backbone, feature network, box/class prediction network 등 모든 곳에 적용을 하였습니다.

>> backbone = efficientNet

>> Feature network =  BiFPN

>> box/class prediction network = 
 
