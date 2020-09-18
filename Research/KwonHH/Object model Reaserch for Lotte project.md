# 롯데 정보통신 무인편의점 구축을 위한 모델선정
## 상품 데이터셋에 적합한 모델들 검토
- 20.09.16 모델 발표(RODEO) 이후 피드백 : RODEO는 Incremetal(증분) learning에 대한 논문<br>이는 기존에 학습된 모델에 새로운 학습 데이터가 발생할 경우, 모델을 처음부터 다시 학습하는 것이 아니라 새로운 데이터에 대해서만 학습시킨 뒤 업데이트<br>하지만 Incremental Learning의 경우 성능이 보장되지 않고, 어려운 분야이므로, 현재는 연구분야에 그치고 있음.<br>따라서 새로운 Object Detection 모델을 선정 및 실습하고, 성능을 확인하는 것을 추천<br><br>Mobilenet V2 Classification Augmentation에 대해서는 깔끔하게 진행되었음. <br>추가적으로 ColorJitter에 Contrast를 조정해보는 것도 좋은 시도가 될 것 같음.<br><br>
1. Rethinking ImageNet Pre-training
    1. 아이디어
        - 최근 large scale data로부터 학습을 시킨 이후  더 작은 data 를 학습시키기 위해 fine-tuned model을 사용하는 방식이 널리 사용되고 있음
        - 그 증거로 pre-training 방식은 classification 과 object detection 분야에서 많은 모델들이 sota 순위권에 등재
        - 그러나 그런 데이터셋과는 맞지 않는, 즉 작은 사이즈의 데이터를 사용하는 task가 있을 수 있음
        - 따라서 논문에서는 COCO 데이터셋에 대해서 pretrain 없이 훈련을 시키고, 그에 대한 competitive object detection 과 instance segmentation accuracy 를 발표함
        - 더 놀라운 것은 fine tuning pretrained model에 최적화 된 hyper parameters를 사용했음에도 이런 결과를 얻을 수 있었다는 것이다
        - 논문에서는 COCO를 사용해서 random 초기화 한 것은 ImageNet의 Pre-train 과 AP를 포함한 다양한 기준에서 40~50% 이상 동등한 수준을 보임<br>더욱이 이는 COCO를 10% 더 적게 train 했음에도 이와 같은 적합성을 찾았다는 것이다
        - 또한 ResNet 101 보다 최대 4배 큰 모델을 overfitting 없이 만들 수 있다<br> https://arxiv.org/pdf/1811.08883v1.pdf <br><br>
1. SpineNet : Learning Scale-Permuted Backbone for Recognition and Localization
    1. 아이디어
        - input image는 conv layers를 거치며 해상도가 낮아짐
        - 이것은 classification에는 필요한 방식이지만, recognition 과 localization 에는 불필요함
        - encoder-decoder 구조는 이것을 해결하기 위해서 적용되었으며, backbone에 decoder network를 적용하여 classification 작업을 수행
        - 본 논문에서는 multi-scale feature에서는 이러한 encoder-decoder 구조가 backbone에서의 scale 감소로 인해 적합하지 않음을 밝힘
        - 따라서 SpineNet에서는 NAS에 의해서 object detection 에서 학습된 scale-permuted 와 cross-scale(서로 다른 scale 간의 상호연산) 구조를 사용
        - 비슷한 building block을 사용하여 SpineNet은 ResNet-FPN 보다 최대 3% 더 높은 AP를 얻음<br> !(SpinNet AP)[] <br> !(SpineNet backbone models and param, resol)<br>
1. NAS-FPN : Learning Scalable Feature Pyramid Architecture fo Object Detection
    - SpinNet 논문을 읽어본 결과, SpineNet 보다 성능이 낮은 것 같음
1. SaccadeNet: A Fast and Accurate Object Detector
    1. 아이디어
        - 사람의 눈은 물체를 인식할 때 모든 시각적인 정보로부터 물체를 인식하는 것이 아닌, 물체의 중요한 정보로부터 물체를 인식하고 이해함
        - 이것은 더 빠르고 정확하게 정보를 확인할 수 있게 해줌
        - 여기서, speed 와 accuracy 사이에 trade-off 를 위해서 물체의 center point 의 최상단에 4개의 bounding box 꼭지점을 정보의 key point 로 설정하였음
        - 이 꼭지점들을 연속적으로 포함시키고, 집적하여 정확한 추론이 가능하도록 하였다
        - 즉, 중심점 검출 모듈 + 꼭지점 검출 모듈 + Attentin Transitive 모듈 (모듈의 중심점과 꼭지점을 토대로 bounding box 추론) => 집적 모듈 => 정확한 bounding 박스 추론 <br> !(SACCAD method)[] <br><br>
    1. 방식
        - backbone model
            - up-sampling : DLA-34
            - down-sampling : CenterNet
        - Head module : SACCAD Net<br><br> !(saccad train)[] <br>
        - 384 x 384 size 입력
        - batch size 32
        - learning rate 1.25 * 10^(-4)
        - 70 epochs
    1. Efficiency Study
    
1. SOLOv2: Dynamic, Faster and Stronger
1. CBNet: A Novel Composite Backbone Network Architecture for Object Detection
1. Grid R-CNN Plus: Faster and Better
1. GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond