# Rich feature hierarchies for accurate object detection and semantic segmentation Tech Report
## Tech Report Review
- Abstract
    - object detection 은 지난 몇년간 정체기
    - 주로 low level과 high level의 앙상블을 하는것이 가장 좋은 효과를 보였지만, 다소 복잡한 방식
    - 따라서 본 paper에서는 간단하고 scalable 가능한 detection 알고리즘을 제안하여 이전 VOC2012의 최우수 모델보다 mAP를 30% 이상 끌어올림<br>그 방법으로는
        1. 이미지를 분리하고, localize 하기 위해서 아래에서부터 윗방향으로 region proposal 에 대해서 high capacity CNN 적용
        1. 만약 라벨 된 훈련 데이터가 부족할 경우 중요한 performance boost를 하도록 함
    - Proposal region 에 CNN을 적용했기 때문에 R-CNN이라고 함<br><br>
1. Introduction
    - CNN은 1990년대 굉장히 많이 쓰였지만, SVM의 성장으로 인기가 식었다
    - 2012년  Krizhevsky et al. 논문에서 CNN을 사용해서 정확도를 꽤 높이면서 다시 인기를 끌러올리는 계기가 되었다<br><br>
    - 본 논문은 PASCAL VOC 데이터에서 HOG 기반이 아닌, CNN 을 사용했을 때 극적으로 object detectiospecificn 동작을 끌어올린다는 것을 보여주는 최초의 논문
    - 이것을 수행하기 위해서 다음의 문제에 주목했다
        1. localizing object with a deep network
        1. training a high-capacity model with only a small quantity of annotated detection data
            - 이미지 분류와는 다르게 detection은 localizing을 필요로 함
            - 한 가지 접근법은 regression problem처럼 localization 처럼 틀을 잡는 것인데, Szcegedy et al. 의 논문에서 이러한 전략은 성공하지 못할 것이라고 했다<br> => 대안은 sliding window 사용하는 것
            - CNN은 전형적으로 정해진 category들에 대해서 분류하기 위해서 사용되어졌다
            - 높은 spatial resoultion 을 유지하기 위해서 CNN은 주로 2개의 convolution 과 pooling을 가진다
            - 본 논문에서도 역시 sliding window를 사용하려 했지만<br>5개의 layer와 큰 receptive field(195 X 195 pixels), large stride(32 X 32 pixels) 로 인해서 기술적인 과제가 발생했다
                1. 첫 번째 과제
                    - 대신에 region을 사용한 recognition을 사용해서 localization 문제를 해결<br> => object detection 과 semantic segmentation 에 모두 성공적
                    - test time에서 region proposal에 독립적인 약 2000개의 category를 생성하고, 각 region에서 특정한 category로 SVM을 이용해서 length가 고정된 feature vector를 추출
                        - 여기에 affine image warping 기술 사용 => region 의 모양과는 관계없이 region proposal 로부터 고정된 크기의 CNN 입력을 계산하는 방식<br><br>
                    - 본 논문의 version을 계속 업그레이드 하면서 R-CNN 과 OverFeat detection을 하나하나 비교하였다
                        - OverFeat 에서는 sliding window CNN을 이용해서 detection 했는데, 현재(당시 ILSVRC 2013) 에서는 가장 우수한 성능을 보였다<br> => R-CNN을 이용했을 때 이런 OverFeat 보다 우수한 성능을 보였다(mAP 31.4%(R-CNN) : 24.3%(OverFeat))<br><br>
                1. 두 번째 과제
                    - 큰 CNN을 training 하기에는 데이터의 양이 부족한 경우 관습적인 해결책은 unsupervised pre-training 이다<br> (참고) unsupervised pre-training : SGD optimization에서 보통은 weight를 random으로 설정하여 cost함수를 최소화 하도록 학습이 진행되는데, Deep network에서 이러한 방식은 성공적이지 못함<br> 따라서 각 층에서 pre-train된 weight를 사용해서 auto encoder처럼 동작하게 함 
                    - 본 논문에서 특정 domain에서 작은 dataset을 fine tunning 하고, 거대한 보조 dataset 에 대해서 unsupervised pre-training 하는 방식을 선보임
                        - data가 부족한 경우 high-capacity CNN에 효과적<br>high-capacity : model parameter를 증가시킨 network -> 더 복잡한 함수를 구현할 수 있지만, Over fitting 문제를 야기할 수 있음
                    - 위 방법을 통해서 2010년 mAP 54%를 달성 ; HOG 기반의 DPM(Deformable Part Model) 은 33%<br><br>
1. Object detection with R-CNN
    1. Module design
        - Region Proposals
            - 다양한 최근 paper들은 category와 독립적인 region proposal 방식을 제공한다
        - Feature extraction
            - open source 중 Caffe 를 사용하여 각 region proposal 로부터 4096 차원의 특징 벡터를 추출
            - 227 X 227 의 RGB 이미지는 5개의 convolution layer를 거치고, 출력된 특징들은 foward propagation 하여 mean-subtracted 방식으로 계산된다
            - region proposal 의 특징들을 검출하기 위해서 우선적으로 CNN에 호환이 되는 형태로 변환이 필요하다
            - 논문에서는 변환 방법들 중 가장 간단한 방법을 택했다<br>size 나 candidate 의 aspect ratio 에 관계없이 모든 pixel 들을 필요한 크기의 tight bounding box 로 warp 하였다<br> 이때 warp 되는 이미지는 원래 box의 p pixel을 차지하도록 했다(논문에서는 16 사용)<br><br>
    1. Test-time detection
        - test time에서 test 이미지로부터 약 2000개의 regioin proposal을 추출하기 위해서 selective search 를 하였음<br><br>
        - Run-time analysis : 다음의 2가지가 detection을 더욱 효과적으로 만듦
            1. 모든 CNN parameter 가 모든 category에서 공유되도록 함
                - parameter를 공유했을 때 결과는 region proposal 과 특징을 계산하는 시간이 모든 class에 대해서 분할 및 감소했다
                - 특정 class 에서 feature 와 SVM weight 간에 내적이 이뤄진다
                - 실제로 모든 image의 내적은 행렬 간 곱으로 묶여진다<br>feature matrix 는 주로 2000 X 4096 , SVM weight matrix 는 4096 X N (N = class 의 수)
            1. 특징 vector들이 다른 접근 방법들(ex. ppyramids with bag of visual word encoding)에 비해서 낮은 차원이었음<br> 예를 들면, UVA detection에서 사용된 feature의 차원의 자리수가 더 높았음(360K dimension vs. 4K dimension)
            - 이러한 분석은 마치 hashing 처럼 R-CNN이 비슷한 것끼리 재정렬 없이 1000개의 object class 의 크기를 조정할 수 있다는 것을 보여준다
            - 만약 class 수가 100K 일지라도 현대의 multi core cpu 에서 10초면 계산이 완료된다
            - 이러한 효율은 단지 region proposal 과 feature share 때문만은 아니다
                - UVA system 에서 높은 차원의 특징으로 인해서 134GB의 메모리가 필요하다( 본 논문에서는 낮은 차원의 feature로 인해서 1.5GB 필요 )<br><br>
    1. Training