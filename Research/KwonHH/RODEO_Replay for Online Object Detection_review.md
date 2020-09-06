# RODEO : Replay for Online Object Detection
## 논문 리뷰
- 용어 참고<BR>agent : 딥러닝 에이전트는 딥러닝을 활용해 업무 수행과 개선을 수행하는 임의의 자율적 또는 반자율적 AI 주도 시스템이다.<br>iid : independent identically distributed <br><br>
- Abstract
    - 본 논문은 연산시간과 저장공간의 제약이 따르는 Online Streaming 상황에서 Object Detection 방법을 개척
        - Object Detection 은 모든 바운딩 박스에 대해서 올바르게 이미지 라벨을 보여줘야 함
        - 본 논문에서는 이전의 연구와는 다르게  시간이 지남에 따라 소개된 새로운 class를 사용해 온라인 방식으로 연구를 할 수 있었음<br><br>
1. Introduction       
    - Object Detection 개념 : 대상에 바운딩 박스를 적용해 label class를 예측하고, 위치까지 찾아내는 것
        - 기존 Detection의 경우 <u>오프라인에서 적용</u> 되었기 때문에 새로운 object class의 업데이트가 불가능
        - 반면, 인간 및 포유류는 한 번에 정지되지 않은(유동적인) 샘플로부터 더 많은 부분을 학습하고 이해할 수 있음<br> ==> <u>__Streaming Learning__</u> <u>__(= Online Learning)__</u>
        - 그러나, 통상적인 모델에 __Streaming Learning__ 을 적용하면 __Catastrophic Forgetting__ 문제가 발생<br>***<br> Catastrophic Forgetting 란? 인공신경망이 단일 작업에 대해서는 뛰어난 성능을 보이지만, 다른 종류의 작업을 학습하면 이전 학습 내용을 잊어버리는 것<br><br>
        - Streaming Learning 은 새로운 class를 추가하거나, 계절 등 시간 경과의 따라 객체를 포함시키는 어플리케이션에 적용될 수 있는 방식이다
            - 기존의 Incremental Object Detection 방식은 상당한 한계를 가지고 있으며, Streaming Learning 이 불가능
            - 기존의 Incremental Object Detection 방식 : 현재의 장면을 바로 업데이트하지 않고, <u>큰 배치사이즈로 이미지를 업데이트 하는 방식을 사용</u><br>(이런 시스템은 forgetting 문제를 완화하기 위해서 distillation 방식을 사용)
            - 즉, 시간 t일 때의 batch에서 학습이 발생하기 전에 batch 내의 모든 장면들을 예측해야만 하며, 이후 batch에서도 이러한 상황이 반복된다 <br><br>
        - 이전의 Incremental 이미지 인식 연구에서 Replay 메커니즘이 Catastrophic Forgetting을 완화하는데 효과적임을 보여준다
            - Replay 아이디어는 인간이 어떻게 기억들간에 관계를 짓고, 그것이 강화되는지에서 영감을 얻었다(대뇌에서 일어나는 현상에서 영감을 얻음)
            - 추가적으로 Hioopcampal Indexing 이론에 따르면 인간은 압축된 기억을 기억해낼 때, 인덱싱을 메커니즘을 사용하는 것으로 알려졌다<br>반면 다른 연구들에서는 가공되지 않은 sample을 replay하였으며, 이는 생물학적으로 합리적이지 않다고 언급함
            - 즉, RODEO 에서는 고정된 용량의 buffer로부터 압축된 데이터를 replay하고 이를 점진적으로 Object Detection 수행
            - 이렇게 했을 경우 <u>효율적인 연산</u> 및 <u>다른 어플리케이션으로 확장이 용이</u>한 장점이 있음을 확인<br><br>
        - 본 논문이 기여한 내용
            1. Streamin Learning 방식으로 Object Detection 수행 및 강력한 Baseline을 세움
            1. RODEO 모델을 제안 : Replay 방식을 통해 Catastrophic Forgetting 완화, Incremental batch object detection 알고리즘보다 더 우수한 성취 결과<br><br><br>
1. Problem Setup
    1. Continual Learning
        - Continual Learning (= Incremental batch learning)은 Streaming learning보다 훨씬 쉬운 문제이며, 최근 Classification 과 Detection 에서 큰 성공을 거두었다
        - Continual Learning 에서는 T 사이즈로 나눠진 batch가 필요
        - 각 반복횟수 t에서 batch만큼 반복적으로 학습
        - 이는 real-time에 이상적인 방식이 아니며 그 이유는 2가지
            1. 학습이 발생하기 전에 batch 데이터가 축적되기를 기다려야 함
            1. batch의 반복학습이 끝나야만 evaluate 할 수 있음
    1. Streaming Leaarning
        - 최근에서야 Image classification 에 쓰이기 시작했으며, Object Detection 에서는 사용되지 않았고, 본 논문에서 그 길을 개척
        - <u>Streaming Object Detection</u> 은 Training 동안 dataset으로부터 임시로 순서가 정해진 바운딩 박스 이미지와 그것의 라벨을 받는다
        - 시간 t에서의 이미지 I_t가 주어졌을 때 모델은 evaluation 동안 모델은 이미지 내에서 라벨이 표시된 바운딩 박스를 반드시 생성해야 한다
        - Streaming Learning은 모델에게 독특한 도전을 요구 : 전체 dataset에서 single epoch 내에 하나의 example을 학습하도록 요구함
        - Streaming Learning 에서 모델 evaluation 은 traing 중 어디서든 발생할 수 있다
        - 또한, 개발자는 memory 와 time에 제약을 가함으로써 real-time에서 학습할 수 있는 능력을 높일 수 있음<br><br><br>
1. Related Work
    1. Object Detection
        - Image Classification 에서는 agent가 이미지 안에 있는 '무엇'에 대해서 대답하는 반면, Object Detection은 추가적으로 agent가 localization 까지 요주됨(즉, '어디' 에도 대답할 수 있어야 함)
        - 게다가 모델은 여러 object 에 대해서 localization 가능해야 하며, 종종 이미지 내에 다양한 category 가 있어야 함
        - 최근 2가지 타입의 Architecture가 제안되면서 위의 문제를 지적하였음
            1. Single Stage Architectures ; SSD, YOLO, RetinaMet
                - end-to-end network : proposal box를 생성하고, 다수 class를 인식하는 바운딩 박스를 생성하고, 각 박스를 single stage로 구별
                - Train 속도가 빠른 반면 2 Stage Architecture에 비해서 성능이 낮음
            1. 2 Stage Architectures ; Fast RCNN, Faster RCNN
                - 최초로 Region proposal 을 사용함
                - Region proposal 박스는 classification 에 사용되고, 바운딩 박스의 좌표는 regression을 통해서 더욱 미세조정 됨
                - 모든 detection 모델의 출력은 각 category 와 가장 유사할 확률 점수를 포함하며, 바운딩 박스의 좌표로 나타냄
                - Incremental Object Detection 이 최근 Continual Learning 양식에서 발견되었지만, 논문 저자들은 Streaming Object Detection의 길을 개척하고, 이로부터 더욱 현실적인 시스템 실현<br><br>
    1. Incremental Object Recognition
        - Continual Learning 방식이 Streaming Learning 방식에서 쉬운 방식이긴 하지만, 두 양식 모두 Train 대상이 바뀌면 Catastrophic Forgetting 문제가 발생
        - Catastrophic Forgetting 은 <u>Stability-Plastic Dilemma (안정성-가소성 딜레마)</u>
            - Stability-Plastic Dilemma : 인공신경망이 기존에 학습된 내용을 유지하면서도 새로운 내용을 학습해야할 때 발생하는 딜레마.<br>
                1. Neural Network 의 Stability 에 중점을 둔 경우<br>![Stability-Plastic Dilemma_stability](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/RODEO_Replay%20for%20Online%20Object%20Detection/Stability-Plastic%20Dilemma_stability.JPG?raw=true) <br>
                    - 그림1은 시간의 경과(왼쪽->오른쪽)일 때 Node 사이의 Weight가 변하지 않고 유지
                    - 그림2는 시간의 경과에 따라 Node 사이의 Weight가 약해지지만 다시 원상태로 돌아오는 모습<br> => 이런 경향이 Stability
                1. Neural Network 의 Plastic 에 중점을 둔 경우<br>![Stability-Plastic Dilemma_plastic](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/RODEO_Replay%20for%20Online%20Object%20Detection/Stability-Plastic%20Dilemma_plastic.JPG?raw=true) <br>
                    - 그림3은 핵심 node는 그대로 유지되고 있지만, 학습이 진행되면서 Weight가 변하면서 Node 간 연결이 강해졌다, 약해졌다 변화함
                    - 즉, Node 에 저장된 메모리는 유지되지만 weight가 변화하면서 저장된 값의 변화가 생기고, 기존의 값이 손상됨<br>
                - 학습된 기억을 지속적으로 유지하는 것도 중요(Stability)하지만, 새로운 내용을 학습함에 있어서는 Weight를 조정함(Plastic)으로써 약간의 변형이 불가피
                - 즉, 우수한 신경망을 만들기 위해서는 Stability / Plastic 어느쪽도 우세해서는 안되므로 빠지게 되는 Dilemma<br>[출처]:https://elecs.tistory.com/295
            - Forgetting을 극복할 수 있는 방법이 몇 가지가 존재
                1. Weight 업데이트에 제약을 두어 정규화하는 방법
                1. 간섭 현상을 완화하기 위해서 Network 의 Weight 업데이트를 드문드문 하는 방법
                1. 여러개의 classifier 사용하여 앙상블 하는 방법
                1. 이전 training 의 input의 일부분을 저장하고, 새로운 network가 업데이트 되었을 때 함께 mix하여 => model을 rehearsal/replay 하는 방법
                - 많은 이전의 연구들은 이러한 방법들을 혼합하였고, 특히 __replay 방법은 Sota 의 image recognition에서 많은 모델을 생산함__
    1. Incremental Object Detection
        - Streaming Object Detection이 발견되지 않았을 때, Continual learning 양식이 사용되어왔다
        - Konstantin 등 3명의 2017년 ICCV 에 발표된 "Incremental learning of object detectors without catastrophic forgetting" 논문에서 replay 방식을 사용하지 않고 distillation 기반의 방식이 제안되었음
            - 네트워크는 초기에 class의 일부분에 대해서만 train 되고, Weight 값은 frozen시키며, 새로운 class에 대한 parameter와 함께 직접 새로운 network로 복사
            - 기본적인 Cross-entropy loss는 추가적으로 frozen network에서 weight의 변동을 제한하기 위해서 사용됨
            - Yu Hao et al [An end-to-end architecture for class-incremental object detection with knowledge distillation(2019)] 에서 고정된 proposal이 발생하는 문제를 극복하려 함
                - 자세한 설명은 무슨말인지 이해를 못하겠음...
            - 유사하게, Dawei Li et al. [near real-time incremental learning for object detection at the edge Symposium on Edge Computing, 2019] 에서는 classification prediction 과 bounding box 좌표, end-to-end incrementl network를 훈련하기 위한 feature 등을 distillation 함
            - 추가적으로 Shin et al. [Incremental deep learning for robust object detection in unknown cluttered environments. IEEE Access, 6, 2018]에서는 새로운 incremental 프레임워크를 사용하여 지도 학습과 유사한 방법으로 active learning 을 통합함<br>
        - 위에 언급한 모든 방법은 일괄적으로 작동하며, <u>__한번에 하나의 example을 학습하도록 설계되지 않음__</u><br><br><br>
1. Replay for the Online Detection of Objects(RODEO)
    - RODEO 는 Tyler L Hayes et al. [Remind your neural network to prevent catastrophic forgetting. In ECCV, 2020] 에서 영감을 얻음
    - RODEO는 example을 한 번에 하나씩 학습하는 방법이 적용된 Online Object Detection
        - 즉, 새로운 대상이 관찰되자마자 새롭게 업데이트 가능하며, __incremental batch 방식보다 real-time applicatin에 적용하기 더 좋음__
    - Online 학습을 용이하게 하기 위해서 메모리 버퍼를 사용하여 이곳에 representation의 example들을 저장함
    - 이러한 representation들은 CNN backbone의 중간 계층에서 얻어지며, 저장 공간을 줄이기 위해서 압축이 이뤄짐
    - Training 동안 RODEO는 새로 입력된 이미지를 압축함
        - 그 다음 이 새로운 입력을 재생 버퍼에서 임의로 재구성된 샘플 subset 과 결합한 후 이 재생 mini-batch로 모델을 업데이트
    - 정식으로 설명하면, H(RODEO Model) 는 다음과 같이 표현될 수 있다
        - H(x) = F(G(x))    : 여기서 x는 입력 image 를 의미  / G 는 이전 CNN layers 로 구성 / F 는 나머지 layers
        1. 먼저 dataset 전체 class 중 절반만 offline에서 train 된 model에서 base initailization 단계에서 G를 초기화
        1. 위 단계 이후 CNN 의 이전 layers 가 일반적이고 전이가능한 representation들을 학습하기 때문에 G 의 layers는 frozen
        1. 이후 streaming learning 동안 오직 F만 plastic 상태로 유지되고, 새로운 데이터가 업데이트 됨
    - replay buffer 안에 가공되지 않은(pixel 단계) samples 를 저장하던 이전의 Incremental image recognition 방식과는 다르게,<br>RODEO는 압축된 feature map의 representations를 저장
        - 이것의 장점은 필요한 저장 공간을 대폭 감소할 수 있음
        - 특히 입력 이미지 x에 대해 G(x)의 출력은 feature map(z) 인데, 이것의 사이즈는 p x q x d 이고, p x q 는 spatial grid 의 크기를 의미, d는 feature 의 차원을 의미
        - G 가 base initialization 단계에서 초기화 되면, G를 통해서 모든 base initialization samples 를 밀어넣고, feature map을 얻는다
        - 그리고 그것은 product quantization(PQ)를 train 할 때 사용된다
        - PQ model 은 각 feature map의 tensor를 p x q x s 의 integer 배열로 encode 함. 이때의 s는 저장공간에 필요한 인덱스의 수
        - PQ model 의 train이 끝나면, 모든 base initialization samples 로부터 압축된 representation를 구하고, memory replay 버퍼에 그 값을 더해줌
        - 이어서 새로운 examples를 H로 각 time 당 하나씩 stream
        - 새로운 sample을 PQ model을 사용해서 압축하고, 메모리 버퍼로부터 무작위로 reconstruct , 그리고 F를 업데이트 함
        - replay 버퍼를 메모리 상한값으로 설정
        - 만약 메모리 버퍼가 꽉 차면, 새로운 압축된 sample 이 추가되고, 기존의 sample을 제거함
        - 그렇지 않으면 새로운 압축 sample을 직접 더해줌
        - 모든 실험에서 codebook의 index들을 저장하여 8비트로 사용
        - COCO 에서는 64개, VOC 에서는 32개의 codebook 사용
        - PQ 연산을 위해서는 Faiss library를 사용<br><br>지금까지의 과정이 모두 설명된 알고리즘 사진은 다음과 같음<br>![Algorithm1_Incremental update procedure for RODEO on COCO](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/RODEO_Replay%20for%20Online%20Object%20Detection/Algorithm1_Incremental%20update%20procedure%20for%20RODEO%20on%20COCO.JPG?raw=true) <br><br>
        - 평생의 학습 agent 에서는 학습을 위해서 무한한 data stream이 필요하고, 이것은 이전의 examples를 memory replay buffer에 모두 저장하는 것이 불가능<br>반면 RODEO에서는 버퍼 사이즈가 고정되어있기 때문에 시간에 따라서 덜 필요한 examples를 제거하는 것이 필수적<br>=> 따라서 replay buffer로부터 이미지의 최소 라벨수가 가장 적은 이미지를 대체하는 replacement 전략을 사용
1. Experimental Result<br>
![table1](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/RODEO_Replay%20for%20Online%20Object%20Detection/Table1_mAP%20result%20for%20VOC%20and%20COCO.JPG?raw=true) <br>** Real feature는 Plastic Layer(F)를 지나기 전에 Reconstruction(Recon)을 하지 않았음<br><br>
    - mAP를 정규화하기 위해서 VOC 와 COCO 에 대해서 각각 0.42, 0.715 의 mAP를 기록한 모델을 사용
    - VOC에 대해서 RODEO는 4개의 sample만 replay해도 이전 방식이 적용된 모델들보다 모두 결과가 좋았다
    - 또한 다양한 크기로 replay를 해본 결과 Recounstruct 했을 때보다 Real Feature일 때, VOC 와 COCO 모두에서 좋은 결과를 보였다
    - 다음의 그래프에서 알 수 있듯이 다른 모델들보다 RODEO가 forgetting하는 경향성이 적었다<br>![Figure2_Learning Curve for VOC 2007](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/RODEO_Replay%20for%20Online%20Object%20Detection/Figure2_Learning%20Curve%20for%20VOC%202007.JPG?raw=true) <br>
    - SLDA+Regress 모델은 BackBone을 업데이트 할 필요 없이 VOC 와 COCO, 두 데이터셋 모두 놀라울 정도로 경쟁력이 있다
    - RODEO에서 4개의 sample만 replay한 경우 다른 모델들에 피해 큰 폭으로 mAP가 증가함을 확인할 수 있다<br><br>
    1. Additional Studies of RODEO Components
        - Buffer Management 방법을 사용했을 때의 효과를 연구하기 위해 COCO에 다음의 대체 방법들을 적용<br>(BAL : 전체 class 분포에 최소의 영향을 주도록 Balance를 주는 방식 <br>MIN,MAX : label이 가장 많은 경우와 가장 적은 경우의 이미지로 대체하는 방식 <br>RANDOM : 랜덤으로 buffer의 이미지를 대체하는 방식 <br>NO-REPLACE : 모든 sample 저장하고, buffer가 무한으로 확장하도록 하는 방식 )![Tabel2_Incremental mAP results for several variants of RODEO](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/RODEO_Replay%20for%20Online%20Object%20Detection/Tabel2_Incremental%20mAP%20results%20for%20several%20variants%20of%20RODEO.JPG?raw=true) <br>
        - 이상적인 경우 RODEO의 replay 수를 4로, buffer 크기를 무제한(모든 이미지 sample)으로 설정
            - 이 경우 mAP 는 0.928 획득
        - Replacement 방식이 적용된 모델들의 경우 17,668개의 sample만 저장하도록 함
            - 이 중에서 RANDOM 방식과 비교했을 때 MAX 방식에서 mAP가 더 낮았음 : 이것은 더 다양한 고유의 카테고리의 sample을 저장하기 때문
        - MIN 방식을 적용한 경우 real 과 recon 모두에서 가장 좋은 결과
            - 이것은 MIN 방식에서 고유한 image 카테고리를 가장 많이 가지기 때문에, forgetting을 극복하기 위해서 buffer가 더욱 다양해 지는것으로 추측<br><br>
        - VOC 실험에서 buffer에 아무것도 replace하지 않고, replay하는 sample 수를 4 -> 12 로 증가시켰을 때 real feature에서는 0.3%, reconstructed feature에서는 5.3% 의 성능 증가
        - 반면, COCO 실험에서는 buffer에 replacement를 적용했더니, replay 하는 sample수를 증가시켰음에도 성능이 하락
            - COCO 가 VOC 와 비교했을 때 이미지의 배경으로 취급되는 region proposal 내에 더 많은 object가 있기 때문인 것으로 추정<br><br>
    1. Training Time
        
        
        