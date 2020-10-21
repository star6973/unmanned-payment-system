# Joint Face Detection and Alignment using Multi-task Cascaded Convolution Networks
## 논문 리뷰
- Abstract
    - Face Detection and Alignment 는 1) 다양한 Poses 를 고려, 2) Illumination & Occlusion 문제에 대한 제약을 해결해야 하는 도전과제가 존재하는 분야
    - 최근 Deep learning 의 활용으로 Face Detection 과 Alignment 성능이 향상됨<br><br>
    - 본 논문에서는, "Deep cascaded multi-task"라는 Framework를 제안하고, 각각의 perfomance를 끌어올리기 위해서 내재된 correlation을 활용함
    - 특히, Framework는 3개의 신중하게 설계된 deep conv.network 와 cascaded structure를 적용 => "coarse-to-fine manner" 방식으로 face 예측 및 landmark 의 location 예측하도록 함<br><br> < Coarse-to-fine manner > <br> : 낮은 해상도에서 높은 해상도로 점진적으로 증가시켜가며 적용하는 방식. 이후 더 좋은 결과를 보인 image를 propagate 함<br><br><br>
1. Introduction
    - Face detection & alignment 는 face application 에 필수적인 필수적이지만, 광범위의 variation of face (ex 1) occlusions 2) 다양한 pose variation 3) extreme lighting) 로 인해서 굉장히 어려운 문제
    - 이에 "P.Viola et at. <Robust real-time face detection>" 논문에서는 cascaded classifier를 사용하여 real time에서 좋은 성능을 보임<br>그러나, 몇몇 논문에서 이러한 detector는 현실 속 사람 얼굴에서의 다양한 variation에 대해서는 성능이 낮다는 것을 보여줌
    - cascade structure 외에도 는 몇몇 논문에서 DPM(Deformable Part Model)을 소개하고, 좋은 성능을 보이는데 성공함
    - 그러나, DPM 방식도 단점을 갖고 있었는데, 그것은 바로 연산량이 많다는 점<br><br>
    - CNNs 등장 이후 많은 연구에서 CNN 기반으로 Face detection 접근을 시도
        - S. Yang et al. “From facial parts responses to face detection: A deep learning approach”
            - deep CNN 을 훈련시켜, face 예측지점에 window를 생성하는 방식으로, 얼굴에 대한 높은 응답을 얻으려고 시도
            - 그러나, CNN의 복잡성으로 인해서 이 과정에는 많은 시간이 소요됨
        - H. Li et al. “A convolutional neural network cascade for face detection”
            - cascaded CNN 방식을 적용하여 face detection
            - 그러나, 이 방식은 bounding box에 대한 calibration 이 필요하며, 이것은 많은 연산량을 요구하고 facial landmarks의 위치 간 내재된 correlation 를 무시함
    - 또한, Face alignment 분야에서도 Deep learning 을 활용한 연구들이 많이 이뤄짐
    - 그러나, Face Detection 과 Alignment 의 많은 방법들은 Detection 과 Alignment 사이의 내재된 correlation을 무시함
    - 따라서 이러한 문제를 해결하기 위한 여러 연구 역시 있음
        - Chen et al. “Joint cascade face detection and alignment,”
            - features 의 픽셀 값 차이를 사용한 Random forest 방식
        - zhang et al. “Improving multiview face detection with multi-task deep convolutional neural networks,”
            - multi-task CNN 을 사용하여 multi-view face detection의 정확도를 향상시키려고 함
            - 그러나, detection 정확도는 weak face deector로부터 생성된 초기 detection windows로 인해서 제한된 값을 가짐<br><br>
    - 한편, training 과정에서 어려운 sample을 사용하는 것은 detection을 강하게 할 수 있기 때문에 중요함
    - 그러나, 정통적인 방식의 sample mining 은 offline 방식으로 진행되는 경우가 일반적임
    - online 방식으로 어려운 sample mining 을 한다면, Face detection & alignment 에서 자동으로 training 과정에 대한 적응이 이뤄지기 때문에 더욱 바람직하다<br><br>
    - 본 논문에서, Face detectio 과 alignment 통합을 위해서 정의되지 않은 cascaded CNNs 가 multi-task learning 방식으로 사용된 Framework를 제안
        - 먼저, 얕은 CNN을 통해서 빠르게 후보 windows를 생산
        - 그 다음, 깊은 CNN을 통해서 얼굴이 아닌 곳에 대한 window를 생산하고, 첫 번째 단계에서 생성된 window를 제거
        - 최종적으로, 더욱 powerful CNN을 사용하여 결과 및 Face landmark 위치에 대한 정제 과정을 거침<br><br>
    - Contribution
        1. 새로운 cascaded CNNs 기반의 framework 를 제안하여 face detecion 및 alignment 수행
            - 또한 real-time 을 위한 light CNN 구조를 설계
        1.  성능 향상을 위해서 online 방식의 hard sample mining 제안
        1. 확장된 실험을 통해서 sota의 detection 과 alignment 방식과는 비교되는 방식으로 성능 향상을 이룸(?) <br><br><br>
1. Approach
    - Overall framework
        1. <br>![Approach1](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/Approach1.JPG?raw=true) <br>입력 Image에 대해서 각각 다른 사이즈로 resize 하여 Image Pyramid 만듦
        1. <br>![Approach2](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/Approach2.JPG?raw=true) <br>P-Net(Proposal Network) 이라는 FCN 을 적용하여 예상 windows를 얻고, 이로부터 bounding box regression 의 vectors를 계산<br>(S. S. Farfade et al. “Multi-view face detection using deep convolutional neural networks” 과 유사한 방식 적용)<br>==> 이로부터 bounding box regression을 calibration 하고, NMS(Non-maximum Suppression) 방식으로 많이 겹쳐진 box들을 병합
        1. <br>![Approach3](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/Approach3.JPG?raw=true) <br>모든 예상 box들은 R-Net(Refine Net)에 의해서 거짓 box들이 제거하여 box의 regression을 calibration하고, NMS box들이 병합
        1. <br>![Approach2](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/Approach4.JPG?raw=true) <br>b번째 과정과 유사하며, Face를 더욱 정밀하게 묘사하는 것에 초점을 두고, 5개의 Landmark를 출력함<br><br>
    - CNN Architecture
        - H. Li et al. “A convolutional neural network cascade for face detection” 의 논문에서는 Face Detection 을 위해서 multiple CNNs 설계함
        - 그러나, 이것은 다음 2가지에 의해서 성능에 제한이 따름<br>  1) 일부 filter 의 weight diversity 가 부족 ==> 다양한 변별럭을 가지는 묘사를 생산하는데 제한이 있음<br>  2) 다른 multi-class object detection & classification 과 비교하면, face detection은 2진 분류 분야에 대한 도전임<br>따라서 더 적은 filter가 필요하지만, 구별 능력은 더 높아야 함
        - 본 논문에선, filter의 수를 감소하고, 연산량 감소를 위해서 5x5 flter를 3x3 filter로 변경하고, 반면 더 좋은 성능을 위해서 depth는 증가시킴
        - 이런 시도로 인해서 H. Li et al. 의 논문에 비해서 더 적은 runtime 이 요구됨<br>아래 표에 본 논문의 방식과 [19 ; H. Li et al.] 의 방식을 비교함 (공평한 비교를 위해서 동일 데이터 사용)<br> ![Table1](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/Table1.JPG?raw=true)
        - 또한 이렇게 설계된 본 논문의 CNN architecture는 다음과 같음<br> ![CNN Architecture](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/CNN%20architecture.JPG?raw=true) <br><br>
    - Training
        - "face/non-face classification / bounding box regression / facial landmark localization" 의 3가지 이점을 사용하여 본 논문의 CNN detector를 train 진행함
            1. Face classification
                - objective learning 은 2개 class의 classification 문제로 공식화 할 수 있음. 이를 위해서 각 sample ![x_i](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/X_i.JPG?raw=true) 에 대해서 cross-entropy loss를 사용<br> ![train_1](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/train_1.JPG?raw=true) <br>위 식에서 ![P_i](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/P_i.JPG?raw=true) 는 얼굴이라고 network가 예상한 확률을 의미<br> ![Y_det](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/Y_det.JPG?raw=true) 는 ground-truth label을 의미
            1. Bounding box regression
                - 각 예상 window 와 그 인접한 ground-truth 사이의 offset 을 예측
                - objective learning 은 regression 문제로 공식화 할 수 있고, 이를 위해서 Euclidean loss 를 사용<br> ![Train_2](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/train_2.JPG?raw=true) <br> ![Y_hat](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/Y_hat.JPG?raw=true) 은 regression target으로 network에 의해서 얻어지며<br>ground-truth의 좌표를 의미<br>각 좌표는 좌상단, 높이, 너비를 포함하므로 ![R^4](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/R%5E4.JPG?raw=true) 의 공식이 성립
            1. Facial landmark localization
                - bouding box의 regression task 와 유사하게 regression 문제로 공식화 할 수 있고, 다음의 Euclidean loss를 최소화함<br> ![Train_3](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/train_3.JPG?raw=true) <br> ![y hat landmark](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/y_hat_landmark.JPG?raw=true) 는 facial landmark 의 좌표이며, network에 의해서 얻어짐.<br> ![y^landmark](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/y%5Elandmark.JPG?raw=true) 는 ground-truth의 좌표이고, "왼쪽눈, 오른쪽눈, 코, 왼쪽 입끝, 오른쪽 입끝"의 5개 landmark가 있으므로 ![R^10](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/R%5E10.JPG?raw=true) 로 표현됨<br><br>
            1. Multi-source training
                - 각각 다른 task에 대해서 다른 CNN을 적용했기 때문에 각 learning 과정에 대해서 서로 다른 images가 존재 (예를 들면, face, non-face, partialy aligned face)
                - 이 경우 loss 함수의 일부는 사용되지 않음. 예를 들면, 배경 region의 sample의 경우, ![Y_det](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/Y_det.JPG?raw=true) 만 계산하고, 다른 2개의 loss는 0으로 설정
                - 이것은 sample 형식의 지표에 의해서 직접 실행될 수 있다. 그리고 전체 learning target는 다음의 식으로 공식화된다.<br> ![Train_4](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/train_4.JPG?raw=true) <br>N이 training sample 의 개수일 때, ![alpha_j](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/alpha_j.JPG?raw=true) 는 task 중요도를 의미<br>O-Net을 사용하여 얼굴과 mark의 정확도를 향상시키려고, P-Net은 ![alpha_det](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/alpha_det.JPG?raw=true) = 1, ![alpha_box](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/alpha_box.JPG?raw=true) = 0.5, ![alpha_landmark](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/alpha_land.JPG?raw=true) = 0.5 로 하였고,R-Net은 ![alpha_det](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/alpha_det.JPG?raw=true) = 1, ![alpha_box](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/alpha_box.JPG?raw=true) = 0.5, ![alpha_landmark](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/alpha_land.JPG?raw=true) = 1 로 하였다
                - 이 경우 CNN train을 위해서 SGD를 적용하는 것이 자연스러움<br><br>
            1. Online ard sample minig
                - 원래 방식으로 classifier가 훈련되고, 전통적인 hard sample minig 방식을 적용하던 것과는 다르게, 본 논문에서는 online hard sample minig을 적용하여 face classification 수행
                - 특히, 각각 mini-batch에서 순전파에 의해서 계산된 loss를 정렬하여, 상위 70%를 hard sample로 선정
                - 역전파 계산에서 위에서 선택된 hard sample에 대해서만 gradient를 계산
                - 이것은 easy sample을 무시하여 Network 가 weak하지 않도록 하기 위함이며, Experiment 에서 확인할 수 있듯이 이러한 전략은 manual sample selection 없이도 좋은 성능을 낼 수 있도록 함<br><br><br>
1. Experiment
    - 이 장에서는 본 논문에서 제안된 hard sample mining 의 효율성에 대해서 평가한다
    - 다음 FDDB, WIDER FACE, AFLW 에 대해서 본 논문의 모델과 다른 sota 모델들의 성능을 비교한다<br><br>
    1. Training Data
        - Negatives : IoU 가 0.3 이하인 경우
        - Positive : IoU 가 0.65 이상인 경우
        - Part faces : IoU 가 0.4 ~ 0.65 사이값을 가지는 경우
        - Land mark faces : 5개 landmark의 위치가 label 표시된 face<br><br>
        - Negatives + Positives : classification task에 사용
        - Positives + Part faces : bounding box regression에 사용
        - Landmark faces : landmark를 localization 할 때 사용<br><br>
        1. P-Net
            - WIDER FACE에서 무작위로 crop하여 positives, negatives, part face 를 수집
            - 이후 CelebA로부터 face를 crop하여 landmark faces로 사용
        1. R-Net
            - P-Net을 사용하여 WIDER FACE 에서 positives, negatives, part face 를 detect하고, 동시에 CelebA로부터 face landmark를 detect
        1. O-Net
            - R-Net 과 유사하지만, P-Net, R-Net을 모두 사용<br><br>
    1.  The effectiveness of online hard sample minig
        - O-Net을 Online hard sampling 을 적용한 경우와 적용하지 않은 경우에 대한 loss curve 비교
        - 더욱 직접적인 비교를 위해서 O-Net을 classification에 대해서만 train
            - networ 초기화를 비롯한 모든 train parameters는 위 두 가지 경우에서 동일
        - 더욱 취운 비교를 위해서 고정된 learning rate를 사용
        - hard sample mining이 더욱 성능을 높일 수 있다는 결과를 보였으며, 그 그래프는 아래와 같음<br> ![fig3_a](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/fig3_a.JPG?raw=true) <br><br>
    1. The effectiveness of joint detection and alignment
        - 다음은 Joint detection and alignment를 적용했을 때 더욱 높은 성능을 보인 것을 확인<br> ![fig3_b](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/fig3_b.JPG?raw=true) <br><br>
    1. Evaluation on face detection
        - WIDER FACE에 대해서 본 논문의 모델과 sota 모델들을 비교하였으며, 다른 모델에 비해서 큰 성능 향상을 보임<br> ![fig4_atod](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/fig4_atod.JPG?raw=true) <br><br>
    1. Evaluation on face alignment
        - RCPR, TSPM, Luxand face SDK, ESR, CDM, SDM, TCDCN 에 대해서 성능을 비교했고, Test 과정에서 본 논문의 모델이 detect 실패한 13장의 face에 대해서<br>중간 지역을 crop하여 다시 test를 진행했고, 높은 성능향상을 확인함<br> ![fig4_e](https://github.com/star6973/lotte_studying/blob/KwonHH/reference_image/KwonHH/Joint%20Face%20Detection%20and%20Alignment%20using%20Multi-task%20Cascaded%20Convolutional%20Networks/fig4_e.JPG?raw=true) <br><br>
    1. Runtime dfficiency
        - 2.6GHz CPU 에서는 16fps , Nvidia titan black 에 대해서는 99 fps<br><br><br>        
1. Conclusion
    - 본 논문에서는 joint detection and alignment 기반의 multi-task cascaded CNN 방식의 framework를 제안함
    - 제안된 framework를 사용하여 여러 Dataset에 대해서 sota 모델에 비해서 높은 성능향상을 보임
    - 추후 face detection 과 다른 face 분석 작업 사이의 내재된 상관 관계를 개척할 예정 