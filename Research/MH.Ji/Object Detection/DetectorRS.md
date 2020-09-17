# 논문 내용 정리(DetectorRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution)
## 1. Abstract
- 본 논문에서는 object detection을 위한 backbone 디자인을 탐구할 것이다.
- Macro Level(모델 전체 구조)에서는, FPN(Feature Pyramid Network) 구조에 추가적인 피드백 커넥션을 통합시킨 [RPN(Recursive Feature Pyramid)]를 제안한다.
- Micro Level(모델 세부 구조)에서는, 서로 다른 atrous rates와 switch functions을 사용해 결과를 취합시킨 [Switchable Atrous Convolution]을 제안한다.
- 위의 두 모델을 합친 것이 DetectorRS이며, object detection의 성능을 향상시켰다.

<br><br>

## 2. Introduction
- two-stage object detection 방식으로 유명한 Faster R-CNN은 object를 감지하기 위해 regional features를 기반으로 object proposal을 출력한다.
- 같은 방식으로, Cascade R-CNN은 multi-stage object detection 방식으로 후속 검출기의 헤드는 더 많은 선택적 예시와 함께 훈련된다.
- Macro Level
    + FPN 위에다가 FPN layer에서부터 bottom-up backbone layer까지의 추가적인 피드백 커넥션들을 통합시킴으로써 구축한 RFP 모델을 제안한다.
    
    + RFP 모델은 반복적으로 FPN의 성능을 향상시켜주기 위해 강화시킨다.
    
    + Resembling Deeply Supervised Nets 논문에 따르면, 피드백 커넥션은 디텍터 헤드로부터 직접적으로 gradients를 받아온 features를 낮은 레벨의 bottom-up backbone에 돌려줌으로써, 훈련과 부스트 퍼포먼스의 속도를 높여준다.
    
- Micro Level
    + 다른 atrous 비율을 가진 같은 input feature를 서로 얽히고, switch function을 사용해 결과를 얻는 SAC(Switchable Atrous Convolution)를 제안한다.

    + switch function은 공간적으로 독립적이고, 각 feature map의 location은 SAC의 출력 결과를 제어하기 위한 서로 다른 switches이다.

    + 디텍터로서 SAC를 사용하기 위해서, 3x3 convolution layer를 SAC의 bottem-up backbone으로 변환했다. 이렇게 하면 large margin에 의해 디텍터의 퍼포먼스가 향상된다.

    + 이전의 방식들은 

<br><br>

## 3. Related Works


<br><br>

## 4. Recursive Feature Pyramid
### 4.1. Feature Pyramid Networks



### 4.2. Recursive Feature Pyramid



### 4.3. ASPP as the Connecting Module


### 4.4. Output Update by the Fusion Module


<br><br>

## 5. Switchable Atrous Convolution
### 5.1. Atrous Convolution


### 5.2. Switchable Atrous Convolution

### 5.3. Global Context

<br><br>

## 6. Experiments


