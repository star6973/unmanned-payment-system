## MTCNN


#### 1. P-NET 
- 300x200 이미지가 입력되어 들어오면 200x166 100x66 30x20 크기로 리사이즈 하여 이미지의 리스트를 만듭니다. 내가 갖고있는 박스 크기에 맞에 얼굴이 다 들어오도록 이미지 입력의 해상도를 늘리거나 줄여서 박스를 확인합니다. 

- [None-fully-connect] = 이것은 위치정보를 잃어버리는것을 방지 하기 위해서 FC레이어를 배제하고 Conv레이어만으로 네트워크를 구성함

![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fp85rD%2FbtqASyia6pf%2FgOfzyYk6A02SXEFkCJB3Hk%2Fimg.png)

- 그 다음 이렇게 찾은 박스들을 대상으로 Non-Maximum-Suppression과 bouding box regression을 적용해줍니다.

#### 2. R-Net 
- 우리는 P-Net으로 얼굴로 추정되는 박스들의 리스트를 얻었습니다. R-NET의 역할은 이 박스들 중에서도 진짜 얼굴에 해당되는 영역들을 추려내고 bounding box regression을 더 정교하게 수행하는 것입니다. 

- R-Net을 통과 시키기 전에 박스들을 24x24크기로 resize를 한 이후에 R-Net을 통과시킵니다. 

![rnet](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcbwlwX%2FbtqAWcyfdyv%2FkNQCkGYC982KSpliEZ283k%2Fimg.png)

- [fully-connect]으로 정답 레이어을 만듭니다. 

- R-Net에서 찾아 낸 박스는 마찬가지로 원래 입력 이미지 크기로 되돌린 다음,NMS와 BBR을 적용시킨 후 살아 남은 박스들을 O-Net으로 전달한다. 


## 3. O-Net

![onet](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbZUfBH%2FbtqAWxa1WWV%2FY1QOKf8jrlYj8vgaV01Z20%2Fimg.png)

- O-Net은 R-Net을 통해 찾아낸 박스들을 모두 48x48 크기로 리사이즈한 입력을 받습니다. 점점 필터를 키우면서 얼굴의 feature level을 높이며 최종 face detection, face alignment 결과값을 찾습니다. 

>> 이 MTCNN을 구성하는 P-Net, R-Net, O-Net은 세 가지 네트워크 모두 구조가 비슷하고 output의 형태도 동일하게 내는 점입니다. 

## Multi Task Loss

- MTCNN은  classification, regression, localization 세가지 테스크에 대해 각각 loss를 구한 뒤 이를 가중치를 두어 합치는 방식을 사용합니다. 

1.    classification
- log loss를 사용합니다. ( cross Entrophy로 봐도 무방합니다)

![class_los](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc3kW0Y%2FbtqAUul3y70%2FLP8rHWa08gFLljvWWVRvqK%2Fimg.png)

2. bounding Box Regression Loss
- 흔히 bbr의 결과값으로 좌측 상단의 점의 x,y좌표의 위치 조절 값과 너비와 높이의 조절 비율을 얻어내며 이는 4차원의 벡터에 해당합니다. 이를 bbr값과 ground truth box와 유클리드 거리를 측정하여 loss로 사용합니다. 

![bbr_loss](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FblBWga%2FbtqAXPPAQba%2FeatqkzcAPrBfGy7Fp6aZD0%2Fimg.png)
![bbr_loss2](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbqCfgT%2FbtqASzIcdRr%2FdNkaZYF5I6pxrVFOdsCyC1%2Fimg.png)


3.  localization loss
-  얼굴의 각 주요지점을 나타내는 좌표는 x,y로 이루어져 있으며 총 다섯개의 지점을 예측합니다. 그 결과 10차원의 벡터라고 볼 수 있으며, 앞선 regression loss와 비슷하게 계산합니다. 
![local_loss](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdPawka%2FbtqATgn3GYY%2F0OO9wgRvuGeN2KAfVNk840%2Fimg.png)

4. 위의 3가지 로스를 각각의 로스에 가중치를 반영하여 모두 합쳐준 것이며, P-Net, R-Net, O-Net 각각 가중치를 주어서 계산합니다. 아래에 N은 전체 데이터 수이며 베타는 특정 샘플이 선택되었는지 안되었는지를 (0,1)로 표현합니다. Stochastic Gradient Descent를 적용하여 전체 데이터 셋에서 일부분만 뽑아내어 로스를 계산한다는 의미입니다.
![loss_weight](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbRwDo2%2FbtqAWb68wlg%2FAXgNvUFnSu1cGWeN0g5u6K%2Fimg.png)
![loss_weights](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fp8WyT%2FbtqAVHd8TmL%2FCGrsny3grfNK31yEo1bVhk%2Fimg.png)

## 한계점 및 개선 방향
- 지금까지 MTCNN을 살펴보았습니다. 뛰어난 모델이지만 분명히 개선할 점이 있다고 생각합니다. 먼저 입력 이미지에서 피라미드를 만들고, 이를 12x12 크기의 윈도우로 CNN inference를 하는 부분이 가장 큰 bottle neck으로 작용할 수 있습니다. 이미지의 크기가 커질 수록 CNN inference의 수가 급격히 늘어나게 되고 이는 곧 성능 저하로 이어집니다. 실제로 이용해본 결과 1080x720의 고해상도 이미지를 입력하면 GPU를 사용함에도 10초가 넘게 걸리는 현상을 경험했었습니다.



