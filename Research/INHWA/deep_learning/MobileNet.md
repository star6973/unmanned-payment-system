## MobileNet, 2017


#### Depwise Seperable Convolution


#### Width Multiplier
- for a given layer and width multiplier a, the number of input cannels M becomes aM and the number of output channels N becomes aN - where a with typical settings of 1,0.75, 0.6 and 0.25
    
    
>> input channel 을 output으로 움직일떄 줄이겟다 


#### Resolution Multiplier
- the second hyper-parameter to reduce the computational cost of a neural network is a resulution multiplier p


- 0<p<1, which is typically set of implicitly so that input resolution of network is 224,192, 160, 128

>> 이미지의 가로, 세로 resolution 을 줄여서 집어넣겠다.


>> MobileNet도 핵심은 Depthwise-Separable Convolution 연산을 적절히 사용하는 것이며, 이는 직전에 소개드린 Xception 논문에서 제안한 아이디어입니다. 약간의 차이가 있다면, architecture 구조를 새롭게 제안을 하였고, Depthwise Convolution과 Point Convolution 사이에 BN과 ReLU를 넣은 점이 차이점입니다. 또한 Xception은 Inception보다 높은 정확도를 내기 위해 Depthwise-Separable Convolution을 적절히 사용하는 데 집중한 반면, MobileNet은 Depthwise-Separable Convolution을 통해 Mobile Device에서 돌아갈 수 있을 만큼 경량 architecture 구조를 제안하는데 집중을 했다는 점에서도 차이가 있습니다. 즉, 같은 연산을 사용하였지만 바라보고 있는 곳이 다른 셈이죠.

![MobileNet](https://hoya012.github.io/assets/img/image_classification_guidebook/35.PNG)


- deep 한것을 냅두고 narrow한게 만드는게 더 좋다.





