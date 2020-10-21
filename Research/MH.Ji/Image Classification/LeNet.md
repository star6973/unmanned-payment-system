## LeNet
- MLP가 가지는 한계점인 input의 픽셀수가 많아지면 parameter가 기하급수적으로 증가하는 문제를 해결할 수 있는 CNN 구조 제시하였다.

- 특징
    + input을 2차원으로 확장하였고, parameter sharing을 통해 input의 픽셀 수가 증가해도 parameter 수가 변하지 않는다는 특징을 가지고 있다.

- 구조
    <center><img src="/reference_image/MH.Ji/Deep Learning Image Classification/99.PNG" width="70%"></center><br>

    + C1 layer에서는 32x32 픽셀의 image를 input으로 사용하고, 6개의 5x5 필터와 convolution 연산을 해준다. 그 결과, 6장의 28x28 feature map을 산출한다.

    + S2 layer에서는 C1 layer에서 출력된 feature map에 대해 2x2 필터를 stride=2로 설정하여 sub sampling을 진행한다. 그 결과, 14x14 feature map으로 축소된다. sub sampling 방법은 average pooling 기법을 사용하였는데, original 논문인 [Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)에 따르면, 평균을 낸 후에 한 개의 trainable weight을 곱해주고, 또 한 개의 trainable bias를 더해준다고 한다. activation function은 sigmoid를 사용한다.

    + C3 layer에서는 6장의 14x14 feature map에 convolution 연산을 수행해서 16장의 10x10 feature map을 산출한다.

    + S4 Layer에서는 16장의 10x10 feature map에 sub sampling을 진행해 16장의 5x5 feature map으로 축소시킨다.

    + C5 layer에서는 16장의 5x5 feature map에 120개의 5x5x16 필터를 convolution 연산을 수행해서 120장의 1x1 feature map을 산출한다.

    + F6 layer는 84개의 유닛을 가진 Feed Forward Neural Network로, C5 layer의 결과를 연결시켜준다.

    + Output layer는 10개의 Euclidean radial Bias Function(RBF) 유닛으로 구성되어 F6 layer의 결과를 받아서, 최종적으로 이미지가 속한 클래스를 분류한다.

- 참고자료
> [Deep Learning Image Classification Guidebook [1] LeNet, AlexNet, ZFNet, VGG, GoogLeNet, ResNet](https://hoya012.github.io/blog/deeplearning-classification-guidebook-1/)

> [LeNet-5의 구조](https://bskyvision.com/418)