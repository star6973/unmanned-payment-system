## SqueezeNet , 2016

- Squeeze라는 단어는 쥐어짜내는 것을 뜻하며 network를 쥐어 짜내는 것을 의미하며, 제목에서 알 수 있듯이 AlexNet의 parameter를 50배 이상 줄여서 0.5MB 이하의 model size를 가질 수 있는 architecture 구조를 제안하고 있습니다.

![squeeze](https://hoya012.github.io/assets/img/image_classification_guidebook/27.PNG)


#### fire module

- Fire module에는 총 세 가지 하이퍼파라미터(s1x1, e1x1, e3x3)가 있습니다. 먼저 s1x1는 squeeze layer에서 1x1 필터의 총 갯수입니다. e1x1는 expand layer에서의 1x1 필터 총 갯수이며 e3x3는 expand layer에서의 3x3 필터의 총 갯수입니다. Fire module을 만들 때는 s1x1의 값을 e1x1+e3x3보다 더 작게 설정해주었습니다.  3x3 필터로 들어가는 input channel의 수를 제한할 수 있게합니다.

#### SqueezeNet Architecture

- 스퀴즈넷의 전체적인 구조는 다음과 같습니다. 모든 스퀴즈넷은 전부 1개의 Convolution filter를 거치고 나서 max pooling이 이어집니다. 그 이후 총 8개의 fire module로 이루어져 있고 마지막에 convolution filter를 거치고 GAP(Global Average Pooling)로 마무리가 됩니다. Pooling layer를 conv1, fire4, fire8, conv10 이후에 배치하며 3번 전략(activation map을 크게)을 취했다고 합니다.


#### Evaluation of SqueezeNet

![squeezeNet1](https://i.imgur.com/OcwRopR.png)

- 일단 논문 제목에도 나와있지만, 알렉스넷과 비교했을 때 파라미터 수를 확 줄였으면서도 성능이 비슷하다고 했으므로 실험 결과도 알렉스넷과 비교를 합니다. 실험은 이미지넷 데이터셋을 사용했습니다.

- 위 표에서 왼쪽 CNN architecture와 compression approach를 보시면 됩니다. 단순 SqueezeNet만 사용했을 때는 50배 가까이 모델 사이즈가 줄어들었습니다. 게다가 기존 AlexNet의 top-1 & top-5 accuray에 근접하거나 뛰어넘는 모습을 보여줍니다.

- 여기에 더해 uncompressed 된 32bit의 데이터 타입을 사용한 생짜 SqueezeNet과 deep compression을 적용한 8bit, 6bit짜리 데이터 타입을 사용한 결과도 매우 놀랍습니다. 최고의 결과물은 모델 사이즈가 510배까지 줄어들었으며 성능도 큰 차이가 나지 않습니다. 이는 SqueezeNet 또한 모델 압축에 굉장히 유연하다는 뜻입니다.

- 또한 기존의 파라미터가 많은 VGG나 AlexNet같은 모델들 뿐만 아니라 이미 컴팩트한 모델도 압축할 수 있다는 것을 보여주었습니다.

>> Pruning, Compression 등 모델 경량화 기법들을 많이 사용하였지만 architecture 관점에서도 연산량을 줄이기 위한 시도를 논문에서 보여주고 있습니다. 다만, fire module의 아이디어는 이미 지난 번 소개 드린 Inception v2의 Conv Filter Factorization과 비슷한 방법이고, Inception v2에 비해 정확도가 많이 낮아서 좋은 평가를 받지 못한 것으로 생각됩니다.