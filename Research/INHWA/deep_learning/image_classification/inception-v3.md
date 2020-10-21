## Inception-v3, 2016
- Inception-v3는 Inception-v2의 architecture는 그대로 가져가고, 여러 학습 방법을 적용한 버전입니다. 한 논문에서 Inception-v2와 Inception-v3를 동시에 설명하고 있습니다. Inception-v2와 중복되는 내용이 많아서 간략하게 달라진 점만 정리하고 넘어가겠습니다.

#### Model Regularization via Label Smoothing
- one-hot vector label 대신 smoothed label을 생성하는 방식이며 자세한 설명은 제가 작성했던 글 의 3-B를 참고하시기 바랍니다.
#### Training Methodology
- Momentum optimizer –> RMSProp optimizer / gradient clipping with threshold 2.0 / evaluation using a running average of the parameters computed over time
#### BN-auxiliary
- Auxiliary classifier의 FC layer에 BN을 추가