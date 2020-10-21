# FaceNet : A Unified Embedding for Face Recognition and Clustering
## 논문 리뷰
3. Method
- 본 장에서는 FaceNet에서 사용된 Deep conv network 중 "visualizing and understanding convolution networks" 와 "Inception" 핵심 구조를 설명함
- 본 논문의 접근 방식 중 가장 중요한 부분은 전체 시스템에서 end-to-end learning 을 구현하는 것
- 또한, 이미지 x에서 d차원의 공간으로 모든 face에 대해서 embedding하여, 이미지 조건에 관계없이 동일인이면 작은 squared distance를 갖고, 타인이면 큰 squared distance를 갖도록 하는 것을 목표로 함
- 본 논문에서 모든 loss함수를 고려하진 않았지만, face verification에서 원하는 결과를 위해서 Triplet loss가 가장 적합할 것으로 판단되어 triplet loss 적용
- Triplet loss의 경우 참/거짓 각 쌍에 대한 margin이 크도록 강요하기 때문에 face를 구별하는데 더욱 좋은 효과를 보임<br><br><br>
    1. Triplet Loss
        - embedding 된 loss 함수는 ![]() 로 표현되며, d차원의 유클리드 공간으로 emdbed됨
        - 추가적으로 이러한 embedding을 d차원 공간으로 제한
        - 이때 loss는 "nearest-neighbor classication"에서 영감을 얻음
            - ![]() : 특정인에 대한 anchor
            - ![]() : 동일인에 대한 다른 이미지
            - ![]() : 다른 사람에 대한 이미지
        - 이를 그림으로 나타내면 다음과 같다<br> ![]() <br>
        - ![]() 식에서 좌변의 alpha는 margin을 의미하고, 위 그림에 따라 학습이 잘 이뤄질 경우 이러한 부등식이 성립해야 한다
        - 따라서 Triplet loss 는 다음 식을 표현된다<br> ![]() <br>
        - 모든 가능한 Triplet을 생성하는 것은 