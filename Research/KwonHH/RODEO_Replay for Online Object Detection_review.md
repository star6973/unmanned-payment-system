# RODEO : Replay for Online Object Detection
## 논문 리뷰
- Abstract
    - 본 논문은 연산시간과 저장공간의 제약이 따르는 Online Streaming 상황에서 Object Detection 방법을 개척
        - Object Detection 은 모든 바운딩 박스에 대해서 올바르게 이미지 라벨을 보여줘야 함
        - 본 논문에서는 이전의 연구와는 다르게  시간이 지남에 따라 소개된 새로운 class를 사용해 온라인 방식으로 연구를 할 수 있었음<br><br>
1. Introduction       
    - Object Detection 개념 : 대상에 바운딩 박스를 적용해 label class를 예측하고, 위치까지 찾아내는 것
        - 기존 Detection의 경우 <u>오프라인에서 적용</u> 되었기 때문에 새로운 object class의 업데이트가 불가능
        - 반면, 인간 및 포유류는 한 번에 정지되지 않은(유동적인) 샘플로부터 더 많은 부분을 학습하고 이해할 수 있음<br> ==> <u>__Streaming Learning__</u> <u>__(= Online Learning)__</u>
        - 그러나, 통상적인 방식의 __Streaming Learning__ 은 __Catastrophic Forgetting__ 문제가 발생<br>***<br> Catastrophic Forgetting 란? 인공신경망이 단일 작업에 대해서는 뛰어난 성능을 보이지만, 다른 종류의 작업을 학습하면 이전 학습 내용을 잊어버리는 것<br><br>
        - Streaming Learning 은 새로운 class를 추가하거나, 시기별로 detector를 적용하는 경우, 혹은 시간 경과의 따라 객체를 포함시키는 어플리케이션에 적용될 수 있는 방식이다
            - 기존의 Incremental Object Detection 방식은 상당한 한계를 가지고 있으며, Streaming Learning 이 불가능
            - Streaming Learning을 하기 위해서는 현재의 장면을 바로 업데이트하지 않고, <u>큰 배치사이즈로 이미지를 업데이트 하는 방식을 사용</u><br><br>
        - 기존의 Incremental 이미지 인식 방식과 비교했을 때, Replay 메커니즘이 Catastrophic Forgetting을 완화하는데 효과적임을 보여준다
            - Replay 아이디어는 인간이 어떻게 기억들간에 관계를 짓고, 그것이 강화되는지에서 영감을 얻었다(대뇌에서 일어나는 현상에서 영감을 얻음)
            - 추가적으로 Hioopcampal Indexing 이론에 따르면 인간은 압축된 기억을 기억해낼 때, 인덱싱을 메커니즘을 사용하는 것으로 알려졌다
            - 즉, RODEO 에서는 고정된 용량의 buffer로부터 압축된 데이터를 replay하고 이를 점진적으로 Object Detection 수행<br>지금까지의 Incremental 방식은 생물학적으로 합리적이지 않다고 언급함
            - 이렇게 했을 경우 <u>효율적인 연산</u> 및 <u>다른 어플리케이션으로 확장이 용이</u>한 장점이 있음을 확인<br><br>
        - 본 논문이 기여한 내용
            1. Streamin Learning 방식으로 Object Detection 수행 및 강력한 Baseline을 세움
            1. RODEO 모델을 제안 : Replay 방식을 통해 Catastrophic Forgetting 완화, Incremental batch object detection 알고리즘보다 더 우수한 성취 결과<br><br><br>
1. Experimental Result
             
        