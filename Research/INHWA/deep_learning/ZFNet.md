## ZFNet , 2013

ZFNet은 기법은 비슷하나 Visualization 기법으로 많은 관심을 받았다. 

### Deconvolution 

- 중간 layer에서 feature의 activity가 어떻개 되는지 알수 있어야 한다.
그런데 중간 레이어의 동작은 보기 어려우니 이 activity를 다시 입력이미지 공간에 mapping을 시키는 기법이 바로 
'Visualizing'기법의 핵심이 된다.

- CNN구조는 여러개의 컨폴루션 레이어를 기반으로 하며 그 feature map을 받아 convolution을 수행하여 Relu로 활성화 시킨이후에 subsampling을 취한다.

특정 feature의 activity가 입력 이미지에서 어떻게 mapping이 되는지를 이해하려면 역으로 수행하면 된다

하지만 여기서 문제가 되는 부분은 max-pooling에 대한 역[reverse]을 구하는 것이다.

Max-pooing단계에서 주변에서 가장 강한 신호만 다음 단계로 전달하기에 역방향으로 진행 할 떄는 가장 강한 신호가 어디 위치인지를 파악할수 있는 방법이 없다.

그래서 ZFNet팀에서는 'switch'라는 개념을 생각했다. 

###Switch

- switch는 가장 강한 자극의 위치 정보를 갖고잇는 일종의 꼬리표[flag]이다.
가장 강한 신호의 위치 정보를갖고있기 때문에 역으로 un-pooling을 수행할 때는 switch정보를 활용하여 가장 강한 자극의 위치로 정확하게 찾아갈수있다.
하지만 결국 약한 자극들은 사라져있기에 어느정도의 데이터는 손실되어있다


![ZFNet](https://lh3.googleusercontent.com/proxy/AkSUmgySFDD-ffTYFG-f1X1TYXSPpxYSevvOf6Sxd0-tyvvsL4bN4ZFahfmnuenC3kUUDid7GR0R259z3-0JZtYtz0w37J5ItPTu7lRyZzYdL1LfUzFfFz0TA0-WgSIOEpdxlxcVUoNJK8RKKK-l1JeavHF3DKFlAYmEznI3h10)



https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220673615573&proxyReferer=https:%2F%2Fwww.google.com%2F