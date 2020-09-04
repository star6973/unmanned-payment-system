# 20200902 데이터 crop 하기
- 우선은 사이다에 대해서만 진행
    - 이미지 사이즈 조정은 잠시 보류
1. 필요한 것들 import
<pre><code>
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
</code></pre>
1.csv 파일 불러오기
<pre><code>
label_data = pd.read_csv('/content/drive/My Drive/ldata/csv_hh/hh.csv')
</code></pre>
1. csv 파일을 각 항목별로 나누기
<pre><code>
cidar = label_data.loc[label_data['label']=='cidar',['image','xmin','ymin','xmax','ymax']]
coke = label_data.loc[label_data['label']=='coke',['image','xmin','ymin','xmax','ymax']]
welchs = label_data.loc[label_data['label']=='welchs',['image','xmin','ymin','xmax','ymax']]
vita500 = label_data.loc[label_data['label']=='vita500',['image','xmin','ymin','xmax','ymax']]
toreta = label_data.loc[label_data['label']=='toreta',['image','xmin','ymin','xmax','ymax']]
samdasu = label_data.loc[label_data['label']=='samdasu',['image','xmin','ymin','xmax','ymax']]

squid_peanut = label_data.loc[label_data['label']=='squid_peanut',['image','xmin','ymin','xmax','ymax']]
diget_ori = label_data.loc[label_data['label']=='diget_ori',['image','xmin','ymin','xmax','ymax']]
diget_choco = label_data.loc[label_data['label']=='diget_choco',['image','xmin','ymin','xmax','ymax']]
zec = label_data.loc[label_data['label']=='zec',['image','xmin','ymin','xmax','ymax']]
buttering = label_data.loc[label_data['label']=='buttering',['image','xmin','ymin','xmax','ymax']]
homerunball = label_data.loc[label_data['label']=='homerunball',['image','xmin','ymin','xmax','ymax']]
couque_white = label_data.loc[label_data['label']=='couque_white',['image','xmin','ymin','xmax','ymax']]
couque_coffee = label_data.loc[label_data['label']=='couque_coffee',['image','xmin','ymin','xmax','ymax']]
chocopie = label_data.loc[label_data['label']=='chocopie',['image','xmin','ymin','xmax','ymax']]
jjolbyung_noodle = label_data.loc[label_data['label']=='jjolbyung_noodle',['image','xmin','ymin','xmax','ymax']]
jjolbyung_ori = label_data.loc[label_data['label']=='jjolbyung_ori',['image','xmin','ymin','xmax','ymax']]

tuna = label_data.loc[label_data['label']=='tuna',['image','xmin','ymin','xmax','ymax']]

gumi_gumi = label_data.loc[label_data['label']=='gumi_gumi',['image','xmin','ymin','xmax','ymax']]
juicyfresh = label_data.loc[label_data['label']=='juicyfresh',['image','xmin','ymin','xmax','ymax']]
spearmint = label_data.loc[label_data['label']=='spearmint',['image','xmin','ymin','xmax','ymax']]
ID_gum = label_data.loc[label_data['label']=='ID_gum',['image','xmin','ymin','xmax','ymax']]
print(cidar)
</code></pre>
1. image를 원하는 영역만 자르는 함수 선언
<pre><code>
def trim_image(image, x_min, y_min, x_max, y_max, dir):
  trimmed_image = image[y_min:y_max, x_min:x_max]
  cv2.imwrite(dir, trimmed_image)
</code></pre>
1. 해당 경로에 폴더 생성하는 함수 선언 및 폴더 생성
<pre><code>
def make_folder(directory_path):
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
# make directory for cidar
save_cidar_path = os.path.join(os.getcwd(), 'drive/My Drive/crop_data/cidar')
make_folder(save_cidar_path)
</code></pre>
1. cidar 데이터 crop 하여 미리 생성된 폴더에 저장
<pre><code>
# save cidar data to cidar diretcory
for i in range(len(cidar)):
  img_source_path = os.path.join('/content/drive/My Drive/ldata/csv_hh/temp', cidar.iloc[i,0])
  img = cv2.imread(img_source_path)
  if i < 10 and i >= 0:
    img_name = '0000' + str(i) + '.jpg'
    save_img_name = os.path.join(os.getcwd(), 'drive/My Drive/crop_data/cidar', img_name)
  elif i < 100 and i >= 10:
    img_name = '000' + str(i) + '.jpg'
    save_img_name = os.path.join(os.getcwd(), 'drive/My Drive/crop_data/cidar', img_name)
  elif i < 1000 and i >= 100:
    img_name = '00' + str(i) + '.jpg'
    save_img_name = os.path.join(os.getcwd(), 'drive/My Drive/crop_data/cidar', img_name)
  elif i < 10000 and i >= 1000:
    img_name = '0' + str(i) + '.jpg'
    save_img_name = os.path.join(os.getcwd(), 'drive/My Drive/crop_data/cidar', img_name)
  elif i < 100000 and i >= 10000:
    img_name = str(i) + '.jpg'
    save_img_name = os.path.join(os.getcwd(), 'drive/My Drive/crop_data/cidar', img_name)

  img_source_path = os.path.join('/content/drive/My Drive/ldata/csv_hh/temp', cidar.iloc[i,0])
  trim_image(img, int(cidar.iloc[i,1]),int(cidar.iloc[i,2]),int(cidar.iloc[i,3]),int(cidar.iloc[i,4]),save_img_name)
</code></pre>