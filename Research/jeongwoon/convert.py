#1.해상도에 따라 각각의 빈 txt 파일 만들기
import cv2
import shutil

file = open("C:/Users/User/Desktop/train_final.txt",'r')
path_base720 = "C:/Users/User/Desktop/Yolov5/dataset2/labels_720/"
path_base960 = "C:/Users/User/Desktop/Yolov5/dataset2/labels_960/"
path_image = 'C:/Users/User/Desktop/Yolov5/dataset/train/images/'


for a in range(14179): # train 각 클래스 데이터 개수
    line_list = file.readline()
    file_num = (line_list.split(',')[0]).strip(".jpg")
    imageFile = path_image + str(file_num) + '.jpg'
    src_gray = cv2.imread(imageFile,0)
    height, _ = src_gray.shape
    if height == 720:
        F=open(path_base720+str(file_num)+".txt",'w')
    else:
        F=open(path_base960+str(file_num)+".txt",'w')

    F.close()

'''

# 위에 것을 주석처리 하고
# 2.각 데이터셋 내용 빈 txt에 써넣기
import os

tag_class = ['ID_gum', 'buttering', 'couque_coffee', 'chocopie', 'cidar', 'couque_white', 'coke', 'diget_ori', 'diget_choco', 'gumi_gumi', 'homerunball', 'jjolbyung_noodle', 'juicyfresh', 'jjolbyung_ori', 'spearmint', 'squid_peanut', 'samdasu', 'tuna', 'toreta', 'vita500', 'welchs', 'zec']
# print(len(tag_class))
for i in range(3577):
    line_list =file.readline()
    A=line_list.split(',')
    for j in range(22):
        if tag_class[j] == A[5].strip("\n"):
            A = A[:-1]
            A.insert(1,j)
    A[0] = A[0].strip(".jpg")

    # height 720과 960 따로따로 !
    
    # height 720
    # A[2:] = list(map(float,A[2:]))
    # A[2:] = [(A[2]+A[4])/2560,(A[3]+A[5])/1440,(A[4]-A[2])/1280,(A[5]-A[3])/720]
    # height 960
    A[2:] = list(map(float, A[2:]))
    A[2:] = [(A[2] + A[4]) / 2560, (A[3] + A[5]) / 1920, (A[4] - A[2]) / 1280, (A[5] - A[3]) / 960]
    # print(A)

    P = os.listdir(path_base960)
    for p in P:
        if p.strip(".txt") == A[0]:
            files = open(path_base960 + p, 'a')
            files.write(' '.join(map(str, A[1:]))+"\n")

    # 이렇게 들어간다.
    print(' '.join(map(str, A[1:])))

'''