import os
import xml.etree.ElementTree as ET


def int2str(num):
    if num >= 0 and num <= 9:
        return '000' + str(num)
    elif num >= 10 and num <= 99:
        return '00' + str(num)
    elif num >= 100 and num < 999:
        return '0' + str(num)
    else:
        return str(num)

folder_list = ['/pascalVOC_hh/Annotations/', '/pascalVOC_ih/Annotations/','/pascalVOC_jw/Annotations/', '/pascalVOC_mh/Annotations/']
numbering = 0
origin_dict = {}
temp_dict = {}
for fl in folder_list:
    # targetDir = os.path.join('C:/Users/사용자/Desktop/pascal_voc', fl)
    targetDir = 'C:/Users/사용자/Desktop/pascal_voc' + fl
    # print(targetDir)
    temp_list = os.listdir(targetDir)
    for tl in temp_list:
        temp_dict[tl.split('.')[0]] = numbering
        numbering += 1
    origin_dict.update(temp_dict)
    temp_dict = {}
    # print(temp_dict)
print(origin_dict)
print('\n')

text_list = ['/pascalVOC_hh/ImageSets/Main/',
               '/pascalVOC_ih/ImageSets/Main/',
               '/pascalVOC_jw/ImageSets/Main/',
               '/pascalVOC_mh/ImageSets/Main/']

modified_list = []
for tl in text_list:
    text_path = 'C:/Users/사용자/Desktop/pascal_voc' + tl + 'buttering_train.txt'
    file = open(text_path, "r")
    line = file.readlines()
    for l in line:
        if l.split(' ')[1].strip() == '-1':
            continue
        else:
            text = l.split(' ')[0]
            num = int2str(origin_dict[text.split('.')[0]])
            modified = l.replace(l.split('.')[0], num).split(' ')[0]
            modified_list.append(modified)
    file.close()
print(modified_list)

tranvalcount = 0
with open('C:/Users/사용자/Desktop/buttering_train.txt', mode = 'w') as ff:
    for ml in modified_list:
        tranvalcount += 1
        ff.write(ml + '\n')

print('train, val 수 : ', tranvalcount)

# ================================================================================= #

modified_list = []
for tl in text_list:
    text_path = 'C:/Users/사용자/Desktop/pascal_voc' + tl + 'buttering_val.txt'
    file = open(text_path, "r")
    line = file.readlines()
    for l in line:
        if l.split(' ')[1].strip() == '-1':
            continue
        else:
            text = l.split(' ')[0]
            num = int2str(origin_dict[text.split('.')[0]])
            modified = l.replace(l.split('.')[0], num).split(' ')[0]
            modified_list.append(modified)
    file.close()
print(modified_list)

testcount = 0
with open('C:/Users/사용자/Desktop/buttering_val.txt', mode = 'w') as ff:
    for ml in modified_list:
        testcount += 1
        ff.write(ml + '\n')

print('test 수 : ',testcount)