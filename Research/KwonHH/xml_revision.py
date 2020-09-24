import os
import xml.etree.ElementTree as ET
from PIL import Image

def int2str(num):
    if num >= 0 and num <= 9:
        return '000' + str(num)
    elif num >= 10 and num <= 99:
        return '00' + str(num)
    elif num >= 100 and num < 999:
        return '0' + str(num)
    else:
        return str(num)

numbering = 0

targetDir = '../pascal_voc/pascalVOC_hh/Annotations/'
save_Dir = 'C:/Users/사용자/Desktop/Annotations/'

img_targetDir = '../pascal_voc/pascalVOC_hh/JPEGImages/'
img_save_Dir = 'C:/Users/사용자/Desktop/JPEGImages/'

file_listhh = os.listdir(targetDir)
print('hh\n', file_listhh)
# xml_list1 = []
# for file in file_list1:
#     if '.xml' in file:
#         xml_list1.append(file)

file_dict1 = {}


for file in file_listhh:
    if '.xml' in file:
        file_dict1[file] = numbering
        numbering += 1
print(file_dict1)


for xml_file in file_listhh:
    num = file_dict1[xml_file]
    str_len_xml = int2str(num)
    target_path = os.path.join(targetDir, xml_file)

    targetXML = open(target_path, 'rt', encoding='UTF-8')

    tree = ET.parse(targetXML)
    root = tree.getroot()

    target_filename = root.find("filename")
    original = target_filename.text  # 원본 String
    modified = original.replace(original.split('.')[0], str_len_xml)
    target_filename.text = modified

    target_path = root.find("path")
    # original_path = target_path.text
    # modified_original_path = original_path.replace((original_path.split('/')[2]).split('.')[0], str_len_xml)
    modified_original_path = 'PascalVoc/Annotations/' + str_len_xml + '.jpg'
    target_path.text = modified_original_path


    save_path = os.path.join(save_Dir, str_len_xml) + '.xml'
    tree.write(save_path)

    img_path = img_targetDir + xml_file.split('.')[0] + '.jpg'
    # print('img_path', img_path)

    im = Image.open(img_path)
    img_save_path = img_save_Dir + str_len_xml + '.jpg'

    im.save(img_save_path)


#=================================================================================#


targetDir = '../pascal_voc/pascalVOC_ih/Annotations/'
img_targetDir = '../pascal_voc/pascalVOC_ih/JPEGImages/'

file_listih = os.listdir(targetDir)
print('ih\n', file_listih)
# xml_list2 = []
# for file in file_list2:
#     if '.xml' in file:
#         xml_list2.append(file)

file_dict2 = {}

for file in file_listih:
    if '.xml' in file:
        file_dict2[file] = numbering
        numbering += 1
print(file_dict2)


for xml_file in file_listih:
    num = file_dict2[xml_file]
    str_len_xml = int2str(num)
    target_path = os.path.join(targetDir, xml_file)

    targetXML = open(target_path, 'rt', encoding='UTF-8')

    tree = ET.parse(targetXML)
    root = tree.getroot()

    target_filename = root.find("filename")
    original = target_filename.text  # 원본 String
    modified = original.replace(original.split('.')[0], str_len_xml)
    target_filename.text = modified

    target_path = root.find("path")
    # original_path = target_path.text
    # modified_original_path = original_path.replace((original_path.split('/')[2]).split('.')[0], str_len_xml)
    modified_original_path = 'PascalVoc/Annotations/' + str_len_xml + '.jpg'
    target_path.text = modified_original_path


    save_path = os.path.join(save_Dir, str_len_xml) + '.xml'
    tree.write(save_path)

    img_path = img_targetDir + xml_file.split('.')[0] + '.jpg'
    # print('img_path', img_path)

    im = Image.open(img_path)
    img_save_path = img_save_Dir + str_len_xml + '.jpg'

    im.save(img_save_path)







#=================================================================================#


targetDir = '../pascal_voc/pascalVOC_ih/Annotations/'
img_targetDir = '../pascal_voc/pascalVOC_ih/JPEGImages/'

file_listjw = os.listdir(targetDir)
print('jw\n', file_listjw)
# xml_list3 = []
# for file in file_list3:
#     if '.xml' in file:
#         xml_list3.append(file)

file_dict3 = {}

for file in file_listjw:
    if '.xml' in file:
        file_dict3[file] = numbering
        numbering += 1
print(file_dict3)


for xml_file in file_listjw:
    num = file_dict3[xml_file]
    str_len_xml = int2str(num)
    target_path = os.path.join(targetDir, xml_file)

    targetXML = open(target_path, 'rt', encoding='UTF-8')

    tree = ET.parse(targetXML)
    root = tree.getroot()

    target_filename = root.find("filename")
    original = target_filename.text  # 원본 String
    modified = original.replace(original.split('.')[0], str_len_xml)
    target_filename.text = modified

    target_path = root.find("path")
    # original_path = target_path.text
    # modified_original_path = original_path.replace((original_path.split('/')[2]).split('.')[0], str_len_xml)
    modified_original_path = 'PascalVoc/Annotations/' + str_len_xml + '.jpg'
    target_path.text = modified_original_path


    save_path = os.path.join(save_Dir, str_len_xml) + '.xml'
    tree.write(save_path)

    img_path = img_targetDir + xml_file.split('.')[0] + '.jpg'
    # print('img_path', img_path)

    im = Image.open(img_path)
    img_save_path = img_save_Dir + str_len_xml + '.jpg'

    im.save(img_save_path)

    # =================================================================================#

targetDir = '../pascal_voc/pascalVOC_ih/Annotations/'
img_targetDir = '../pascal_voc/pascalVOC_ih/JPEGImages/'

file_listmh = os.listdir(targetDir)
print('mh\n',file_listmh)
# xml_list4 = []
# for file in file_list4:
#     if '.xml' in file:
#         xml_list4.append(file)

file_dict4 = {}

for file in file_listmh:
    if '.xml' in file:
        file_dict4[file] = numbering
        numbering += 1
print(file_dict4)

for xml_file in file_listmh:
    num = file_dict4[xml_file]
    str_len_xml = int2str(num)
    target_path = os.path.join(targetDir, xml_file)

    targetXML = open(target_path, 'rt', encoding='UTF-8')

    tree = ET.parse(targetXML)
    root = tree.getroot()

    target_filename = root.find("filename")
    original = target_filename.text  # 원본 String
    modified = original.replace(original.split('.')[0], str_len_xml)
    target_filename.text = modified

    target_path = root.find("path")
    # original_path = target_path.text
    # modified_original_path = original_path.replace((original_path.split('/')[2]).split('.')[0], str_len_xml)
    modified_original_path = 'PascalVoc/Annotations/' + str_len_xml + '.jpg'
    target_path.text = modified_original_path

    save_path = os.path.join(save_Dir, str_len_xml) + '.xml'
    tree.write(save_path)

    img_path = img_targetDir + xml_file.split('.')[0] + '.jpg'
    # print('img_path', img_path)

    im = Image.open(img_path)
    img_save_path = img_save_Dir + str_len_xml + '.jpg'

    im.save(img_save_path)










































