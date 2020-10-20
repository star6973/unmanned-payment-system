
import sys
import os
import json
import xml.etree.ElementTree as ET


START_BOUNDING_BOX_ID = 1

# If necessary, pre-define category and its id
PRE_DEFINE_CATEGORIES = {'background': 0, 'ID_gum': 1, 'buttering': 2, 'couque_coffee': 3, 'chocopie': 4, 'cidar': 5, 'couque_white': 6, 'coke': 7, 'diget_ori': 8, 'diget_choco': 9, 'gumi_gumi': 10, 'homerunball': 11, 'jjolbyung_noodle': 12, 'juicyfresh': 13, 'jjolbyung_ori': 14, 'spearmint': 15, 'squid_peanut': 16, 'samdasu': 17, 'tuna': 18, 'toreta': 19, 'vita500': 20, 'welchs': 21, 'zec': 22}



def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError(
            'The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def convert(xml_list, xml_dir, json_file):
    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    image_id = 0
    for line in xml_list:
        line = line.strip()
        print("Processing %s" % (line))
        xml_f = os.path.join(xml_dir, line+'.xml')
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'path')
        # if len(path) == 1:
        #     filename = os.path.basename(path[0].text)
        # elif len(path) == 0:
        #     filename = get_and_check(root, 'filename', 1).text
        # else:
        #     raise NotImplementedError(
        #         '%d paths found in %s' % (len(path), line))
        filename = get_and_check(root, 'filename', 1).text
        if len(filename.split('.')) == 1:
            filename = filename+'.jpg'

        image_id = image_id+1
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width,
                 'id': image_id}
        json_dict['images'].append(image)
        # Cruuently we do not support segmentation
        # segmented = get_and_check(root, 'segmented', 1).text
        # assert segmented == '0'
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:
                new_id = len(categories)+1
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = float(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = float(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = float(get_and_check(bndbox, 'xmax', 1).text)
            ymax = float(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict, indent=4)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == '__main__':
    xml_dir = '/data/inhwa/coco2017/val_xml/'
    img_dir = '/data/inhwa/coco2017/val2017'
    img_list = os.listdir(img_dir)
    img_list = [img.split('.')[0] for img in img_list]
    print(len(img_list))
    xml_list = os.listdir(xml_dir)
    xml_list = [xml.split('.')[0] for xml in xml_list]
    print(len(xml_list))
    xml_list = list(set(img_list).intersection(set(xml_list)))
    print(len(xml_list))
    json_file = '/data/inhwa/coco2017/annotations/instances_val2017.json'

    convert(xml_list, xml_dir, json_file)
