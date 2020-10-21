import xml.etree.ElementTree as ET
import yaml
# http://egloos.zum.com/sweeper/v/3045370

dir = r'C:\Users\zkzh6\OneDrive\Desktop\0000.xml'
tree = ET.parse(dir)
root = tree.getroot()


folder = root.find('folder')
filename = root.find('filename')
path = root.find('path')

ann_dict = {}

source = root.findall("source")
ann_dict['source'] = source[0].find('database').text

size = root.findall('size')
ann_dict['size'] = size[0].find('width').text
ann_dict['size'] = size[0].find('height').text
ann_dict['size'] = size[0].find('depth').text

print(ann_dict)

tree.write(r'C:\Users\zkzh6\OneDrive\Desktop\0000.yml')

with open(r'C:\Users\zkzh6\OneDrive\Desktop\0000.yml') as ff:
    veg = yaml.load(ff, Loader=yaml.FullLoader)
    print(veg)
