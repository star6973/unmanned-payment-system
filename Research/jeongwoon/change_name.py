import os
import shutil

path = 'C:/Users/User/Pictures/라벨링된사진/'
resultPath = 'C:/Users/User/Pictures/바뀐사진/'

list = os.listdir(path)


def getfiles(dirpath):
    a = [s for s in os.listdir(dirpath)
         if os.path.isfile(os.path.join(dirpath, s))]
    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)))
    return a


count = 1
for filename in list:
    fromFilePathName = path + filename
    resultFilePathName = resultPath + str(count) + 'jw' + '.jpg'
    shutil.move(fromFilePathName, resultFilePathName)

    print(fromFilePathName)
    print(resultFilePathName)
    count += 1