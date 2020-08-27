#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import os
import cv2


# In[32]:


train_csv = pd.read_csv("train.csv")
test_csv = pd.read_csv("test.csv")


# In[33]:


train_csv.head()


# In[34]:


test_csv.head()


# In[35]:


def make_folder(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)
        
path_train = os.path.join(os.getcwd(), 'EMNIST/Train')
path_test = os.path.join(os.getcwd(), 'EMNIST/Test')


# In[36]:


make_folder(path_train)
make_folder(path_test)


# In[37]:


for i in range(len(train_csv)):
    index = train_csv.loc[i, 'id']
    image = train_csv.loc[i, '0':].values.reshape(28, 28).astype(float)

    train_image = os.path.join(path_train, '{0:05d}'.format(index) + '.jpg')
    cv2.imwrite(train_image, image)


# In[38]:


for i in range(len(test_csv)):
    index = test_csv.loc[i, 'id']
    image = test_csv.loc[i, '0':].values.reshape(28, 28).astype(int)
    
    test_image = os.path.join(path_test, '{0:05d}'.format(index) + '.jpg')
    cv2.imwrite(test_image, image)


# In[39]:


data_transforms = transforms.Compose([transforms.ToTensor(),])


# In[40]:


# img_size = 64
# data_transforms = transforms.Compose{
#     transforms.Compose([
#         transforms.Resize([img_size, img_size]),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(20),
#         transforms.Grayscale(num_output_channels=1),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#     ])
# }


# In[41]:


data_dir = os.path.join(os.getcwd(), 'EMNIST')
emnist_train = dset.ImageFolder(os.path.join(data_dir), data_transforms)
emnist_test = dset.ImageFolder(os.path.join(data_dir), data_transforms)

train_data = torch.utils.data.DataLoader(emnist_train, batch_size=16, shuffle=True, drop_last=True)
test_data = torch.utils.data.DataLoader(emnist_test, batch_size=16, shuffle=False, drop_last=True)


# In[42]:


# def custom_imshow(img):
#     img = img.numpy()
#     plt.imshow(np.transpose(img, (1, 2, 0)))
#     plt.show()

# def process():
#     cnt = 0
#     for batch_idx, (inputs, targets) in enumerate(train_data):
#         if cnt == 10:
#             break
#         custom_imshow(inputs[0])
#         cnt += 1

# process()


# In[43]:


class customCNN(nn.Module):
    def __init__(self):
        super(customCNN, self).__init__()
        
        self.layer1 = self.conv_module(3, 16)
        self.layer2 = self.conv_module(16, 24)
        self.layer3 = self.conv_module(24, 32)
        self.layer4 = self.conv_module(32, 64)
        self.layer5 = self.conv_module(64, 128)
        self.gap = self.global_avg_pool(128, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(-1, 10)
        
        return out
    
    def conv_module(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
    
    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )


# In[45]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = customCNN().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)


# In[46]:


print(model)


# In[47]:


loss_arr = []
for i in range(10):
    for j, [image, label] in enumerate(train_data):
        x = image.to(device)
        y_ = label.to(device)
        
        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output, y_)
        loss.backward()
        optimizer.step()
        
        if j % 1000 == 0:
            print(loss)
            loss_arr.append(loss.cpu().detach().numpy())


# In[50]:


correct = 0
total = 0

with torch.no_grad():
    for image, label in test_data:
        x = image.to(device)
        y_ = label.to(device)
        print(y_)
        
        output = model.forward(x)
        _, output_index = torch.max(output, 1)
        print(output_index)
        total += label.size(0)
        correct += (output_index == y_).sum().float()
        
    print("Accuracy of Test Data: {}".format(100 * correct / total))


# In[48]:


submission = pd.read_csv('submission.csv')
submission.digit = torch.cat(output).detach().cpu().numpy()
submission.to_csv('./result/CustomCNN_result.csv', index=False)


# In[ ]:




