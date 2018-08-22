
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

from torchvision import transforms
from model.lstm_pm import LSTM_PM
from data.handpose_data2 import UCIHandPoseDataset

import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader
import os

import scipy.misc
import json
import numpy as np
import os
import utils


# In[6]:


nb_temporal = 4
batch_size=1
nb_joint = 21
if_save_img = False
if_sum = False
if_pck = False
if_max = True
#run = [10,15,20,25,27,30,33,35,37,40,45,50,55,60]
run = [30]
pck_sigma = 0.04
path_root = '/home/danningx/code/8-15/'
ckpt_path = '/home/danningx/code/8-15/checkpoint'


# In[4]:


avg_pck_savepath = os.path.join(path_root, "pck.json")
img_save_path = utils.mkdir(os.path.join(path_root, "runtime_heatmaps/test/"))

if if_sum:
    save_sum_path = os.path.join(path_root, "sum_history.json")
if if_max:
    save_max_path = os.path.join(path_root, "max_history.json")


transform = transforms.Compose([transforms.ToTensor()])
data_dir = '/mnt/UCIHand/test/test_data'
label_dir = '/mnt/UCIHand/test/test_label'
dataset = UCIHandPoseDataset(data_dir=data_dir, label_dir=label_dir, temporal=nb_temporal,train=False)
test_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False)

net = LSTM_PM(T=nb_temporal).cuda()
net = torch.nn.DataParallel(net)

ckpt_list = os.listdir(ckpt_path)


# In[7]:



pck_history = {}
if if_sum:
    hm_sum = [[] for x in range(nb_joint)]
    hm_sum_avg = {}

if if_max:
    hm_max = [[] for x in range(nb_joint)]
    hm_max_avg = {}
    
for ckpt_name in ckpt_list:
    if int(ckpt_name.split('_')[-1]) in run:
        print 'Testing ckeckpoint '+ ckpt_name.split('_')[-1] + '*****************************************'
        pck_all = []
        img_name = []
        net.load_state_dict(torch.load(os.path.join(ckpt_path, ckpt_name)))
        net.eval()
        for step, (images, label_map, center_map, imgs) in enumerate(test_dataset):
            images = Variable(images.cuda())
            label_map = Variable(label_map.cuda())
            center_map = Variable(center_map.cuda()) 
            
            predict_heatmaps = net(images, center_map) 

            pck = utils.lstm_pm_evaluation(label_map, predict_heatmaps, sigma=pck_sigma, temporal=4)
            pck_all.append(pck)
            img_name.append(imgs)
            if if_sum:
                for pre_hm in predict_heatmaps:
                    for j, hm in enumerate(pre_hm[0]):
                        hm_sum[j].append(float(sum(sum(hm))))
            
            if if_max:
                for pre_hm in predict_heatmaps:
                    for j, hm in enumerate(pre_hm[0]):
                        hm_max[j].append(float(torch.max(hm)))
                        
            if step%100==0:
                if if_save_img:
                    utils.save_image(img_save_path+'e'+ckpt_name.split('_')[-1]+'_stp'+str(step)+'_b', nb_temporal, predict_heatmaps, label_map)
                print "pck: " + str(pck)
        avg_pck = sum(pck_all)/float(len(pck_all))
        print "checkpoint "+ckpt_name.split('_')[-1]+" : "+str(avg_pck)
        pck_history[int(ckpt_name.split('_')[-1])] = {'avg':avg_pck,'pck_all':pck_all, 'img_name':img_name}
        json.dump(pck_history, open(avg_pck_savepath, 'wb')) 
if if_sum:
    for j,s in enumerate(hm_sum):
        hm_sum_avg[j] = sum(s)/len(s)
    avg_sum = sum(hm_sum_avg.values())/len(hm_sum_avg.keys())
    hm_sum_avg['avg_overall'] = avg_sum
    hm_sum_avg['history'] = hm_sum
    json.dump(test_history, open(save_sum_path, 'wb')) 
      
if if_max:
    for j,s in enumerate(hm_max):
        hm_max_avg[j] = sum(s)/len(s)
    avg_max = sum(hm_max_avg.values())/len(hm_max_avg.keys())
    hm_max_avg['avg_overall'] = avg_max
    hm_max_avg['history'] = hm_max
    json.dump(hm_max_avg, open(save_max_path, 'wb')) 

