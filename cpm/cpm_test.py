
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import cpm_model 
import numpy as np
from torch.autograd import Variable
from handpose_data_cpm import UCIHandPoseDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import json
#%matplotlib inline
import matplotlib.pyplot as plt
import utils
from collections import OrderedDict


# In[2]:


#**************hyper parameters needed to fill manully******************************
#nb_epochs=100
batch_size=1
nb_joint = 21
nb_stage = 6
background = False
if_sum = False
if_max = False
if_save = True #if save image
out_c = nb_joint+1 if background else nb_joint
heat_weight = 45 * 45 * out_c / 1.0
path_root = '/home/danningx/danningx/cpm_xdn/8-18/test_all/'
ckpt_path = '/home/danningx/danningx/cpm_xdn/8-18/checkpoint/cpm_30'
#************************************************************************************


# In[3]:


save_json_path = os.path.join(path_root, "test_history.json")
save_test_heatmap_path = utils.mkdir(os.path.join(path_root, "heatmaps/"))
if if_sum:
    save_sum_path = os.path.join(path_root, "sum_history.json")
if if_max:
    save_max_path = os.path.join(path_root, "max_history.json")

transform = transforms.Compose([transforms.ToTensor()])
#test data loader
data_dir = '/mnt/UCIHand/test/test_data'
label_dir = '/mnt/UCIHand/test/test_label'
dataset = UCIHandPoseDataset(data_dir=data_dir, label_dir=label_dir)
test_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# In[4]:



net = cpm_model.CPM(out_c=nb_joint, background=background).cuda()
state_dict = torch.load(ckpt_path)

# create new OrderedDict that does not contain `module.`
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    namekey = k[7:]  # remove `module.`
    new_state_dict[namekey] = v
# load params
net.load_state_dict(new_state_dict)
net = torch.nn.DataParallel(net)


# In[5]:


test_history = {}
net.eval()
runtime_pck= []
img_name = []
if if_sum:
    hm_sum = [[] for x in range(nb_joint)]
    hm_sum_avg = {}
if if_max:
    hm_max = [[] for x in range(nb_joint)]
    hm_max_avg = {}
for idx, (images, label_map, center_map, imgs) in enumerate(test_dataset):
    images = Variable(images.cuda())
    label_map = Variable(label_map.cuda())
    center_map = Variable(center_map.cuda()) 
    center_map = center_map[:,0,:,:]

    predict_heatmaps = net(images, center_map) 
    
    runtime_pck.append(utils.cpm_evaluation(label_map, predict_heatmaps, sigma=0.04))
    img_name.append(imgs)
    if if_save:
        utils.save_image_cpm(save_test_heatmap_path+'idx_'+str(idx), predict_heatmaps, label_map)
    if if_sum:
        for stg in predict_heatmaps[0]:
            for j, hm in enumerate(stg):
                hm_sum[j].append(float(sum(sum(hm))))
    if if_max:
        for stg in predict_heatmaps[0]:
            for j, hm in enumerate(stg):
                hm_max[j].append(float(torch.max(hm)))
                
    if idx%100 == 0:
        print str(idx)+' '+str(runtime_pck[-1])
        
avg_pck = sum(runtime_pck)/float(len(runtime_pck))

if if_sum:
    for j,s in enumerate(hm_sum):
        hm_sum_avg[j] = sum(s)/len(s)
    avg_sum = sum(hm_sum_avg.values())/len(hm_sum_avg.keys())
    hm_sum_avg['avg_overall'] = avg_sum
    hm_sum_avg['history'] = hm_sum
    json.dump(hm_sum_avg, open(save_sum_path, 'wb')) 
if if_max:
    for j,s in enumerate(hm_max):
        hm_max_avg[j] = sum(s)/len(s)
    avg_max = sum(hm_max_avg.values())/len(hm_max_avg.keys())
    hm_max_avg['avg_overall'] = avg_max
    hm_max_avg['history'] = hm_max
    json.dump(hm_max_avg, open(save_max_path, 'wb')) 
    
print " avg pck : "+str(avg_pck)
test_history = {'avg':avg_pck,'pck_all':runtime_pck, 'img_name':img_name}
json.dump(test_history, open(save_json_path, 'wb')) 

