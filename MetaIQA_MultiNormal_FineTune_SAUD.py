from __future__ import print_function, division
import os
import torch
from torch import nn
import pandas as pd

import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import time
import math
import copy
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
import torchvision

import cv2 as cv
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr, pearsonr


from EffV2AllFeatureStairMSFBHFFLinear import FCNet, FeatureNet

use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True

#torch.backends.cudnn.benchmark = True

##########################################


os.environ["CUDA_VISIBLE_DEVICES"]="0"
ModelLoad_path='pre_model/UID2021_SOTA_IQA_Meta_EffV2AllFeatureStairMSFBHFFLinear.pt' 
ModelSave_path='finetune_model/UID2021_SOTA_IQA_Meta_EffV2AllFeatureStairMSFBHFFLinear_FineSAUD_model.pt' #save finetune
ResultSave_path='result/UID2021_SOTA_EffV2AllFeatureStairMSFBHFFLinear_SAUD.txt' 
DatasetPrePath = 'database/SAUD/'


##########################################



def make_gradeint(img):
    sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)#默认ksize=3
    sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
    gradxy = cv.magnitude(sobelx,sobely )
   
    gradxy = cv.normalize(gradxy, None, 0, 1, cv.NORM_MINMAX, cv.CV_64F)
  
    img = img.astype(gradxy.dtype)
    End_image = cv.multiply(img, gradxy  )
    End_image_8U = cv.normalize(End_image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    
    End_image_8U  = Image.fromarray( End_image_8U )
    return  End_image_8U 


class ImageRatingsDataset(Dataset):
    """Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images_frame = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        # try:
        img_name = str(os.path.join(self.root_dir, str(self.images_frame.iloc[idx, 0])))
   
        
        # im
        im = Image.open(img_name).convert('RGB')
       
        # im_gra
        im_gra = cv.imread(img_name)
        im_gra = cv.cvtColor(im_gra, cv.COLOR_BGR2RGB)
        im_gra = make_gradeint(im_gra)

        rating = self.images_frame.iloc[idx, 1]

        im = self.transform(im)
        im_gra = self.transform(im_gra)

        return im, im_gra, rating



def computeSpearman(dataloader_valid, model):
    ratings = []
    predictions = []
    with torch.no_grad():
        for batch_idx, (image, image_gra, score) in enumerate(tqdm(dataloader_valid)):
            inputs_im = image
            inputs_gra = image_gra
            batch_size = inputs_im.size()[0]
            labels = score.view(batch_size, -1)
            # labels = labels / 100.0
            if use_gpu:
                try:
                    inputs_im, inputs_gra = inputs_im.float().cuda(), inputs_gra.float().cuda()
                    labels = labels.float().cuda()
                except:
                    print(inputs_im, inputs_gra, labels)
            else:
                inputs_im, inputs_gra, labels = inputs_im.float(), inputs_gra.float(), labels.float()
            outputs_a = model(inputs_im, inputs_gra)
            ratings.append(labels.float())
            predictions.append(outputs_a.float())

   

    ratings_i = np.vstack([r.cpu().numpy() for r in ratings])
    predictions_i = np.vstack([p.cpu().numpy() for p in predictions])
    a = ratings_i[:, 0]
    b = predictions_i[:, 0]
    sp = spearmanr(a, b)[0]
    pl = pearsonr(a, b)[0]
    return sp, pl


class Net(nn.Module):
    def __init__(self, headnet, net):
        super(Net, self).__init__()
        self.headnet = headnet
        self.net = net

    def forward(self, x1,x2):
        f1 = self.headnet(x1,x2)
        output = self.net(f1)
        return output
def normalization(data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range
def finetune_model():
    epochs =50
    srocc_l = []
    best_srocc = 0
    print('=============Saving Finetuned Prior Model===========')
    data_dir = os.path.join(DatasetPrePath)
    images = pd.read_csv(os.path.join(data_dir, 'image_labeled_by_score.csv'), sep=',')
    scores=images['mos']
    normalized_scores = normalization(scores)
    images['mos'] = normalized_scores
    images_fold = DatasetPrePath
    if not os.path.exists(images_fold):
        os.makedirs(images_fold)
    for i in range(0,10):
        with open(ResultSave_path, 'a') as f1:  # 设置文件对象data.txt
            print(i,file=f1)
        images_train, images_test = train_test_split(images, train_size = 0.8)
        train_path = images_fold + "train_imagenormal" +str(i+1) +".csv"
        test_path = images_fold + "test_imagenormal" +str(i+1) + ".csv"
        images_train.to_csv(train_path, sep=',', index=False)
        images_test.to_csv(test_path, sep=',', index=False)

        net1 = FeatureNet()
        net2 = FCNet()
        model = Net(headnet=net1, net=net2)
        print("create success")
        

      
        model.load_state_dict(torch.load(ModelLoad_path))
      

        for m in model.modules():
            if 'Conv' in str(type(m)):
                setattr(m, 'padding_mode', 'zeros')
        criterion = nn.L1Loss()

        optimizer = optim.Adam(model.parameters(), lr=1e-4,  weight_decay=0)
        model.cuda()

        spearman = 0
        plcc = 0
        for epoch in range(epochs):
            optimizer = exp_lr_scheduler(optimizer, epoch)

            if epoch == 0:
                dataloader_valid = load_data('train',i)
                model.eval()

                sp,pl = computeSpearman(dataloader_valid, model)
                if sp > spearman:
                    spearman = sp
                    plcc=pl
                print('no train srocc {:4f} plcc {:4f}'.format(sp,pl))


            # Iterate over data.
            #print('############# train phase epoch %2d ###############' % epoch)
            dataloader_train = load_data('train',i)
            model.train()  # Set model to training mode
            for batch_idx, (image, image_gra, score) in enumerate(tqdm(dataloader_train)):

                inputs_im = image
                inputs_gra = image_gra
                batch_size = inputs_im.size()[0]
                labels = score.view(batch_size, -1)
                # labels = labels / 100.0
                if use_gpu:
                    try:
                        inputs_im, inputs_gra = inputs_im.float().cuda(), inputs_gra.float().cuda()
                        labels = labels.float().cuda()
                    except:
                        print(inputs_im, inputs_gra, labels)
                else:
                    inputs_im, inputs_gra, labels = inputs_im.float(), inputs_gra.float(), labels.float()

                optimizer.zero_grad()
                outputs = model(inputs_im, inputs_gra)
              
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            #print('############# test phase epoch %2d ###############' % epoch)
            dataloader_valid = load_data('test',i)
            model.eval()

            sp, pl = computeSpearman(dataloader_valid, model)
            if sp > spearman:
                spearman = sp
                plcc=pl
            if sp > best_srocc:
                best_srocc = sp
                print('=====Prior model saved===Srocc:%f========'%best_srocc)
                best_model = copy.deepcopy(model)
                torch.save(best_model.cuda(),ModelSave_path)
            
            print('Validation Results - Epoch: {:2d}, PLCC: {:4f}, SROCC: {:4f}, '
                  'best SROCC: {:4f}'.format(epoch, pl, sp, spearman))

      
        with open(ResultSave_path, 'a') as f1:  # 设置文件对象data.txt
            print('{:4f},{:4f}'.format(plcc, spearman),file=f1)
  




def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=13):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate =  0.9**(epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


output_size = (384, 384)
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((448, 448)),
    torchvision.transforms.RandomHorizontalFlip(0.5),
    torchvision.transforms.RandomCrop(size=output_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((384, 384)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])



def load_data(mod = 'train',i=0):

    bsize = 20
    data_dir = os.path.join(DatasetPrePath)
    traincsv_name="train_imagenormal" +str(i+1) +".csv"
    testcsv_name="test_imagenormal" +str(i+1) + ".csv"
    train_path = os.path.join(data_dir,  traincsv_name)
    test_path = os.path.join(data_dir,  testcsv_name)


    transformed_dataset_train = ImageRatingsDataset(csv_file=train_path,
                                                    root_dir=DatasetPrePath,
                                                    transform=train_transforms)
    transformed_dataset_valid = ImageRatingsDataset(csv_file=test_path,
                                                    root_dir=DatasetPrePath,
                                                    transform=test_transforms)

    if mod == 'train':
        dataloader = DataLoader(transformed_dataset_train, batch_size=bsize,
                                  shuffle=True, num_workers=4, collate_fn=my_collate)
    else:
        dataloader = DataLoader(transformed_dataset_valid, batch_size= 20,
                                    shuffle=False, num_workers=4, collate_fn=my_collate)

    return dataloader

finetune_model()
