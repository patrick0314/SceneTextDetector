import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import custom_dataset
from model import EAST
from loss import Loss
import os
import time
import numpy as np


def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter):
    file_num = len(os.listdir(train_img_path)) # number of training data
    trainset = custom_dataset(train_img_path, train_gt_path)
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    
    criterion = Loss() # import from the file "loss.py", loss(score_gt, score_pred, geometry_gt, geometry_pred, ignored_map)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST() # import form the file "model.py", model(img)
    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter//2], gamma=0.1) # adaptive laerning rate
    
    odd_loss = 100
    for epoch in range(epoch_iter):    
        model.train() # training mode
        #scheduler.step() # update learning rate
        epoch_loss = 0
        epoch_time = time.time()
        for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
            start_time = time.time()
            img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
            pred_score, pred_geo = model(img) # predict
            loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map) # calculate the loss function
            
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
                                            epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))
        
        scheduler.step() # update learning rate

        print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
        print(time.asctime(time.localtime(time.time())))
        print('='*50)
        if (epoch_loss / int(file_num/batch_size)) < odd_loss: # for every lnterval times, save the model 
            print('Save model')
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch+1)))

            odd_loss = epoch_loss / int(file_num/batch_size)


if __name__ == '__main__':
    if torch.cuda.is_available():
        print('Empty Cuda')
        torch.cuda.empty_cache()
        torch.no_grad()

    #train_img_path = os.path.abspath('../ICDAR_2015/train_img') # training data x
    #train_gt_path  = os.path.abspath('../ICDAR_2015/train_gt') # training data y, groundtruth
    train_img_path = os.path.abspath('../TrainDataset/img')
    train_gt_path = os.path.abspath('../TrainDataset/gt')
    pths_path      = './pths' # model patameter save path
    batch_size     = 8 
    lr         = 1e-3
    num_workers    = 4
    epoch_iter     = 300
    train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter)    
    
