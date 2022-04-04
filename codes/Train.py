#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ZhangX
"""
import sys # remove the path of ROS
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import cv2
import util
import time
import torch
import argparse
import numpy as np
from Loss import EVDI_loss_func
from Dataset import train_dataset
from Networks.EVDI import EVDI_Net
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
os.environ['CUDA_VISIBLE_DEVICES']="3,2,1,0" # choose GPU

def train_EVDI(opt):
    # create dirs
    util.mkdir(opt.temp_path)
    util.mkdir(opt.model_path)
    
    ## prepare dataset
    trainDataset_list = []
    for i in range(len(opt.train_path)):
        current_dataset = train_dataset(opt.train_path[i], num_bins=16, num_frames=opt.num_frames, roi_size=(128,128)) 
        trainDataset_list.append(current_dataset)
    trainDataset = ConcatDataset(trainDataset_list)
    trainLoader = DataLoader(trainDataset, batch_size=opt.bs, pin_memory=True, num_workers=4, shuffle=True, drop_last=True)
    
    ## initialize network
    print("Begin training " + opt.model_path +"...")
    net = EVDI_Net()
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    if opt.conTrain:
        net.load_state_dict(torch.load(opt.conTrain_path), strict=False)
        print("Load pretrained network from " + opt.conTrain_path)
    net = net.train()

    ## define loss function and optimizer
    criterion = EVDI_loss_func(opt.loss_wei)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    
    ## train ------
    train_loss_best = 0
    for epoch in range(opt.num_epoch):
        train_loss = 0
        running_loss = 0
        
        blur_sharp_loss = 0
        blur_event_loss = 0
        sharp_event_loss = 0
        
        start_train_time = time.time()
        print('==========================================================')
        for i, (leftB_inp1,leftB_inp2,leftB,leftB_w1,leftB_w2, 
                rightB_inp1,rightB_inp2,rightB,rightB_w1,rightB_w2,
                num_leftB, num_rightB, mid_events, leftB_coef, rightB_coef) in enumerate(trainLoader):
    
            net.zero_grad()
            optimizer.zero_grad()
            
            # load to GPU
            B,N,C,H,W = leftB_inp1.shape
            mid_events = mid_events.float().cuda() 
            leftB_coef = leftB_coef.reshape((B*N,1,1,1)).float().cuda() 
            rightB_coef = rightB_coef.reshape((B*N,1,1,1)).float().cuda() 
            
            leftB = leftB.unsqueeze(1).float().cuda() 
            leftB_w1 = leftB_w1.reshape((B*N,1,1,1)).float().cuda()
            leftB_w2 = leftB_w2.reshape((B*N,1,1,1)).float().cuda()
            leftB_inp1 = leftB_inp1.reshape((B*N,C,H,W)).float().cuda()
            leftB_inp2 = leftB_inp2.reshape((B*N,C,H,W)).float().cuda()
            
            rightB = rightB.unsqueeze(1).float().cuda() 
            rightB_w1 = rightB_w1.reshape((B*N,1,1,1)).float().cuda() 
            rightB_w2 = rightB_w2.reshape((B*N,1,1,1)).float().cuda()
            rightB_inp1 = rightB_inp1.reshape((B*N,C,H,W)).float().cuda()
            rightB_inp2 = rightB_inp2.reshape((B*N,C,H,W)).float().cuda()
          
            ## process by EVDI network
            res, Ef1, Ef2 = net(leftB_inp1, leftB_inp2, leftB_w1, leftB_w2, 
                                  rightB_inp1, rightB_inp2, rightB_w1, rightB_w2,
                                  leftB, rightB, leftB_coef, rightB_coef)
            res = res.reshape((B,N,1,H,W))
            Ef1 = Ef1.reshape((B,N,1,H,W))
            Ef2 = Ef2.reshape((B,N,1,H,W))
            
            # calculate loss 
            loss, loss_list = criterion(leftB, num_leftB, Ef1,rightB, num_rightB, Ef2,res, mid_events)
            running_loss += loss.item()
            train_loss += loss.item()
            blur_sharp_loss += loss_list[0].item()
            blur_event_loss += loss_list[1].item()
            sharp_event_loss += loss_list[2].item()
        
            loss.backward()
            optimizer.step()
            
            ## save temporary results
            if (i+1) % opt.save_int == 0:
                 print ('Epoch [%d/%d], Step [%d/%d], TrainLoss: %.5f '%(epoch+1, opt.num_epoch, i+1, len(trainDataset) // opt.bs, running_loss))
                 running_loss = 0
                 print('Learning rate:%.5f,  Time elasped: %.2f' %(optimizer.param_groups[0]['lr'], time.time()-start_train_time))
                 print('| L_B_S: %.4f | L_B_E: %.4f | L_S_E: %.4f |' %(blur_sharp_loss/(i+1), blur_event_loss/(i+1), sharp_event_loss/(i+1)))
                 idx = np.random.randint(0,opt.num_frames)
                 tmp_res = (res[0,idx,:]).squeeze()
                 show = torch.cat((leftB[0,...].squeeze(), rightB[0,...].squeeze(), tmp_res), 1)
                 show = show.cpu().detach().numpy()
                 name = opt.temp_path + str(epoch) + '-' + str(i) + '.png'
                 cv2.imwrite(name,show)
                 
        scheduler.step()
    
        ## save model when loss decreases
        print('---------------- Summary of Epoch ----------------')
        print('| L_B_S: %.4f | L_B_E: %.4f | L_S_E: %.4f |' %(blur_sharp_loss/(i+1), blur_event_loss/(i+1), sharp_event_loss/(i+1)))
        blur_sharp_loss = 0
        blur_event_loss = 0
        sharp_event_loss = 0
        
        print('Total training loss: %04f .' %(train_loss))
        if (epoch == 0):
            train_loss_best = train_loss
            print('Saving-------------')
            print('Best trainLoss: %.5f' %(train_loss_best))
            save_log_name = opt.model_path + 'epoch_%04d.pth'%(epoch+1)
            torch.save(net.state_dict(), save_log_name)
        else:
            if (train_loss_best > train_loss):
                train_loss_best = train_loss
                print('Saving-------------')
                print('Best trainLoss: %.5f' %(train_loss_best))
                save_log_name = opt.model_path + 'epoch_%04d.pth'%(epoch+1)
                torch.save(net.state_dict(), save_log_name)

 
if __name__ == '__main__':
    ## parameters
    parser = argparse.ArgumentParser(description="Train EVDI")
    parser.add_argument("--model_path", type=str, default="./PreTrained/EVDI/", help="model saving path")
    parser.add_argument("--temp_path", type=str, default="./TempRes/EVDI/", help="path to save temporal result")
    parser.add_argument("--save_int", type=int, default=50, help="epoch interval for saving temporal reconstruction result")
    parser.add_argument("--train_path", type=list, default=["./Database/train/"], help="path of training datasets")
    parser.add_argument("--conTrain", type=int, default=0, help="continue training (1) or not (0)")
    parser.add_argument("--conTrain_path", type=str, default="./PreTrained/EVDI-GoPro.pth", help="path to load model")
    parser.add_argument("--num_epoch", type=int, default=100, help="number of epoch")
    parser.add_argument("--loss_wei", type=list, default=[1,256,1e-1], help="weights for loss functions: [blur-sharp, blur-event, sharp-event]")
    parser.add_argument("--num_frames", type=int, default=49, help="recover how many frames per input, i.e., 'N' in paper.\
                        We observe that higher N leads to better performance but will need more training time (recommended N>=13). ")
    parser.add_argument("--bs", type=int, default=4, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    
    opt = parser.parse_args()
    
    train_EVDI(opt)
