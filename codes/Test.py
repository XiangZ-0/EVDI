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
import torch
import argparse
from Dataset import test_dataset
from Networks.EVDI import EVDI_Net
os.environ['CUDA_VISIBLE_DEVICES']="0" # choose GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_EVDI(opt):
    ## initialize 
    current_save_path = opt.save_path + '/'
    util.mkdir(current_save_path + 'Result/')
    util.mkdir(current_save_path + 'Blur/')
    
    ## load EVDI model 
    net = EVDI_Net()
    net = torch.nn.DataParallel(net)
    print("Testing EVDI model: "+ opt.model_path)
    net.load_state_dict(torch.load(opt.model_path,map_location=torch.device('cpu')))
    net.to(device)
    net = net.eval()
    
    ## load dataset
    testDataset = test_dataset(opt.test_path, 16, opt.test_ts)
    
    ## testing
    with torch.no_grad():
        for k in range(0, len(testDataset)):
            leftB_inp1,leftB_inp2,leftB,leftB_w1,leftB_w2, \
            rightB_inp1,rightB_inp2,rightB,rightB_w1,rightB_w2, \
                leftB_coef, rightB_coef, prefix = testDataset[k]
            print('Processing img %d ...' %(k))
            
            # CPU to GPU
            B = 1
            N,C,H,W = leftB_inp1.shape
            leftB_coef = torch.from_numpy(leftB_coef).reshape((B*N,1,1,1)).float().to(device)
            rightB_coef = torch.from_numpy(rightB_coef).reshape((B*N,1,1,1)).float().to(device)
            
            leftB = torch.from_numpy(leftB).reshape((1,1,H,W)).float().to(device) 
            leftB_w1 = torch.from_numpy(leftB_w1).reshape((B*N,1,1,1)).float().to(device) 
            leftB_w2 = torch.from_numpy(leftB_w2).reshape((B*N,1,1,1)).float().to(device)
            leftB_inp1 = torch.from_numpy(leftB_inp1).reshape((B*N,C,H,W)).float().to(device)
            leftB_inp2 = torch.from_numpy(leftB_inp2).reshape((B*N,C,H,W)).float().to(device)
            
            rightB = torch.from_numpy(rightB).reshape((1,1,H,W)).float().to(device) 
            rightB_w1 = torch.from_numpy(rightB_w1).reshape((B*N,1,1,1)).float().to(device)
            rightB_w2 = torch.from_numpy(rightB_w2).reshape((B*N,1,1,1)).float().to(device)
            rightB_inp1 = torch.from_numpy(rightB_inp1).reshape((B*N,C,H,W)).float().to(device)
            rightB_inp2 = torch.from_numpy(rightB_inp2).reshape((B*N,C,H,W)).float().to(device)
        
            ## process by EVDI network
            res, _, _ = net(leftB_inp1, leftB_inp2, leftB_w1, leftB_w2, 
                              rightB_inp1, rightB_inp2, rightB_w1, rightB_w2,
                              leftB, rightB, leftB_coef, rightB_coef)
            res = res.reshape((B,N,1,H,W))
            
            ## save blurry images & results 
            res_name = current_save_path + 'Blur/' + prefix + '_leftB.png'
            res_img = leftB[0,0,:].unsqueeze(0).permute(1,2,0).cpu()
            res_img = res_img.detach().numpy()
            cv2.imwrite(res_name, res_img)
            
            res_name = current_save_path + 'Blur/' + prefix + '_rightB.png'
            res_img = rightB[0,0,:].unsqueeze(0).permute(1,2,0).cpu()
            res_img = res_img.detach().numpy()
            cv2.imwrite(res_name, res_img)
            
            res_name = current_save_path + 'Result/' + prefix + '_%.3f'%(opt.test_ts) + '.png'
            res_img = res[0,0,:].permute(1,2,0).cpu()
            res_img = res_img.detach().numpy()
            cv2.imwrite(res_name, res_img)
                
            
if __name__ == '__main__':
    ## parameters
    parser = argparse.ArgumentParser(description="Test EVDI")
    parser.add_argument("--model_path", type=str, default="./PreTrained/EVDI-RBE.pth", help="path of pretrained model")
    parser.add_argument("--test_path", type=str, default="./Database/RBE/", help="path of test data")
    parser.add_argument("--save_path", type=str, default="./Result/EVDI-RBE/", help="saving path")
    parser.add_argument("--test_ts", type=float, default=0.5, help="test timestamp in [0,1]")
    
    opt = parser.parse_args()
    test_EVDI(opt)
