#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ZhangX
"""
import torch
import torch.nn as nn
from Networks.networks import ResnetBlock, get_norm_layer

class ChannelAttention(nn.Module):
    ## channel attention block
    def __init__(self, in_planes, ratio=16): 
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    ## spatial attention block
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def conv_layer(inDim, outDim, ks, s, p, norm_layer='none'):
    ## convolutional layer
    conv = nn.Conv2d(inDim, outDim, kernel_size=ks, stride=s, padding=p)
    relu = nn.ReLU(True)
    assert norm_layer in ('batch', 'instance', 'none')
    if norm_layer == 'none':
        seq = nn.Sequential(*[conv, relu])
    else:
        if (norm_layer == 'instance'):
            norm = nn.InstanceNorm2d(outDim, affine=False, track_running_stats=False) # instance norm
        else:
            momentum = 0.1
            norm = nn.BatchNorm2d(outDim, momentum = momentum, affine=True, track_running_stats=True)
        seq = nn.Sequential(*[conv, norm, relu])
    return seq

def LDI_subNet(inDim=32, outDim=1, norm='none'):  
    ## LDI network
    convBlock1 = conv_layer(inDim,64,3,1,1)
    convBlock2 = conv_layer(64,128,3,1,1,norm)
    convBlock3 = conv_layer(128,64,3,1,1,norm)
    convBlock4 = conv_layer(64,16,3,1,1,norm)
    conv = nn.Conv2d(16, outDim, 3, 1, 1) 
    seq = nn.Sequential(*[convBlock1, convBlock2, convBlock3, convBlock4, conv])
    return seq

def pre_subNet(inDim=128, outDim=16, norm='none', n_blocks = 2, para=[5,2,2]):
    # sub network in fusion
    pre_net = nn.ModuleList()
    for i in range(n_blocks-1):
        pre_layer = conv_layer(inDim,inDim*2,para[0],para[1],para[2],norm)
        pre_net.append(pre_layer)
        inDim = inDim * 2
    # last layer
    pre_layer = conv_layer(inDim,outDim,para[0],para[1],para[2],norm)
    pre_net.append(pre_layer)
    return pre_net

def post_subNet(inDim=128, outDim=16, norm='none', n_blocks = 2, para=[5,2,2]):
    # sub network in fusion
    post_net = nn.ModuleList()
    for i in range(n_blocks-1):
        post_layer = conv_layer(inDim,inDim//4, para[0],para[1],para[2],norm)
        post_net.append(post_layer)
        inDim = inDim // 2
    # last layer
    post_layer = conv_layer(inDim,outDim,para[0],para[1],para[2],norm)
    post_net.append(post_layer)
    return post_net

class EVDI_Net(nn.Module):
    def __init__(self):
        super(EVDI_Net, self).__init__()
        
        ## LDI network
        self.LDI = LDI_subNet(32,1)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid() 
        
        ## fusion network
        self.convBlock1 = conv_layer(5,16,3,1,1)
        self.Pre = pre_subNet(16,64,'none',n_blocks=2,para=[3,1,1])
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.resBlock1 = ResnetBlock(64,'zero',get_norm_layer('none'), False, True)
        self.resBlock2 = ResnetBlock(64,'zero',get_norm_layer('none'), False, True)
        self.Post = post_subNet(128,16,'none',n_blocks=2, para=[3,1,1])
        self.conv = nn.Conv2d(16, 1, 3, 1, 1) 

    def forward(self, leftB_inp1, leftB_inp2, leftB_w1, leftB_w2,
                rightB_inp1, rightB_inp2, rightB_w1, rightB_w2,
                leftB, rightB, leftB_coef, rightB_coef): 
        '''
        Parameters
        ----------
        leftB : left blurry image.
        rightB : left blurry image.
        leftB_inp1 : first event segment for leftB.
        leftB_inp2 : second event segment for leftB.
        leftB_w1 : weight for first event segment (related to leftB).
        leftB_w2 : weight for second event segment (related to leftB).
        rightB_inp1 : first event segment for rightB.
        rightB_inp2 : second event segment for rightB.
        rightB_w1 : weight for first event segment (related to rightB).
        rightB_w2 : weight for second event segment (related to rightB).
        leftB_coef : coefficient for L^i_(i+1), i.e., \omega in paper.
        rightB_coef : coefficient for L^i_(i+1), i.e., 1-\omega in paper.

        Returns
        -------
        recon : final reconstruction result.
        Ef1 : learned double integral of events (related to leftB).
        Ef2 : learned double integral of events (related to rightB).
        '''
        
        ## process by LDI networks
        Ef1_tmp1 = self.LDI(leftB_inp1)
        Ef1_tmp2 = self.LDI(leftB_inp2)
        Ef1 = leftB_w1 * Ef1_tmp1 + leftB_w2 * Ef1_tmp2
        Ef1 = self.relu(Ef1) + self.sigmoid(Ef1)
        
        Ef2_tmp1 = self.LDI(rightB_inp1)
        Ef2_tmp2 = self.LDI(rightB_inp2)
        Ef2 = rightB_w1 * Ef2_tmp1 + rightB_w2 * Ef2_tmp2
        Ef2 = self.relu(Ef2) + self.sigmoid(Ef2)
        
        ## process by fusion network
        # generate recon3
        B,C,H,W = leftB.shape
        N = Ef1.shape[0] // B
        Ef1 = Ef1.reshape((B,N,C,H,W))
        Ef2 = Ef2.reshape((B,N,C,H,W))
        leftB = leftB.unsqueeze(1).repeat(1,N,1,1,1)
        rightB = rightB.unsqueeze(1).repeat(1,N,1,1,1)
        recon1 = leftB / Ef1                                                                                                   
        recon2 = rightB / Ef2 
        recon1 = recon1.reshape((B*N,C,H,W))
        recon2 = recon2.reshape((B*N,C,H,W))
        leftB = leftB.reshape((B*N,C,H,W))
        rightB = rightB.reshape((B*N,C,H,W))
        Ef1 = Ef1.reshape((B*N,C,H,W))
        Ef2 = Ef2.reshape((B*N,C,H,W))
        recon3 = recon1 * leftB_coef + recon2 * rightB_coef 
        
        # generate final result
        x = torch.cat((recon1,recon2,recon3,Ef1,Ef2), 1) 
        x = self.convBlock1(x)
        blocks = []
        for i, pre_layer in enumerate(self.Pre):
            x = pre_layer(x)
            blocks.append(x) 
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        for i, post_layer in enumerate(self.Post):
            x = torch.cat((x, blocks[len(blocks)-i-1]), 1)
            x = post_layer(x)
        x = self.conv(x)
        recon = self.sigmoid(x) * 255.
        
        return recon, Ef1, Ef2