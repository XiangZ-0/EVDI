#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ZhangX
"""
import torch
import util
import torch.nn as nn

def blur_sharp_loss(leftB, num_leftB, rightB, num_rightB, res):
    ## calculate blur-sharp loss
    B,N,C,H,W = res.shape
    if B == 1: # for batch size=1
        reblur_leftB = res[:, :num_leftB, :, :,:].mean(1) 
        reblur_rightB = res[:, -num_rightB:, :, :,:].mean(1)
    else: # for batch size>1
        reblur_leftB = torch.zeros((B,1,H,W), device=res.device)
        reblur_rightB = torch.zeros((B,1,H,W), device=res.device)
        for j in range(B):
            num_leftB_tmp = num_leftB[j]
            num_rightB_tmp = num_rightB[j]
            reblur_leftB[j,:,:,:] = res[j, :num_leftB_tmp, :, :,:].unsqueeze(0).mean(1)
            reblur_rightB[j,:,:,:] = res[j, -num_rightB_tmp:, :, :,:].unsqueeze(0).mean(1)
            
    L1 = nn.L1Loss()
    L_B_S = L1(reblur_leftB, leftB) + L1(reblur_rightB, rightB)
    
    return L_B_S

def blur_event_loss(leftB, Ef1, rightB, Ef2, res):
    ## calculate blur-event loss
    bias = 255 * 1e-5
    B,N,C,H,W = res.shape
    sub1 = torch.log(leftB + bias) - torch.log(rightB + bias) 
    sub1 = sub1.unsqueeze(1).repeat(1,N,1,1,1) 
    sub2 = torch.log(Ef1 + bias) - torch.log(Ef2 + bias) 
    
    L1 = nn.L1Loss()
    L_B_E = L1(sub1, sub2)
    
    return L_B_E

def sharp_event_loss(res, mid_events):
    ## calculate sharp-event loss
    loss = nn.L1Loss() 
    B,N,C,H,W = res.shape
    bias = 1e-3
    max_value = 255.
    L_S_E = 0
    assert C in (1,3)
    if C == 3: # convert BGR 2 GRAY
        prev_imgs = res[:,:-1,0,:,:] * 0.114 + res[:,:-1,1,:,:] * 0.587 + res[:,:-1,2,:,:] * 0.299
        prev_imgs = prev_imgs.reshape((B*(N-1),1,H,W))
        log_prev_imgs = torch.log(prev_imgs + bias)
        
        next_imgs = res[:,1:,0,:,:] * 0.114 + res[:,1:,1,:,:] * 0.587 + res[:,1:,2,:,:] * 0.299
        next_imgs = next_imgs.reshape((B*(N-1),1,H,W))
        log_next_imgs = torch.log(next_imgs + bias)
    else:
        prev_imgs = res[:,:-1,:,:,:].reshape((B*(N-1),C,H,W))
        log_prev_imgs = torch.log(prev_imgs + bias)
        next_imgs = res[:,1:,:,:,:].reshape((B*(N-1),C,H,W))
        log_next_imgs = torch.log(next_imgs + bias)
        
    mid_events = mid_events.reshape((B*(N-1),1,H,W))
    for i in range(mid_events.shape[0]):
        if (mid_events[i,...].max() == mid_events[i,...].min()):  # no events, calculate two images
            norm_prev_imgs = util.normalize(log_prev_imgs[i,...], max_val=max_value)
            norm_next_imgs = util.normalize(log_next_imgs[i,...], max_val=max_value)
            L_S_E += loss(norm_prev_imgs, norm_next_imgs) 
        else:
            mask = mid_events[i,...] != 0
            diff_imgs = log_next_imgs[i,...] - log_prev_imgs[i,...]
            norm_diff_imgs = util.normalize(diff_imgs, max_val=max_value)
            norm_mid_event = util.normalize(mid_events[i,...], max_val=max_value)
            L_S_E += loss(norm_diff_imgs * mask, norm_mid_event * mask) # with mask
    L_S_E /= (N-1)
    
    return L_S_E

class EVDI_loss_func:
    def __init__(self, weights):
        self.loss_wei = weights

    def __call__(self, leftB, num_leftB, Ef1, rightB, num_rightB, Ef2, res, mid_events):
        '''
        Parameters
        ----------
        leftB : left blurry image.
        rightB : left blurry image.
        num_leftB : the number of restored latent images located within the exposure period of leftB.
        num_rightB : the number of restored latent images located within the exposure period of rightB.
        Ef1 : learned double integral of events (related to leftB).
        Ef2 : learned double integral of events (related to rightB).
        res : reconstruction result.
        mid_events: events located between adjacent latent frames, used for sharp-event loss.

        Returns
        -------
        total_loss : total loss value.
        loss_list : loss value for each function.

        '''
    
        ## blur-sharp loss
        L_B_S = blur_sharp_loss(leftB, num_leftB, rightB, num_rightB, res)
        
        ## blur-event loss
        L_B_E = blur_event_loss(leftB, Ef1, rightB, Ef2, res)
        
        ## sharp-event loss
        L_S_E = sharp_event_loss(res, mid_events)
        
        ## total loss
        total_loss = self.loss_wei[0] * L_B_S + self.loss_wei[1] * L_B_E + self.loss_wei[2] * L_S_E
        loss_list = [self.loss_wei[0] * L_B_S, self.loss_wei[1] * L_B_E, self.loss_wei[2] *L_S_E]
        
        return total_loss, loss_list