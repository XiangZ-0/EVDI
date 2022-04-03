#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ZhangX
"""
import os
import numpy as np

def get_filename(path,suffix):
    ## function used to get file names
    namelist=[]
    filelist = os.listdir(path)
    for i in filelist:
        if os.path.splitext(i)[1] == suffix:
            namelist.append(i)
    namelist.sort()
    return namelist

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normalize(img, max_val=255.):
    if (img.max() == img.min()):
        return img # no normalization
    else:
        return (img - img.min()) * max_val / (img.max() - img.min())

def fold_time_dim(inp):
    if inp.ndim == 4:
        T,C,H,W = inp.shape
        out = inp.reshape((T*C, H, W)) #[T,C,H,W] -> [T*C,H,W]
    elif inp.ndim == 5:
        N,T,C,H,W = inp.shape
        out = inp.reshape((N,T*C, H, W)) #[N,T,C,H,W] -> [N,T*C,H,W]
    return out

def filter_events(event_data, start, end):
    ## filter events based on temporal dimension
    x = event_data['x'][event_data['t']>=start]
    y = event_data['y'][event_data['t']>=start]
    p = event_data['p'][event_data['t']>=start]
    t = event_data['t'][event_data['t']>=start]
    
    x = x[t<=end]
    y = y[t<=end]
    p = p[t<=end]
    t = t[t<=end]
    return x,y,p,t

def filter_events_by_space(key,x1,x2,x3, start, end): 
    ## filter events based on spatial dimension
    # start inclusive and end exclusive
    new_x1 = x1[key>=start]
    new_x2 = x2[key>=start]
    new_x3 = x3[key>=start]
    new_key = key[key>=start]
    
    new_x1 = new_x1[new_key<end]
    new_x2 = new_x2[new_key<end]
    new_x3 = new_x3[new_key<end]
    new_key = new_key[new_key<end]

    return new_key,new_x1,new_x2,new_x3

def e2f_detail(event,eframe,ts,key_t,interval, noise, roiTL, img_size):
    T,C,H,W = eframe.shape
    eframe = eframe.ravel()
    if key_t < ts:
        ## reverse event time & porlarity
        x,y,p,t = filter_events(event,key_t,ts) # filter events by time
        x,y,p,t = filter_events_by_space(x,y,p,t,roiTL[1], roiTL[1]+img_size[1]) # filter events by x dim
        y,x,p,t = filter_events_by_space(y,x,p,t,roiTL[0], roiTL[0]+img_size[0]) # filter events by y dim
        x -= roiTL[1] # shift minima to zero
        y -= roiTL[0] # shift minima to zero
        new_t = ts - t
        idx = np.floor(new_t / interval).astype(int)
        idx[idx == T] -= 1
        # assert(idx.max()<T)
        p[p == -1] = 0 # reversed porlarity
        np.add.at(eframe, x + y*W + p*W*H + idx*W*H*C, 1)
    else:
        x,y,p,t = filter_events(event,ts,key_t) # filter events by time
        x,y,p,t = filter_events_by_space(x,y,p,t,roiTL[1], roiTL[1]+img_size[1]) # filter events by x dim
        y,x,p,t = filter_events_by_space(y,x,p,t,roiTL[0], roiTL[0]+img_size[0]) # filter events by y dim
        x -= roiTL[1] # shift minima to zero
        y -= roiTL[0] # shift minima to zero
        new_t = t - ts
        idx = np.floor(new_t / interval).astype(int)
        idx[idx == T] -= 1
        # assert(idx.max()<T)
        p[p == 1] = 0 # pos in channel 0
        p[p == -1] = 1 # neg in channel 1
        np.add.at(eframe, x + y*W + p*W*H + idx*W*H*C, 1)
    
    assert noise>=0 and noise<=1
    if noise>0:
        num_noise = int(noise * len(t))
        img_size = (H, W)
        noise_x = np.random.randint(0,img_size[1],(num_noise,1))
        noise_y = np.random.randint(0,img_size[0],(num_noise,1))
        noise_p = np.random.randint(0,2,(num_noise,1))
        noise_t = np.random.randint(0,idx+1,(num_noise,1))
        # add noise
        np.add.at(eframe, noise_x + noise_y*W + noise_p*W*H + noise_t*W*H*C, 1)
        
    eframe = np.reshape(eframe, (T,C,H,W))
    
    return eframe

def event2frame(event, img_size, ts, f_span, total_span, num_frame, noise, roiTL=(0,0)):
    ## convert event streams to [T, C, H, W] event tensor, C=2 indicates polarity
    f_start, f_end = f_span 
    total_start, total_end = total_span
        
    preE = np.zeros((num_frame, 2, img_size[0], img_size[1]))
    postE = np.zeros((num_frame, 2, img_size[0], img_size[1]))
    interval = (total_end - total_start) / num_frame # based on whole event range
    
    if event['t'].shape[0] > 0:
        preE = e2f_detail(event,preE,ts,f_start,interval, noise, roiTL, img_size)
        postE = e2f_detail(event,postE,ts,f_end,interval, noise, roiTL, img_size)

    pre_coef = (ts - f_start) / (f_end - f_start)
    post_coef = (f_end - ts) / (f_end - f_start)
    
    return preE, postE, pre_coef, post_coef 

def event_single_intergral(event, img_size, span, roiTL=(0,0)):
    ## generate event frames for sharp-event loss
    start, end = span
    H, W = img_size
    event_img = np.zeros((H, W)).ravel()
    
    x,y,p,t = filter_events(event, start, end) # filter events by temporal dim
    x,y,p,t = filter_events_by_space(x,y,p,t,roiTL[1], roiTL[1]+img_size[1]) # filter events by x dim
    y,x,p,t = filter_events_by_space(y,x,p,t,roiTL[0], roiTL[0]+img_size[0]) # filter events by y dim
    x -= roiTL[1] # shift minima to zero
    y -= roiTL[0] # shift minima to zero
    
    np.add.at(event_img, x + y*W, p)
    event_img = event_img.reshape((H,W))
    
    return event_img