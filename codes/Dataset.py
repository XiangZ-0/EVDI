#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ZhangX
"""
import util
import numpy as np
from torch.utils.data import Dataset
from skimage.morphology import remove_small_objects

def adaptive_wei(ts,span_leftB,span_rightB):
    ## calculate weights, i.e., \omega & 1-\omega in paper
    right = 0
    left = 0
    if (span_leftB[1] < ts) and (ts < span_rightB[0]):
        right = (ts - span_leftB[0]) / (span_rightB[1] - span_leftB[0])
        left = 1 - right
    if ts >= span_rightB[0]:
        right = 1
    if ts <= span_leftB[1]:
        left = 1
    sum_value = right + left
    right = right / sum_value
    left = left / sum_value
    return left, right

#%% Train datasets           
class train_dataset(Dataset):
    def __init__(self, data_path, num_bins, num_frames, roi_size=(128,128)):
        '''
        Parameters
        ----------
        data_path : str
            path of target data.
        num_bins : int
            the number of bins in event frame.
        num_frames : int
            the number of recovered frames per input, i.e., 'N' in paper.
        roi_size : tuple
            size of region of interest. 
        '''
        self.data_path = data_path
        self.data_list = util.get_filename(self.data_path, '.npz')
        self.data_len = len(self.data_list)
        self.num_bins = num_bins
        self.num_frames = num_frames
        self.roi_size = roi_size
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self,ind):
        '''
        Parameters
        ----------
        ind : data index.

        Returns
        -------
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
        num_leftB : the number of restored latent images located within the exposure period of leftB.
        num_rightB : the number of restored latent images located within the exposure period of rightB.
        mid_events : events located between adjacent latent frames, used for sharp-event loss.
        '''
        
        ## load data
        data = np.load(self.data_path + self.data_list[ind], allow_pickle=True)
        events = data['events'].item()
        leftB = data['blur1']
        exp_start_leftB = data['exp_start1']
        exp_end_leftB = data['exp_end1']
        span_leftB = (exp_start_leftB, exp_end_leftB)
        
        rightB = data['blur2']
        exp_start_rightB = data['exp_start2']
        exp_end_rightB = data['exp_end2']
        span_rightB = (exp_start_rightB, exp_end_rightB)
        
        total_span = (exp_start_leftB, exp_end_rightB)
        
        ## crop roi
        img_size = leftB.shape
        roiTL = (np.random.randint(0, img_size[0]-self.roi_size[0]+1), np.random.randint(0, img_size[1]-self.roi_size[1]+1)) # top-left coordinate
        leftB = leftB[roiTL[0]:roiTL[0]+self.roi_size[0], roiTL[1]:roiTL[1]+self.roi_size[1]]
        rightB = rightB[roiTL[0]:roiTL[0]+self.roi_size[0], roiTL[1]:roiTL[1]+self.roi_size[1]]
        
        ## generate target timestamps
        timestamps = np.linspace(exp_start_leftB, exp_end_rightB, self.num_frames, endpoint=True) # include the last frame
        num_leftB = sum(timestamps <= exp_end_leftB)
        num_rightB = sum(timestamps >= exp_start_rightB)
        
        ## initialize lists
        ts_list = []
        leftB_inp1 = []
        leftB_inp2 = []
        leftB_w1 = []
        leftB_w2 = []
        rightB_inp1 = []
        rightB_inp2 = []
        rightB_w1 = []
        rightB_w2 = []
        mid_events = []
        leftB_coef = []
        rightB_coef = []
        
        for i in range(len(timestamps)):
            ts = timestamps[i]
            
            ## for left blurry image
            leftB_inp1_tmp, leftB_inp2_tmp, leftB_w1_tmp, leftB_w2_tmp = util.event2frame(events, self.roi_size, ts, span_leftB, total_span, self.num_bins, 0, roiTL)
            leftB_inp1_tmp = util.fold_time_dim(leftB_inp1_tmp)
            leftB_inp2_tmp = util.fold_time_dim(leftB_inp2_tmp)
            
            ## for right blurry image
            rightB_inp1_tmp, rightB_inp2_tmp, rightB_w1_tmp, rightB_w2_tmp = util.event2frame(events, self.roi_size, ts, span_rightB, total_span, self.num_bins, 0, roiTL)
            rightB_inp1_tmp = util.fold_time_dim(rightB_inp1_tmp)
            rightB_inp2_tmp = util.fold_time_dim(rightB_inp2_tmp)
            
            ## recon fusion weight 
            left_coef, right_coef = adaptive_wei(ts,span_leftB,span_rightB)
            
            if i > 0:
                mid_events_tmp = util.event_single_intergral(events, self.roi_size, (timestamps[i-1], ts), roiTL)
                mask = mid_events_tmp!=0
                masked_mid_events_tmp = remove_small_objects(mask, 2, 2) * mid_events_tmp # filter noise events
                mid_events.append(masked_mid_events_tmp)
                
            # append list
            ts_list.append(ts)
            leftB_inp1.append(leftB_inp1_tmp)
            leftB_inp2.append(leftB_inp2_tmp)
            leftB_w1.append(leftB_w1_tmp)
            leftB_w2.append(leftB_w2_tmp)
            rightB_inp1.append(rightB_inp1_tmp)
            rightB_inp2.append(rightB_inp2_tmp)
            rightB_w1.append(rightB_w1_tmp)
            rightB_w2.append(rightB_w2_tmp)
            leftB_coef.append(left_coef)
            rightB_coef.append(right_coef)
            
        # to array
        ts_list = np.array(ts_list)
        leftB_inp1 = np.array(leftB_inp1)
        leftB_inp2 = np.array(leftB_inp2)
        leftB_w1 = np.array(leftB_w1)
        leftB_w2 = np.array(leftB_w2)
        rightB_inp1 = np.array(rightB_inp1)
        rightB_inp2 = np.array(rightB_inp2)
        rightB_w1 = np.array(rightB_w1)
        rightB_w2 = np.array(rightB_w2)
        mid_events = np.array(mid_events)
        leftB_coef = np.array(leftB_coef)
        rightB_coef = np.array(rightB_coef)
        
        return leftB_inp1,leftB_inp2,leftB,leftB_w1,leftB_w2, \
            rightB_inp1,rightB_inp2,rightB,rightB_w1,rightB_w2, \
                num_leftB, num_rightB, mid_events, leftB_coef, rightB_coef


#%% Test datasets
class test_dataset(Dataset):
    def __init__(self, data_path, num_bins, target_ts):
        '''
        Parameters
        ----------
        data_path : str
            path of target data.
        num_bins : int
            the number of bins in event frame.
        target_ts : float
            target reconstruction timestamps, normalized to [0,1].

        '''
        self.data_path = data_path
        self.data_list = util.get_filename(self.data_path, '.npz')
        self.data_len = len(self.data_list)
        self.num_bins = num_bins
        self.target_ts = target_ts
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self,ind):
        '''
        Parameters
        ----------
        ind : data index.

        Returns
        -------
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
        save_prefix : prefix of image name.

        '''
        
        ## load data
        data = np.load(self.data_path + self.data_list[ind], allow_pickle=True)
        events = data['events'].item()
        leftB = data['blur1']
        exp_start_leftB = data['exp_start1']
        exp_end_leftB = data['exp_end1']
        span_leftB = (exp_start_leftB, exp_end_leftB)
        
        rightB = data['blur2']
        exp_start_rightB = data['exp_start2']
        exp_end_rightB = data['exp_end2']
        span_rightB = (exp_start_rightB, exp_end_rightB)
        
        img_size = leftB.shape
        total_span = (exp_start_leftB, exp_end_rightB)
        
        ## generate target timestamps
        time_span = exp_end_rightB - exp_start_leftB
        ts = exp_start_leftB + time_span * self.target_ts # [0,1]
        
        ## initialize lists
        leftB_inp1 = []
        leftB_inp2 = []
        leftB_w1 = []
        leftB_w2 = []
        rightB_inp1 = []
        rightB_inp2 = []
        rightB_w1 = []
        rightB_w2 = []
        leftB_coef = []
        rightB_coef = []

        ## for leftB
        leftB_inp1_tmp, leftB_inp2_tmp, leftB_w1_tmp, leftB_w2_tmp = util.event2frame(events, img_size, ts, span_leftB, total_span, self.num_bins, 0, (0,0))
        leftB_inp1_tmp = util.fold_time_dim(leftB_inp1_tmp)
        leftB_inp2_tmp = util.fold_time_dim(leftB_inp2_tmp)
        
        ## for rightB
        rightB_inp1_tmp, rightB_inp2_tmp, rightB_w1_tmp, rightB_w2_tmp = util.event2frame(events, img_size, ts, span_rightB, total_span, self.num_bins, 0, (0,0))
        rightB_inp1_tmp = util.fold_time_dim(rightB_inp1_tmp)
        rightB_inp2_tmp = util.fold_time_dim(rightB_inp2_tmp)
        
        ## recon fusion weight 
        left_coef, right_coef = adaptive_wei(ts,span_leftB,span_rightB)
        leftB_coef.append(left_coef)
        rightB_coef.append(right_coef)
        
        # # append list
        leftB_inp1.append(leftB_inp1_tmp)
        leftB_inp2.append(leftB_inp2_tmp)
        leftB_w1.append(leftB_w1_tmp)
        leftB_w2.append(leftB_w2_tmp)
        rightB_inp1.append(rightB_inp1_tmp)
        rightB_inp2.append(rightB_inp2_tmp)
        rightB_w1.append(rightB_w1_tmp)
        rightB_w2.append(rightB_w2_tmp)
            
        # to array
        leftB_inp1 = np.array(leftB_inp1)
        leftB_inp2 = np.array(leftB_inp2)
        leftB_w1 = np.array(leftB_w1)
        leftB_w2 = np.array(leftB_w2)
        rightB_inp1 = np.array(rightB_inp1)
        rightB_inp2 = np.array(rightB_inp2)
        rightB_w1 = np.array(rightB_w1)
        rightB_w2 = np.array(rightB_w2)
        leftB_coef = np.array(leftB_coef)
        rightB_coef = np.array(rightB_coef)
        
        save_prefix = self.data_list[ind][:-4]
        
        return leftB_inp1,leftB_inp2,leftB,leftB_w1,leftB_w2, \
            rightB_inp1,rightB_inp2,rightB,rightB_w1,rightB_w2, \
                 leftB_coef, rightB_coef, save_prefix
                 