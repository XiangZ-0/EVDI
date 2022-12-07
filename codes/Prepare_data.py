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
import argparse
import numpy as np

def pack_data_for_training(opt):
    events_path = os.path.join(opt.input_path, 'Events.txt')
    blur_path = os.path.join(opt.input_path, 'Blur')
    exp_start_path = os.path.join(opt.input_path, 'Exposure_start.txt')
    exp_end_path = os.path.join(opt.input_path, 'Exposure_end.txt')
    
    blur_name = util.get_filename(blur_path, '.png')
    exp_start = util.load_timestamps_from_txt(exp_start_path)
    exp_end = util.load_timestamps_from_txt(exp_end_path)
    events = util.load_event_from_txt(events_path, img_size=opt.size)
    
    assert len(exp_start) == len(exp_end) == len(blur_name)
    num_data = len(blur_name) // 2
    for i in range(num_data):
        print("Processing data %d ..."%(i))
        if opt.color_flag:
            blur1 = cv2.imread(os.path.join(blur_path, blur_name[i*2])).transpose(2,0,1) # 3, H, W
            blur2 = cv2.imread(os.path.join(blur_path, blur_name[i*2+1])).transpose(2,0,1) # 3, H, W
        else:
            blur1 = cv2.imread(os.path.join(blur_path, blur_name[i*2]), 0)[None,...] # 1, H, W
            blur2 = cv2.imread(os.path.join(blur_path, blur_name[i*2+1]), 0)[None,...] # 1, H, W
        exp_start1 = exp_start[i*2] # exposure start time of blur1 (ns in int format)
        exp_start2 = exp_start[i*2+1]  # exposure start time of blur2 (ns in int format)
        exp_end1 = exp_end[i*2] # exposure end time of blur1 (ns in int format)
        exp_end2 = exp_end[i*2+1]  # exposure end time of blur2 (ns in int format)
        x,y,p,t = util.filter_events(events, start=exp_start1, end=exp_end2)
        target_events = dict()
        target_events['x'] = x.astype(np.int16)
        target_events['y'] = y.astype(np.int16)
        target_events['t'] = t.astype(np.int64)
        target_events['p'] = p.astype(np.int8)
        
        save_name = os.path.join(opt.save_path, '%06d.npz' %(i))
        np.savez(save_name, events=target_events, \
                 blur1=blur1, exp_start1=exp_start1, exp_end1=exp_end1, \
                     blur2=blur2, exp_start2=exp_start2, exp_end2=exp_end2)
        
    print("Task finished.")
                
            
if __name__ == '__main__':
    ## parameters
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument("--input_path", type=str, default="./Database/Raw/", help="path of input data")
    parser.add_argument("--save_path", type=str, default="./Database/Packed/", help="saving path")
    parser.add_argument("--size", type=tuple, default=(260,346), help="size of images and events (H,W)")
    parser.add_argument("--color_flag", type=int, default=0, help="use color (1) or gray (0) image")
    
    opt = parser.parse_args()
    util.mkdir(opt.save_path)
    pack_data_for_training(opt)
