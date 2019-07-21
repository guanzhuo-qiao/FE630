# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:18:11 2019

@author: Qiao Guanzhuo
"""
import os 
import pandas as pd
os.chdir(r"D:\Grad 2\630\630assignment\Final")
import numpy as np
import omega_w


def back_test(etf_return, ff_data, return_period, variance_period, lamb, beta_tm):
    port_return_list = []
    omega_list = []
    omega_p = np.array([1/13]*13)
    look_back =  max(return_period,variance_period)
    next_chang_date = look_back-1
    for i in range(len(etf_return)):
        omega_list.append(omega_p)
        today_return = np.asarray(etf_return.iloc[i,:])
        pr = np.dot(omega_p,today_return)
        port_return_list.append(pr)
        if i == next_chang_date:
            omega_p = omega_w.get_omega(return_r = etf_return.iloc[i+1-return_period:i+1],        
                                factor_r =ff_data.iloc[i+1-return_period:i+1],
                                return_v = etf_return.iloc[i+1-variance_period:i+1],
                                factor_v = ff_data.iloc[i+1-variance_period:i+1],
                                lamb_ = lamb,
                                beta_tm_ = beta_tm,
                                wp_ = omega_p)
            next_chang_date += 5
        else:
            continue
    return port_return_list,omega_list










