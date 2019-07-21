# -*- coding: utf-8 -*-
"""
Created on Fri May 10 21:05:16 2019

@author: Qiao Guanzhuo
"""

import os 
import pandas as pd
os.chdir(r"D:\Grad 2\630\630assignment\Final")
import backtest
import performance
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
from sklearn.neighbors import KernelDensity

all_data = pd.read_csv("630data.csv",index_col=0)
ff_factor = all_data.loc[:,"Mkt-RF":r"RF"]
etf_data = all_data.loc[:,:"EPP"]


etf_return = (etf_data/etf_data.shift(1)-1).dropna(axis = 0)
ff_data = ff_factor.iloc[1:,:]



# before crisis
bc_test_etf_return = etf_return.loc[:"2008-03-03",:]
bc_test_ff_data = ff_data.loc[:"2008-03-03",:]
bc_lookback_list = [[60,60],[60,120],[120,180]]
bc_beta_list = [0.5,1,2]
bc_performance_result = pd.DataFrame([])
bc_return_result = pd.DataFrame([])
omega_list = []
for lb in bc_lookback_list:
    for bt in bc_beta_list:
        res = backtest.back_test(etf_return = bc_test_etf_return,
                                 ff_data = bc_test_ff_data,
                                 return_period = lb[0],
                                 variance_period = lb[1],
                                 lamb = 10,
                                 beta_tm = bt)
        omega_list.append(res[1])
        res = pd.DataFrame(res[0],index = pd.to_datetime(bc_test_etf_return.index))
        res_perf = performance.indicator(X = res,rf = 0.06, confidenceLevel = 0.95, position = 100)
        bc_return_result = pd.concat([bc_return_result,res],axis = 1)
        bc_performance_result = pd.concat([bc_performance_result,res_perf],axis = 1)
        
bc_return_result = pd.concat([bc_return_result,bc_test_etf_return['SPY']],axis = 1)

bc_spy_performance = performance.indicator(X = pd.DataFrame(bc_test_etf_return.loc[:,'SPY']),rf = 0.06, confidenceLevel = 0.95, position = 100)
bc_performance_result = pd.concat([bc_performance_result,bc_spy_performance],axis = 1)
bc_performance_result.columns = [['E[r]_by_60,Cov_by_60','E[r]_by_60,Cov_by_60','E[r]_by_60,Cov_by_60',
                      'E[r]_by_60,Cov_by_120','E[r]_by_60,Cov_by_120','E[r]_by_60,Cov_by_120',
                      'E[r]_by_120,Cov_by_180','E[r]_by_120,Cov_by_180','E[r]_by_120,Cov_by_180','SPY'],
                     ['beta=0.5','beta=1.0','beta=2.0',
                      'beta=0.5','beta=1.0','beta=2.0',
                      'beta=0.5','beta=1.0','beta=2.0','']]
bc_return_result.columns = bc_performance_result.columns
bc_return

# plot 

# =============================================================================
# import scipy.stats
# from scipy.stats import norm

# =============================================================================
for i in range(10):
    plt.plot(100*(np.cumprod(bc_return_result.iloc[:,i]+1)),label = bc_return_result.columns[i][0]+', '+bc_return_result.columns[i][1])
    plt.legend()
plt.xlabel('Time')
plt.ylabel('Net Value')
plt.title('Evolution of Net Value')

# =============================================================================
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(3):
#     c =  ['r', 'g', 'b'][i]
#     z =  [0.5,1,2][i]
#     x,y = np.histogram(bc_return_result.iloc[:,i],bins = 80)
#     y = (y[:-1]+y[1:])/2
#     cs = [c] * len(x)
#     ax.bar(y, x, zs=z, zdir='y', color=cs, alpha=0.7,width = 0.006,label = bc_return_result.columns[i][0]+bc_return_result.columns[i][1])
#     ax.legend()
#     l = mlab.normpdf(y,np.mean(bc_return_result.iloc[:,i]),np.std(bc_return_result.iloc[:,i]))
#     ax.plot(y,[z]*len(y),l,color ='black',linewidth = 2.0)
# 
# =============================================================================



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(3):
    dt = bc_return_result.iloc[:,i]
    column_name = bc_return_result.columns[i]
    c =  ['r', 'g', 'b'][i]
    z =  [0.5,1,2][i]
    x,y = np.histogram(dt,bins = 80)
    x = x/len(dt)
    y = (y[:-1]+y[1:])/2
    cs = [c] * len(x)
    ax.bar(y, x, zs=z, zdir='y', color=cs, alpha=0.7,width = 0.003,label = column_name[0]+', '+column_name[1])
    ax.legend()
    
    samples = np.asarray(dt).reshape(-1,1)
    x_plot = np.linspace(-10,10,80).reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(samples)
    log_dens = kde.score_samples(x_plot)
    dens = np.exp(log_dens)
    ax.plot(x_plot/50,[z]*len(y),dens/8,color ='black',linewidth = 2.0)
ax.set_xlabel('Return')
ax.set_ylabel('Beta')
ax.set_zlabel('Frequency')
ax.set_title('Return Distribution')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(3):
    dt = bc_return_result.iloc[:,i+3]
    column_name = bc_return_result.columns[i+3]
    c =  ['r', 'g', 'b'][i]
    z =  [0.5,1,2][i]
    x,y = np.histogram(dt,bins = 80)
    x = x/len(dt)
    y = (y[:-1]+y[1:])/2
    cs = [c] * len(x)
    ax.bar(y, x, zs=z, zdir='y', color=cs, alpha=0.7,width = 0.003,label = column_name[0]+', '+column_name[1])
    ax.legend()
    
    samples = np.asarray(dt).reshape(-1,1)
    x_plot = np.linspace(-10,10,80).reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(samples)
    log_dens = kde.score_samples(x_plot)
    dens = np.exp(log_dens)
    ax.plot(x_plot/60,[z]*len(y),dens/8,color ='black',linewidth = 2.0)
ax.set_xlabel('Return')
ax.set_ylabel('Beta')
ax.set_zlabel('Frequency')
ax.set_title('Return Distribution')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(3):
    dt = bc_return_result.iloc[:,i+6]
    column_name = bc_return_result.columns[i+6]
    c =  ['r', 'g', 'b'][i]
    z =  [0.5,1,2][i]
    x,y = np.histogram(dt,bins = 80)
    x = x/len(dt)
    y = (y[:-1]+y[1:])/2
    cs = [c] * len(x)
    ax.bar(y, x, zs=z, zdir='y', color=cs, alpha=0.7,width = 0.003,label = column_name[0]+', '+column_name[1])
    ax.legend()
    
    samples = np.asarray(dt).reshape(-1,1)
    x_plot = np.linspace(-10,10,80).reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(samples)
    log_dens = kde.score_samples(x_plot)
    dens = np.exp(log_dens)
    ax.plot(x_plot/70,[z]*len(y),dens/5,color ='black',linewidth = 2.0)
ax.set_xlabel('Return')
ax.set_ylabel('Beta')
ax.set_zlabel('Frequency')
ax.set_title('Return Distribution')






  


# During crisis
dc_test_etf_return = etf_return.loc["2008-03-03":"2010-09-01",:]
dc_test_ff_data = ff_data.loc["2008-03-03":"2010-09-01",:]
dc_lookback_list = [[60,60],[60,120],[120,180]]
dc_beta_list = [0.5,1,2]
dc_performance_result = pd.DataFrame([])
dc_return_result = pd.DataFrame([])
omega_list = []
for lb in dc_lookback_list:
    for bt in dc_beta_list:
        res = backtest.back_test(etf_return = dc_test_etf_return,
                                 ff_data = dc_test_ff_data,
                                 return_period = lb[0],
                                 variance_period = lb[1],
                                 lamb = 10,
                                 beta_tm = bt)
        omega_list.append(res[1])
        res = pd.DataFrame(res[0],index = pd.to_datetime(dc_test_etf_return.index))
        res_perf = performance.indicator(X = res,rf = 0.06, confidenceLevel = 0.95, position = 100)
        dc_return_result = pd.concat([dc_return_result,res],axis = 1)
        dc_performance_result = pd.concat([dc_performance_result,res_perf],axis = 1)
        
dc_return_result = pd.concat([dc_return_result,dc_test_etf_return['SPY']],axis = 1)

dc_spy_performance = performance.indicator(X = pd.DataFrame(dc_test_etf_return.loc[:,'SPY']),rf = 0.06, confidenceLevel = 0.95, position = 100)
dc_performance_result = pd.concat([dc_performance_result,dc_spy_performance],axis = 1)
dc_performance_result.columns = [['E[r]_by_60,Cov_by_60','E[r]_by_60,Cov_by_60','E[r]_by_60,Cov_by_60',
                      'E[r]_by_60,Cov_by_120','E[r]_by_60,Cov_by_120','E[r]_by_60,Cov_by_120',
                      'E[r]_by_120,Cov_by_180','E[r]_by_120,Cov_by_180','E[r]_by_120,Cov_by_180','SPY'],
                     ['beta=0.5','beta=1.0','beta=2.0',
                      'beta=0.5','beta=1.0','beta=2.0',
                      'beta=0.5','beta=1.0','beta=2.0','']]
dc_return_result.columns = dc_performance_result.columns
dc_performance_result.to_csv('dc_performance_result.csv')
# plot 

# =============================================================================
# import scipy.stats
# from scipy.stats import norm
# from sklearn.neighbors import KernelDensity
# =============================================================================
for i in range(10):
    plt.plot(100*(np.cumprod(dc_return_result.iloc[:,i]+1)),label = dc_return_result.columns[i][0]+', '+dc_return_result.columns[i][1])
    plt.legend()
plt.xlabel('Time')
plt.ylabel('Net Value')
plt.title('Evolution of Net Value')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(3):
    dt = dc_return_result.iloc[:,i]
    column_name = dc_return_result.columns[i]
    c =  ['r', 'g', 'b'][i]
    z =  [0.5,1,2][i]
    x,y = np.histogram(dt,bins = 80)
    x = x/len(dt)
    y = (y[:-1]+y[1:])/2
    cs = [c] * len(x)
    ax.bar(y, x, zs=z, zdir='y', color=cs, alpha=0.7,width = 0.003,label = column_name[0]+', '+column_name[1])
    ax.legend()
    
    samples = np.asarray(dt).reshape(-1,1)
    x_plot = np.linspace(-10,10,80).reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(samples)
    log_dens = kde.score_samples(x_plot)
    dens = np.exp(log_dens)
    ax.plot(x_plot/30,[z]*len(y),dens/10,color ='black',linewidth = 2.0)
ax.set_xlabel('Return')
ax.set_ylabel('Beta')
ax.set_zlabel('Frequency')
ax.set_title('Return Distribution')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(3):
    dt = dc_return_result.iloc[:,i+3]
    column_name = dc_return_result.columns[i+3]
    c =  ['r', 'g', 'b'][i]
    z =  [0.5,1,2][i]
    x,y = np.histogram(dt,bins = 80)
    x = x/len(dt)
    y = (y[:-1]+y[1:])/2
    cs = [c] * len(x)
    ax.bar(y, x, zs=z, zdir='y', color=cs, alpha=0.7,width = 0.003,label = column_name[0]+', '+column_name[1])
    ax.legend()
    
    samples = np.asarray(dt).reshape(-1,1)
    x_plot = np.linspace(-10,10,80).reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(samples)
    log_dens = kde.score_samples(x_plot)
    dens = np.exp(log_dens)
    ax.plot(x_plot/40,[z]*len(y),dens/9,color ='black',linewidth = 2.0)
ax.set_xlabel('Return')
ax.set_ylabel('Beta')
ax.set_zlabel('Frequency')
ax.set_title('Return Distribution')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(3):
    dt = dc_return_result.iloc[:,i+6]
    column_name = dc_return_result.columns[i+6]
    c =  ['r', 'g', 'b'][i]
    z =  [0.5,1,2][i]
    x,y = np.histogram(dt,bins = 80)
    x = x/len(dt)
    y = (y[:-1]+y[1:])/2
    cs = [c] * len(x)
    ax.bar(y, x, zs=z, zdir='y', color=cs, alpha=0.7,width = 0.003,label = column_name[0]+', '+column_name[1])
    ax.legend()
    
    samples = np.asarray(dt).reshape(-1,1)
    x_plot = np.linspace(-10,10,80).reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(samples)
    log_dens = kde.score_samples(x_plot)
    dens = np.exp(log_dens)
    ax.plot(x_plot/50,[z]*len(y),dens/9,color ='black',linewidth = 2.0)
ax.set_xlabel('Return')
ax.set_ylabel('Beta')
ax.set_zlabel('Frequency')
ax.set_title('Return Distribution')
#--------------------------------------------------------------------------------------------------------
# After crisis
ac_test_etf_return = etf_return.loc["2010-09-01":'2015-01-02',:]
ac_test_ff_data = ff_data.loc["2010-09-01":'2015-01-02',:]
ac_lookback_list = [[60,60],[60,120],[120,180]]
ac_beta_list = [0.5,1,2]
ac_performance_result = pd.DataFrame([])
ac_return_result = pd.DataFrame([])
omega_list = []
for lb in ac_lookback_list:
    for bt in ac_beta_list:
        res = backtest.back_test(etf_return = ac_test_etf_return,
                                 ff_data = ac_test_ff_data,
                                 return_period = lb[0],
                                 variance_period = lb[1],
                                 lamb = 10,
                                 beta_tm = bt)
        omega_list.append(res[1])
        res = pd.DataFrame(res[0],index = pd.to_datetime(ac_test_etf_return.index))
        res_perf = performance.indicator(X = res,rf = 0.06, confidenceLevel = 0.95, position = 100)
        ac_return_result = pd.concat([ac_return_result,res],axis = 1)
        ac_performance_result = pd.concat([ac_performance_result,res_perf],axis = 1)
        
ac_return_result = pd.concat([ac_return_result,ac_test_etf_return['SPY']],axis = 1)

ac_spy_performance = performance.indicator(X = pd.DataFrame(ac_test_etf_return.loc[:,'SPY']),rf = 0.06, confidenceLevel = 0.95, position = 100)
ac_performance_result = pd.concat([ac_performance_result,ac_spy_performance],axis = 1)
ac_performance_result.columns = [['E[r]_by_60,Cov_by_60','E[r]_by_60,Cov_by_60','E[r]_by_60,Cov_by_60',
                      'E[r]_by_60,Cov_by_120','E[r]_by_60,Cov_by_120','E[r]_by_60,Cov_by_120',
                      'E[r]_by_120,Cov_by_180','E[r]_by_120,Cov_by_180','E[r]_by_120,Cov_by_180','SPY'],
                     ['beta=0.5','beta=1.0','beta=2.0',
                      'beta=0.5','beta=1.0','beta=2.0',
                      'beta=0.5','beta=1.0','beta=2.0','']]
ac_return_result.columns = ac_performance_result.columns
ac_performance_result.to_csv('ac_performance_result.csv')

for i in range(10):
    plt.plot(100*(np.cumprod(ac_return_result.iloc[:,i]+1)),label = ac_return_result.columns[i][0]+', '+ac_return_result.columns[i][1])
    plt.legend()
plt.xlabel('Time')
plt.ylabel('Net Value')
plt.title('Evolution of Net Value')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(3):
    dt = ac_return_result.iloc[:,i]
    column_name = ac_return_result.columns[i]
    c =  ['r', 'g', 'b'][i]
    z =  [0.5,1,2][i]
    x,y = np.histogram(dt,bins = 80)
    x = x/len(dt)
    y = (y[:-1]+y[1:])/2
    cs = [c] * len(x)
    ax.bar(y, x, zs=z, zdir='y', color=cs, alpha=0.7,width = 0.003,label = column_name[0]+', '+column_name[1])
    ax.legend()
    
    samples = np.asarray(dt).reshape(-1,1)
    x_plot = np.linspace(-10,10,80).reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(samples)
    log_dens = kde.score_samples(x_plot)
    dens = np.exp(log_dens)
    ax.plot(x_plot/20,[z]*len(y),dens/10,color ='black',linewidth = 2.0)
ax.set_xlabel('Return')
ax.set_ylabel('Beta')
ax.set_zlabel('Frequency')
ax.set_title('Return Distribution')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(3):
    dt = ac_return_result.iloc[:,i+3]
    column_name = ac_return_result.columns[i+3]
    c =  ['r', 'g', 'b'][i]
    z =  [0.5,1,2][i]
    x,y = np.histogram(dt,bins = 80)
    x = x/len(dt)
    y = (y[:-1]+y[1:])/2
    cs = [c] * len(x)
    ax.bar(y, x, zs=z, zdir='y', color=cs, alpha=0.7,width = 0.003,label = column_name[0]+', '+column_name[1])
    ax.legend()
    
    samples = np.asarray(dt).reshape(-1,1)
    x_plot = np.linspace(-10,10,80).reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(samples)
    log_dens = kde.score_samples(x_plot)
    dens = np.exp(log_dens)
    ax.plot(x_plot/20,[z]*len(y),dens/8,color ='black',linewidth = 2.0)
ax.set_xlabel('Return')
ax.set_ylabel('Beta')
ax.set_zlabel('Frequency')
ax.set_title('Return Distribution')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(3):
    dt = ac_return_result.iloc[:,i+6]
    column_name = ac_return_result.columns[i+6]
    c =  ['r', 'g', 'b'][i]
    z =  [0.5,1,2][i]
    x,y = np.histogram(dt,bins = 80)
    x = x/len(dt)
    y = (y[:-1]+y[1:])/2
    cs = [c] * len(x)
    ax.bar(y, x, zs=z, zdir='y', color=cs, alpha=0.7,width = 0.003,label = column_name[0]+', '+column_name[1])
    ax.legend()
    
    samples = np.asarray(dt).reshape(-1,1)
    x_plot = np.linspace(-10,10,80).reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(samples)
    log_dens = kde.score_samples(x_plot)
    dens = np.exp(log_dens)
    ax.plot(x_plot/30,[z]*len(y),dens/7,color ='black',linewidth = 2.0)
ax.set_xlabel('Return')
ax.set_ylabel('Beta')
ax.set_zlabel('Frequency')
ax.set_title('Return Distribution')

#---------------------------------------------------------------------------------------------------
# Whole Period
wp_test_etf_return = etf_return.loc[:,:]
wp_test_ff_data = ff_data.loc[:,:]
wp_lookback_list = [[60,60],[60,120],[120,180]]
wp_beta_list = [0.5,1,2]
wp_performance_result = pd.DataFrame([])
wp_return_result = pd.DataFrame([])
omega_list = []
for lb in wp_lookback_list:
    for bt in wp_beta_list:
        res = backtest.back_test(etf_return = wp_test_etf_return,
                                 ff_data = wp_test_ff_data,
                                 return_period = lb[0],
                                 variance_period = lb[1],
                                 lamb = 10,
                                 beta_tm = bt)
        omega_list.append(res[1])
        res = pd.DataFrame(res[0],index = pd.to_datetime(wp_test_etf_return.index))
        res_perf = performance.indicator(X = res,rf = 0.06, confidenceLevel = 0.95, position = 100)
        wp_return_result = pd.concat([wp_return_result,res],axis = 1)
        wp_performance_result = pd.concat([wp_performance_result,res_perf],axis = 1)
        
wp_return_result = pd.concat([wp_return_result,wp_test_etf_return['SPY']],axis = 1)

wp_spy_performance = performance.indicator(X = pd.DataFrame(wp_test_etf_return.loc[:,'SPY']),rf = 0.06, confidenceLevel = 0.95, position = 100)
wp_performance_result = pd.concat([wp_performance_result,wp_spy_performance],axis = 1)
wp_performance_result.columns = [['E[r]_by_60,Cov_by_60','E[r]_by_60,Cov_by_60','E[r]_by_60,Cov_by_60',
                      'E[r]_by_60,Cov_by_120','E[r]_by_60,Cov_by_120','E[r]_by_60,Cov_by_120',
                      'E[r]_by_120,Cov_by_180','E[r]_by_120,Cov_by_180','E[r]_by_120,Cov_by_180','SPY'],
                     ['beta=0.5','beta=1.0','beta=2.0',
                      'beta=0.5','beta=1.0','beta=2.0',
                      'beta=0.5','beta=1.0','beta=2.0','']]
wp_return_result.columns = wp_performance_result.columns
wp_performance_result.to_csv('wp_performance_result.csv')

# plot 

# =============================================================================
# import scipy.stats
# from scipy.stats import norm
# from sklearn.neighbors import KernelDensity
# =============================================================================
for i in range(10):
    plt.plot(100*(np.cumprod(wp_return_result.iloc[:,i]+1)),label = wp_return_result.columns[i][0]+', '+wp_return_result.columns[i][1])
    plt.legend()
plt.xlabel('Time')
plt.ylabel('Net Value')
plt.title('Evolution of Net Value')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(3):
    dt = wp_return_result.iloc[:,i]
    column_name = wp_return_result.columns[i]
    c =  ['r', 'g', 'b'][i]
    z =  [0.5,1,2][i]
    x,y = np.histogram(dt,bins = 80)
    x = x/len(dt)
    y = (y[:-1]+y[1:])/2
    cs = [c] * len(x)
    ax.bar(y, x, zs=z, zdir='y', color=cs, alpha=0.7,width = 0.003,label = column_name[0]+', '+column_name[1])
    ax.legend()
    
    samples = np.asarray(dt).reshape(-1,1)
    x_plot = np.linspace(-10,10,80).reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(samples)
    log_dens = kde.score_samples(x_plot)
    dens = np.exp(log_dens)
    ax.plot(x_plot/20,[z]*len(y),dens/8,color ='black',linewidth = 2.0)
ax.set_xlabel('Return')
ax.set_ylabel('Beta')
ax.set_zlabel('Frequency')
ax.set_title('Return Distribution')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(3):
    dt = wp_return_result.iloc[:,i+3]
    column_name = wp_return_result.columns[i+3]
    c =  ['r', 'g', 'b'][i]
    z =  [0.5,1,2][i]
    x,y = np.histogram(dt,bins = 80)
    x = x/len(dt)
    y = (y[:-1]+y[1:])/2
    cs = [c] * len(x)
    ax.bar(y, x, zs=z, zdir='y', color=cs, alpha=0.7,width = 0.003,label = column_name[0]+', '+column_name[1])
    ax.legend()
    
    samples = np.asarray(dt).reshape(-1,1)
    x_plot = np.linspace(-10,10,80).reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(samples)
    log_dens = kde.score_samples(x_plot)
    dens = np.exp(log_dens)
    ax.plot(x_plot/20,[z]*len(y),dens/8,color ='black',linewidth = 2.0)
ax.set_xlabel('Return')
ax.set_ylabel('Beta')
ax.set_zlabel('Frequency')
ax.set_title('Return Distribution')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(3):
    dt = wp_return_result.iloc[:,i+6]
    column_name = wp_return_result.columns[i+6]
    c =  ['r', 'g', 'b'][i]
    z =  [0.5,1,2][i]
    x,y = np.histogram(dt,bins = 80)
    x = x/len(dt)
    y = (y[:-1]+y[1:])/2
    cs = [c] * len(x)
    ax.bar(y, x, zs=z, zdir='y', color=cs, alpha=0.7,width = 0.003,label = column_name[0]+', '+column_name[1])
    ax.legend()
    
    samples = np.asarray(dt).reshape(-1,1)
    x_plot = np.linspace(-10,10,80).reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(samples)
    log_dens = kde.score_samples(x_plot)
    dens = np.exp(log_dens)
    ax.plot(x_plot/20,[z]*len(y),dens/7,color ='black',linewidth = 2.0)
ax.set_xlabel('Return')
ax.set_ylabel('Beta')
ax.set_zlabel('Frequency')
ax.set_title('Return Distribution')