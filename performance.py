#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np
from scipy.stats import kurtosis,skew,norm
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[63]:


# =============================================================================
# datalist=[]
# def processdata(datalist):
#     d=pd.DataFrame()
#     for name in datalist:
#         file='{}.csv'.format(name)
#         data=pd.read_csv(file).set_index(['Date'])['Adj Close']
#         data=data.loc['2007-03-01':'2019-04-01']
#         d=pd.concat([d,data],axis=1,sort=False)
#     #d.dropna()
#     return d
# 
# 
# # In[64]:
# 
# 
# datalist=['SPY','FXE','EWJ','GLD','QQQ','DBA','SHV','GAF','USO','XBI','ILF','FEZ','EPP']
# 
# 
# # In[65]:
# 
# 
# data=processdata(datalist)
# data.columns=datalist
# data=data[data['GAF'].notnull()]
# F=pd.read_csv('F1.csv')
# F=F[(F['Unnamed: 0']>='20070301')&(F['Unnamed: 0']<='20190401')]
# F=F.rename(columns={'Unnamed: 0':'Date'}).set_index('Date')
# F.index = pd.to_datetime(F.index.astype(str), format='%Y%m%d')
# data=data.join(F)
# data.to_csv(r'630data.csv')
# 
# data.head()
# 
# #logdata=data.T.iloc[0:-4].T
# #for col in logdata.columns:
# #    logdata[col]=np.log(logdata[col])-np.log(logdata[col].shift(1))
# #logret=logdata
# #logret.to_csv(r'logret.csv')
# simpledata=data.T.iloc[0:-4].T
# for col in simpledata.columns:
#     simpledata[col]=simpledata[col]-simpledata[col].shift(1)
# simpleret=simpledata  
# simpleret.to_csv(r'simpleret.csv')
# 
# 
# # In[140]:
# 
# =============================================================================

def indicator(X,rf,confidenceLevel,position):
    #PnL:
    
    daily_cum_return=np.cumprod((X+1))
    annual_cum_return = (np.power(daily_cum_return.iloc[-1,0],1/len(X)))**250
# =============================================================================
#     X['PnL']=X['simple ret'].cumsum()
#     plt.plot(X.index, X.PnL, label='PnL')
#     ax = plt.gca()
#     ax.set_xlabel('date')
#     ax.set_ylabel('PnL')
#     plt.show()
# =============================================================================
    
    #Daily Mean return(%):
    annual_arith_MeanRet=np.mean(X)*250
    
    #Geomean is zero if anyone of return rate is zero
    annual_geo_MeanRet=(np.power(daily_cum_return.iloc[-1,0],1/len(X))-1)*250
    #or
    #geoMeanRet=scipy.stats.mstats.gmean(X['simple ret rate'])
    # Min Return
    annual_MinRet = np.min(X)*250
    
    #MDD
    p_v =np.cumprod((X+1))*100
    p_v_extend = pd.DataFrame(np.append([p_v.iloc[0,0]]*9,p_v))
    Roll_Max = p_v_extend.rolling(window=10).max()
    
    tenday_Drawdown = float(np.min(p_v_extend/Roll_Max-1)[0])
    
# =============================================================================
#     Max_10day_Drawdown = round(np.max(tenday_Drawdown),4)
#     tenday_Drawdown.plot(color='r')
#     ax1=plt.gca()
#     ax1.set_xlabel('date')
#     ax1.set_ylabel('percentage drawdown')
#     plt.show()
# =============================================================================
    
    #volatility
    annual_vol=np.std(X)*np.sqrt(250)
    
    #Sharp ratio
    annualRatio=(annual_arith_MeanRet-rf)/annual_vol
    
    
    #Skewness, Kurtosis
    annual_Kurt=kurtosis(X*250)
    annual_sk=skew(X*250)
    
    #MVaR VaR
    daily_Kurt=kurtosis(X)
    daily_sk=skew(X)
    z=norm.ppf(1-confidenceLevel)
    t=z+((1/6)*(z**2-1)*daily_sk)+((1/24)*(z**3-3*z))*daily_Kurt-((1/36)*(2*z**3-5*z)*(daily_sk**2))
    mVaR= position*(np.mean(X)+t*np.std(X))*np.sqrt(250)
    alpha=norm.ppf(1-confidenceLevel, np.mean(X), np.std(X))
    VaR= position*(alpha)
    annualVaR=VaR*np.sqrt(250)
    CVaR = position*np.mean(X[X<=np.quantile(X,1-confidenceLevel)])[0]*np.sqrt(250)
    df=pd.DataFrame([annual_cum_return,annual_arith_MeanRet[0],annual_geo_MeanRet,annual_MinRet[0],tenday_Drawdown,annual_vol[0],annualRatio[0],annual_Kurt[0],annual_sk[0],mVaR[0],VaR[0],annualVaR[0],CVaR],
                    index=['Annual_cumulatedRet','Annual_ariMeanRet','Annual_geoMeanRet','Annual_MinRet','Max_10day_Drawdown','vol',
                           'Annual_SharpeRatio','Annual_Kurtosis','Annual_Skew','Annual_mVaR','Daily_VaR','Annual_VaR','Annual_CVaR'],
                    columns=['result'])
    return df
    


# In[141]:


#input return data frame, annual risk-free rate, confidence level, initial position
#before we input dataframe, we need to change the return dataframe column name to "return"
#indicator(simpleret, 0.03, 0.9, 100)


# In[ ]:




