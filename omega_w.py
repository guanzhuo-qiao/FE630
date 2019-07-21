

import scipy
from scipy.optimize import minimize
from sklearn import linear_model
import numpy as np



def arg_w(rho, lamb, Q, wp, beta_im_ ,beta_T):
    def constrain1(w):
        return np.dot(beta_im_,w)-beta_T
    def constrain2(w):
        return np.sum(w)-1
    cons = [{'type':'eq', 'fun': constrain1},
            {'type':'eq', 'fun': constrain2}]
    bnds = scipy.optimize.Bounds(-2.0, 2.0, keep_feasible = True)
    def f(w):
        return -rho.dot(w) + lamb*(w-wp).dot(Q.dot(w-wp))
    w0 = np.array([1/13]*13)
    res = minimize(f, w0, method='SLSQP', bounds=bnds, constraints=cons,
                   tol=1e-9)
    return res.x

def get_omega(return_r, factor_r, return_v, factor_v, lamb_, beta_tm_, wp_):
    rf = np.asarray(factor_r['RF'])
    rM_rf = np.asarray(factor_r['Mkt-RF'])
    rSMB = np.asarray(factor_r['SMB'])
    rHML = np.asarray(factor_r['HML'])
    SPY = np.asarray(return_r['SPY'])
    
    ri = np.asarray(return_r)
    
    var_market = np.var(SPY,ddof=1)
    beta_im = np.array([0.0]*13)
    for i in range (13):
        temp = np.cov(ri[:,i],SPY,ddof=1)
        beta_im[i] = temp[0,1] / var_market
    
    Ri = ri - rf.reshape(-1,1)
    f = np.array([rM_rf, rSMB, rHML])
    F = f.T
    
    lr = linear_model.LinearRegression().fit(F, Ri)
    alpha = lr.intercept_
    B = lr.coef_
    # get rho of short period
    ft = f[:,-1]
    rho_r = alpha + B.dot(ft) + rf[-1]
    
    # --------------------------------------------------------------------
    rf_v = np.asarray(factor_v['RF'])
    rM_rf_v = np.asarray(factor_v['Mkt-RF'])
    rSMB_v = np.asarray(factor_v['SMB'])
    rHML_v = np.asarray(factor_v['HML'])
    SPY_v = np.asarray(return_v['SPY'])
    
    ri_v = np.asarray(return_v)
    
    var_market_v = np.var(SPY_v,ddof=1)
    beta_im_v = np.array([0.0]*13)
    for i in range (13):
        temp_v = np.cov(ri_v[:,i],SPY_v,ddof=1)
        beta_im_v[i] = temp_v[0,1] / var_market_v
    
    Ri_v = ri_v - rf_v.reshape(-1,1)
    f_v = np.array([rM_rf_v, rSMB_v, rHML_v])
    F_v = f_v.T
    
    lr_v = linear_model.LinearRegression().fit(F_v, Ri_v)
    alpha_v = lr_v.intercept_
    B_v = lr_v.coef_
    
    eph_v = Ri_v.T - (alpha_v.reshape(-1,1) + B_v.dot(f_v))
    eph2_v = np.cov(eph_v,ddof=1)
    eph2_diag_v = np.diag(eph2_v)
    D_v = np.diag(eph2_diag_v)

    omega_f_v = np.cov(f_v,ddof=1)
    cov_Rt_v = B_v.dot(omega_f_v).dot(B_v.T) + D_v


    
    result = arg_w(rho_r, lamb_, cov_Rt_v, wp_, beta_im_v ,beta_tm_)
    
    return result




