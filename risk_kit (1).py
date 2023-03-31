import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize


def get_ffme_returns():
    me_m=pd.read_csv("Portfolios_Formed_on_ME_monthly_EW.csv",header=0,index_col=0,na_values=-99.99)
    rets=me_m[['Lo 10','Hi 10']]
    rets.columns=['SmallCap','LargeCap']
    rets=rets/100
    rets.index=pd.to_datetime(rets.index,format="%Y%m").to_period('M')
    return rets

def drawdown(return_series : pd.Series):
    wealth_index=1000*(1+return_series).cumprod()
    previous_peaks=wealth_index.cummax()
    drawdowns=(wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth":wealth_index,
                         "Previous_Peak":previous_peaks,
                         "Drawdown":drawdowns})
    
    
def get_hfi_returns():
    hfi=pd.read_csv("edhec-hedgefundindices.csv",header=0,index_col=0,parse_dates=True)
    hfi=hfi/100
    hfi.index=hfi.index.to_period('M')
    return hfi

def get_ind_returns():
    ind=pd.read_csv("ind30_m_vw_rets.csv", header=0, index_col=0,parse_dates=True)
    ind=ind/100
    ind.index=pd.to_datetime(ind.index,format="%Y%m").to_period("M")
    ind.columns=ind.columns.str.strip()
    return ind

def get_ind_nfirms():
    ind=pd.read_csv("ind30_m_nfirms.csv", header=0, index_col=0,parse_dates=True)
    ind.index=pd.to_datetime(ind.index,format="%Y%m").to_period("M")
    ind.columns=ind.columns.str.strip()
    return ind

def get_ind_size():
    ind=pd.read_csv("ind30_m_size.csv", header=0, index_col=0,parse_dates=True)
    ind.index=pd.to_datetime(ind.index,format="%Y%m").to_period("M")
    ind.columns=ind.columns.str.strip()
    return ind
    

def skewness(r):
    demeaned_r=r-r.mean()
    sigma_r=r.std()
    exp=(demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    demeaned_r=r-r.mean()
    sigma_r=r.std()
    exp=(demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level=0.01):
    result=scipy.stats.jarque_bera(r)
    return result[1]>level
 
def semideviation(r):
    return r[r<0].std()

def var_historic(r, level=5):
    return np.percentile(r, level)

def var_gaussian(r, level=5):
    z_score=norm.ppf(level/100)
    return r.mean()-(-z_score*r.std())

def var_cornish(r,level=5):
    z_score=norm.ppf(level/100)
    s=skewness(r)
    k=kurtosis(r)
    z_score=(z_score+(z_score**2-1)*s/6 + (z_score**3 - 3*z_score)*(k-3)/24 - (2*z_score**3 - 5*z_score)*(s**2)/36)
    return r.mean()-(-z_score*r.std())

def cvar(r, level=5):
    is_beyond= r<=var_cornish(r, level=level)
    return r[is_beyond].mean()

def ann_vol(r):
    monthly_volatility=r.std()
    annualized_vol=monthly_volatility*np.sqrt(12)
    return annualized_vol
def ann_returns(r):
    number_of_periods=r.shape[0]
    
    ann_returns=(((1+r).prod())**(12/number_of_periods))-1
    return ann_returns
def sharpe_ratio_ind(r, risk_free_rate):
    number_of_periods=r.shape[0]
    ann_returns=(((1+r).prod())**(12/number_of_periods))-1
    monthly_volatility=r.std()
    annualized_vol=monthly_volatility*np.sqrt(12)
    return (ann_returns-risk_free_rate)/annualized_vol
    
def portfolio_return(weights,returns):
    return np.transpose(weights) @ returns
    
def portfolio_vol(weights, covmat):
    return (np.transpose(weights) @ covmat @ weights)**0.5  

def minimize_vol(target_return, er,cov):
    n=er.shape[0]
    init_guess= np.repeat(1/n,n)
    #constraints 
    bounds=((0.0,1.0),)*n   #tuple of tuples
    return_is_target={
        'type':'eq',
        'args':(er,),
        'fun': lambda weights,er: target_return -portfolio_return(weights,er)
    }
    weights_sum_to_1= {
        'type': 'eq',
        'fun': lambda weights : np.sum(weights)-1
    }
    
    results = minimize(portfolio_vol, init_guess, args=(cov,), method="SLSQP",
                      options={'disp':False}, constraints=(return_is_target, weights_sum_to_1),
                       bounds=bounds)
    return results.x
def optimal_weights(n_points,er, cov):
    target_rs=np.linspace(er.min(), er.max(), n_points)
    weights=[minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def plot_ef(n_points,er, cov):
    "Plots the multi-asset efficient frontier"
    weights=optimal_weights(n_points,er,cov)
    rets=[portfolio_return(w,er) for w in weights]
    vols=[portfolio_vol(w,cov) for w in weights]
    ef=pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility",y="Returns",style='.-')



    