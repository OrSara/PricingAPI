import numpy as np
import scipy.stats as asc
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt



def geom_brow_motion(S0, r, sigma, T, M):
    '''Generate Monte Carlo paths for geometric brownian motion
    
       Parameters:
           S0: float, initial stock price
           r: float, constant interest rate
           sigma: float, constant volatility
           T: int, maturity
           N: int, number of time steps
           M: int, number of simulated paths
       
       Output:
            paths: ndarray, shape (M+1,I)
            simulated paths given the parameters
    '''
    N = 200
    dt = float(T)/N
    paths = np.zeros((N+1, M), float)
    paths[0] = S0
    
    for t in range(1, N+1):
        rand = np.random.standard_normal(M)
        #rand = (rand-rand.mean())/rand.std()
        paths[t] = paths[t-1]*np.exp((r-0.5*pow(sigma,2))*dt+sigma*np.sqrt(dt)*rand)
        
    S = paths[-1]
    
    return S

