# -*- coding: utf-8 -*-
#import statements
import numpy as np
    

def Heston_model(S0, v0, kappa, theta, sigma, rho, T, M, r):
    """ Inputs:
        S0: initial asset price
        v0: initial variance
        
        mu: instantaneous rate of return on the stock
        kappa: mean-reversion speed for the variance
        theta: long-term mean of the variance process
        sigma: volatility of the variance
        rho: correlation between asset returns and variance
        
        T: time length of simulation
        N: number of time steps
        M: number of simulations
        
        Outputs:
        S: asset prices array
        v: variances array
    """
    
    #Initialise the parameters
    #time variables
    N = 200                       #number of periods per year
    dt = T/N                      #step
    timesteps = int(T * N)
    
    #means vector
    mu1 = 0   #instantaneous rate of return on the 1st stock
    mu2 = 0   #instantaneous rate of return on the 2nd stock
    mu = np.array([mu1, mu2])

    #covariance matrix
    cov = np.array([[1, rho],
                    [rho, 1]])
    
    #Create price and variance arrays
    #Assign first value to the arrays
    S = np.zeros((timesteps, M))
    v = np.zeros((timesteps, M))

    v[0, :] = sigma
    S[0, :] = S0
    
    #Generating M Correlated Random Normal Variables
    rng = np.random.default_rng()
    W1, W2 = rng.multivariate_normal(mu, cov, M).T
    Z = rng.multivariate_normal(mu, cov, (M, timesteps)).T
    Z1 = Z[0]
    Z2 = Z[1]


    #Simulating geometric Brownian motion with non-constant volatility
    a = (sigma**2)/kappa * (np.exp(-kappa * dt) - np.exp(-2 * kappa * dt)) #with analytic moments
    b = theta * sigma**2/(2*kappa)*(1 - np.exp(-kappa * dt))**2 #with analytic moments

    #Simulated variances and asset prices
    for i in range(1, timesteps):

    #maximum is used to avoid negative volatility as in Ballotta and Fusai(2018)
        v[i,:] = np.maximum(theta+(v[i-1,:]-theta)*np.exp(-kappa*dt) + np.sqrt(a*v[i-1,:]+b)*Z2[i-1,:], 0)
        S[i,:] = S[i-1,:] * np.exp((r - (0.5 * v[i-1])) * dt + np.sqrt(v[i-1,:] * dt) * Z1[i,:])
    
    return S


#Pricing of European options
def pricing(M, T, r, S, K):
    """ Inputs:
        M: number of iterations
        
        T: maturity
        r: risk-free rate
        S: simulated asset prices paths
        K: strike price
        
        Outputs:
        VcMC: mean price, European call option
        VpMC: mean price, European put option
        scMC: variance, European call option
        spMC: variance, European put option
    """
    #call option
    VcMCb = np.zeros((M, 1))

    #put option
    VpMCb = np.zeros((M, 1))

    #Monte Carlo simulation
    for j in range (1, M):

        #Pricing European call and put options
        VcMCb[j] = np.exp(-r * T) * np.mean(np.maximum(S-K,0))
        VpMCb[j] = np.exp(-r * T) * np.mean(np.maximum(K-S,0))

    #Average of the price and volatility paths
    VcMC = np.mean(VcMCb)
    VpMC = np.mean(VpMCb)

    scMC = np.sqrt(np.var(VcMCb)/M)
    spMC = np.sqrt(np.var(VpMCb)/M)
    
    return VcMC, VpMC, scMC, spMC
   