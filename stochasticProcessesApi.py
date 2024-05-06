# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request,send_file
import pandas as pd
from HestonModel import *
from GBM import *
from yahoofinance import *
from BlackScholes import *


app = Flask(__name__)


@app.route('/Heston', methods = ['GET'])
#Heston model
def getHeston():
    M = request.args.get('M', type=int, default = 1000)
    S0 = request.args.get('S0', type=float, default = 1.0)
    K = request.args.get('K', type=float, default = 1.1)
    T = request.args.get('T', type=int, default = 1)
    r = request.args.get('r', type=float, default = getRiskFreeRate())
    v0 = request.args.get('v0', type=float) 
    rho = request.args.get('rho', type=float)
    kappa = request.args.get('kappa', type=float) 
    theta = request.args.get('theta', type=float) 
    sigma = request.args.get('sigma', type=float)
        
    S = Heston_model(S0, v0, kappa, theta, sigma, rho, T, M, r)
    VcMC, VpMC, scMC, spMC = pricing(M, T, r, S, K)
   
    return jsonify({
        "Heston":{
        "mean price European call": VcMC,
        "variance European call": scMC,
            
        "mean price European put": VpMC,
        "variance European put": spMC
        }}), 200


@app.route('/GBM', methods = ['GET'])
#GeometricBrownianMotion
def getGBM():
    M = request.args.get('M', type=int, default = 1000)
    S0 = request.args.get('S0', type=float, default = 1.0)
    K = request.args.get('K', type=float, default = 1.1)
    T = request.args.get('T', type=int, default = 1)
    r = request.args.get('r', type=float, default = getRiskFreeRate())
    sigma = request.args.get('sigma', type=float)

    S = geom_brow_motion(S0, r, sigma, T, M)
    VcMC, VpMC, scMC, spMC = pricing(M, T, r, S, K)
    
    return jsonify({
        "GBM":{
        "mean price European call": VcMC,
        "variance European call": scMC,
            
        "mean price European put": VpMC,
        "variance European put": spMC
        }}), 200



#BlackScholes
@app.route('/BS', methods = ['GET']) 
def getBS():
    S0 = request.args.get('S0', type=float, default = 1.0)
    K = request.args.get('K', type=float, default = 1.1)
    T = request.args.get('T', type=int, default = 1)
    r = request.args.get('r', type=float, default = getRiskFreeRate())
    sigma = request.args.get('sigma', type=float)
    
    VcMC = BS_CALL(S0, K, T, r, sigma)
    VpMC = BS_PUT(S0, K, T, r, sigma)
    
    return jsonify({
        "BlackScholes":{
        "mean price European call": VcMC,
        "mean price European put": VpMC,
        }}), 200


if __name__ == '__main__':
    app.run(debug=True)
