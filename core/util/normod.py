# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 16:26:41 2017

@author: Mosi
"""
import numpy as np
from scipy.stats import norm, genextreme, trim_mean


def normative_prob_map(Y, Y_hat, S_hat_2, S_n_2):
    NPM = (Y - Y_hat) / np.sqrt(S_hat_2 + np.repeat(np.expand_dims(S_n_2,1).T, Y.shape[0], axis = 0))
    return NPM
    
def threshold_NPM(NPM, alpha):
    p_values = norm.cdf(-np.abs(NPM))
    mask = FDR(p_values, alpha)
    return NPM * mask.astype(np.int)
    
def FDR(p_values, alpha):
    dim = np.shape(p_values)
    p_values = np.reshape(p_values,[np.prod(dim),])
    sorted_p_values = np.sort(p_values)
    sorted_p_values_idx = np.argsort(p_values);  
    testNum = len(p_values)
    thresh = ((np.array(range(testNum)) + 1)/np.float(testNum))  * alpha
    h = sorted_p_values <= thresh
    unsort = np.argsort(sorted_p_values_idx)
    h = h[unsort]
    h = np.reshape(h, dim)
    return h

def extreme_value_prob(NPM, perc):
    n = NPM.shape[0]
    t = NPM.shape[1]
    n_perc = int(round(t * perc))
    m = np.zeros(n)
    for i in range(n):
        temp =  np.abs(NPM[i, :])
        temp = np.sort(temp)
        temp = temp[t - n_perc:]
        m[i] = trim_mean(temp, 0.05)
    params = genextreme.fit(m)	
    ev = genextreme(params[0])
    probs = ev.cdf(m)
    return probs
    