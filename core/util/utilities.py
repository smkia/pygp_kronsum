# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 09:25:02 2017

@author: Mosi
"""
import scipy as SP

def fast_dot(A, B):
    f_dot = SP.linalg.get_blas_funcs("gemm", arrays=(A.T, B.T))
    return f_dot(alpha=1.0, a = A.T, b = B.T, trans_a = True, trans_b = True)
    
def fast_kron(A, B):
    f_kron = A[:,SP.newaxis,:,SP.newaxis] * B[SP.newaxis,:, SP.newaxis,:]
    f_kron = SP.reshape(f_kron, [A.shape[0] * B.shape[0], A.shape[1] * B.shape[1]])
    return f_kron

def partial_kron(A, B, i, RorC = 'r'):
    m, n = A.shape
    p, q = B.shape
    if RorC == 'r':
        r = SP.zeros([n*q,])
        for j in range(n):
            r[j*q:(j+1)*q,] = A[i,j] * B[i,:]
        
    elif RorC == 'c':
        r = SP.zeros([m*p,])
        for j in range(m):
            r[j*p:(j+1)*p,] = A[j,i] * B[:,i]
    return r
    
def compute_r2(Y1, Y2):
    """
    return list of squared correlation coefficients (one per task)
    """
    if Y1.ndim==1:
        Y1 = SP.reshape(Y1,(Y1.shape[0],1))
    if Y2.ndim==1:
        Y2 = SP.reshape(Y2,(Y2.shape[0],1))

    t = Y1.shape[1]
    r2 = []
    for i in range(t):
        _r2 = SP.corrcoef(Y1[:,i],Y2[:,i])[0,1]**2
        r2.append(_r2)
    r2 = SP.array(r2)
    return r2
    

def storeHashHDF5(group,RV):
    for key,value in RV.iteritems():
        if SP.isscalar(value):
            value = SP.array([value])
        group.create_dataset(key,data=value,chunks=True,compression='gzip')

def readHDF5Hash(group):
    RV = {}
    for key in group.keys():
        RV[key] = group[key][:]
    return RV

def getVariance(K):
    """get variance scaling of K"""
    c = SP.sum((SP.eye(len(K)) - (1.0 / len(K)) * SP.ones(K.shape)) * SP.array(K))
    scalar = (len(K) - 1) / c
    return 1.0/scalar


def getVarianceKron(C,R,verbose=False):
    """ get variance scaling of kron(C,R)"""
    n_K = len(C)*len(R)
    c = SP.kron(SP.diag(C),SP.diag(R)).sum() - 1./n_K * SP.dot(R.T,SP.dot(SP.ones((R.shape[0],C.shape[0])),C)).sum()
    scalar = (n_K-1)/c
    return 1.0/scalar
