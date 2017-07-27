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