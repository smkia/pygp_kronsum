import scipy as SP
import core.gp.gp_base as gp_base
import core.gp.gp_kronprod as gp_kronprod
import core.gp.gp_kronsum as gp_kronsum
import core.covariance.linear as linear
import core.covariance.diag as diag
import core.covariance.lowrank as lowrank
import core.optimize.optimize_base as optimize_base
import experiments.initialize as initialize
import numpy as np
from  sklearn.preprocessing import StandardScaler
import core.likelihood.likelihood_base as lik


def data_simulation(n_samples, n_dimensions, n_tasks, n_latent, snr):
    # true parameters
    X_c = SP.random.randn(n_tasks, n_latent)
    X_s = SP.random.randn(n_tasks, n_latent)
    X_r = SP.random.randn(n_samples, n_dimensions) * snr
    R = SP.dot(X_r, X_r.T)
    C = SP.dot(X_c, X_c.T)
    Sigma = SP.dot(X_s, X_s.T)
    K = SP.kron(C, R) + SP.kron(Sigma, SP.eye(n_samples))
    y = SP.random.multivariate_normal(SP.zeros(n_tasks * n_samples), K)
    Y = SP.reshape(y, (n_samples, n_tasks), order='F')
    return X_r, Y
    
def get_r2(Y1,Y2):
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

if __name__ == "__main__":
    # simulation settings
    n_latent = 1
    n_tasks = 4
    n_samples = 20
    n_dimensions = 5
    snr = 0.1
    
    # Data Simulation
    X, Y = data_simulation(n_samples, n_dimensions, n_tasks, n_latent, snr) 
    X_train  = X[0 : np.round(0.9 * n_samples), :]
    Y_train  = Y[0 : np.round(0.9 * n_samples), :]
    X_test  = X[np.round(0.9 * n_samples) :, :]
    Y_test  = Y[np.round(0.9 * n_samples) :, :]
    
    # Normalization
    X_scaler = StandardScaler()
    X_scaler.fit(X_train)
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    Y_scaler = StandardScaler()
    Y_scaler.fit(Y_train)
    Y_train = Y_scaler.transform(Y_train)
    
    ################################# GP_base Approach ########################
    Y_pred = np.zeros(np.shape(Y_test))
    for i in range(n_tasks):
        hyperparams, Ifilter, bounds = initialize.init('GPbase_LIN', Y_train[:,i].T, X_train, None)
        covariance = linear.LinearCF(n_dimensions = n_dimensions)
        likelihood = lik.GaussIsoLik()
        gp = gp_base.GP(covar=covariance, likelihood=likelihood)
        gp.setData(Y = Y_train[:,i:i+1], X_r = X_train)  
    
        # Training: optimize hyperparameters
        hyperparams_opt,lml_opt = optimize_base.opt_hyper(gp, hyperparams, bounds = 
                                                      bounds, Ifilter = Ifilter)
        # Testing
        Y_pred[:,i] = gp.predict(hyperparams_opt, Xstar_r = X_test)
    
    Y_pred = Y_scaler.inverse_transform(np.reshape(Y_pred,(Y_test.shape[0], Y_test.shape[1])))
    r2_GP_base = get_r2(Y_test, Y_pred)
    print np.mean(r2_GP_base)
    
    ################################# Pooling Approach ########################
    hyperparams, Ifilter, bounds = initialize.init('GPpool_LIN', Y_train.T, X_train, None)
    covar_c = linear.LinearCF(n_dimensions = 1) # vector of 1s
    covar_r = linear.LinearCF(n_dimensions = n_dimensions)
    likelihood = lik.GaussIsoLik()
    gp = gp_kronprod.KronProdGP(covar_r = covar_r, covar_c = covar_c, likelihood = likelihood)
    gp.setData(Y = Y_train, X_r = X_train, X_c = SP.ones((Y_train.shape[1],1)))

    covar_r.X = X_train
    
    # Training: optimize hyperparameters
    hyperparams_opt,lml_opt = optimize_base.opt_hyper(gp, hyperparams, bounds = 
                                                      bounds, Ifilter = Ifilter)
    # Testing
    Y_pred = gp.predict(hyperparams_opt, Xstar_r = X_test)
    Y_pred = Y_scaler.inverse_transform(Y_pred)    
    r2_GP_pool = get_r2(Y_test, Y_pred)
    print np.mean(r2_GP_pool)
    
    ################################# Kronprod Approach ########################
    hyperparams, Ifilter, bounds = initialize.init('GPkronprod_LIN', Y_train.T, 
                                                   X_train, {'n_c' : n_latent})
    covar_c = lowrank.LowRankCF(n_dimensions = n_latent)
    covar_r = linear.LinearCF(n_dimensions = n_dimensions)
    likelihood = lik.GaussIsoLik()
    gp = gp_kronprod.KronProdGP(covar_r = covar_r, covar_c = covar_c, likelihood = likelihood)
    gp.setData(Y = Y_train, X_r = X_train)

    covar_r.X = X_train
    covar_c.X = hyperparams['X_c']
    
    # Training: optimize hyperparameters
    hyperparams_opt,lml_opt = optimize_base.opt_hyper(gp, hyperparams, bounds = 
                                                      bounds, Ifilter = Ifilter)
    # Testing
    Y_pred = gp.predict(hyperparams_opt, Xstar_r = X_test)
    Y_pred = Y_scaler.inverse_transform(Y_pred)
    r2_GP_kronprod = get_r2(Y_test, Y_pred)
    print np.mean(r2_GP_kronprod)

    ################################# Kronsum Approach ########################
    hyperparams, Ifilter, bounds = initialize.init('GPkronsum_LIN', Y_train.T, 
                                                   X_train, {'n_c' : n_latent, 'n_sigma' : n_latent})
    # initialize covariance functions
    covar_c = lowrank.LowRankCF(n_dimensions = n_latent)
    covar_s = lowrank.LowRankCF(n_dimensions = n_latent)
    covar_r = linear.LinearCF(n_dimensions = n_dimensions)
    covar_o = diag.DiagIsoCF(n_dimensions = n_dimensions)  
    
    # initialize gp and its covariance functions
    covar_r.X = X_train
    covar_o.X = X_train
    covar_o._K = SP.eye(n_samples)
    covar_s.X = hyperparams['X_s']
    covar_c.X = hyperparams['X_c']
    X_o = SP.zeros((Y_train.shape[0], n_dimensions))
    
    gp = gp_kronsum.KronSumGP(covar_c = covar_c, covar_r = covar_r, covar_s = covar_s, 
                              covar_o = covar_o)
    gp.setData(Y = Y_train, X_r = X_train, X_o = X_o)  
    
    # Training: optimize hyperparameters
    hyperparams_opt,lml_opt = optimize_base.opt_hyper(gp, hyperparams, bounds = 
                                                      bounds, Ifilter = Ifilter)
                                                      
    # Testing
    Y_pred, Y_pred_cov = gp.predict(hyperparams, Xstar_r = X_test, compute_cov = True)
    
    Y_pred = Y_scaler.inverse_transform(Y_pred)
    r2_GP_kronsum = get_r2(Y_test, Y_pred)
    print np.mean(r2_GP_kronsum)
