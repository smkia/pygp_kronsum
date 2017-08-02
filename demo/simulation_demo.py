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
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
from core.util.utilities import compute_r2
from core.util.normod import extreme_value_prob, normative_prob_map, threshold_NPM
from sklearn.metrics import roc_auc_score
import seaborn as sns
import pickle

if __name__ == "__main__":
    
    matContent = loadmat('.\\demo\\simulated_data.mat')
    
    n_samples = 150
    n_tasks = 100
    
    X_train  = matContent['X_train'][:n_samples,:]
    Y_train  = matContent['Y_train'][:n_samples,:n_tasks]
    X_test  = matContent['X_test'][:n_samples,:]
    Y_test  = matContent['Y_test'][:n_samples,:n_tasks]
    labels = np.squeeze(matContent['labels'])
    
    n_samples = X_train.shape[0]
    n_tasks = Y_train.shape[1]
    n_dimensions = X_train.shape[1]
    n_latent = 10
    
    # Normalization
    X_scaler = StandardScaler()
    X_scaler.fit(X_train)
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    Y_scaler = StandardScaler()
    Y_scaler.fit(Y_train)
    Y_train = Y_scaler.transform(Y_train)
    
    ################################# GP_base Approach ########################
    Y_pred_base = np.zeros(np.shape(Y_test))
    Y_pred_cov_base = np.zeros(np.shape(Y_test))
    s_n2_base = np.zeros([n_tasks,])
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
        Y_pred_base[:,i], Y_pred_cov_base[:,i]  = gp.predict(hyperparams_opt, Xstar_r = X_test)
        s_n2_base[i] = np.exp(2 * hyperparams_opt['lik'])
    
    Y_pred_base = Y_scaler.inverse_transform(np.reshape(Y_pred_base,(Y_test.shape[0], Y_test.shape[1])))
    Y_pred_cov_base = Y_pred_cov_base * Y_scaler.var_
    s_n2_base = s_n2_base * Y_scaler.var_
    
    NPM_base = normative_prob_map(Y_test, Y_pred_base, Y_pred_cov_base, s_n2_base)
    abnormal_prob_base = extreme_value_prob(NPM_base, 0.01)
    auc_base = roc_auc_score(labels,abnormal_prob_base)
    
    #sns.heatmap(np.reshape(threshold_NPM(NPM_base[41,:], 0.05),[10,10]))
    
    r2_GP_base = compute_r2(Y_test, Y_pred_base)
    mse_GP_base = mean_squared_error(Y_test,Y_pred_base, multioutput='uniform_average')
    print 'GP_Base: R2 = %f, MSE = %f' %(np.mean(r2_GP_base), mse_GP_base)
    
    with open('Results_base.pkl', 'wb') as f:
        pickle.dump(dict(Y_pred_base = Y_pred_base, Y_pred_cov_base = Y_pred_cov_base, 
                         s_n2_base = s_n2_base, NPM_base = NPM_base, 
                         abnormal_prob_base = abnormal_prob_base, auc_base = auc_base,
                         r2_GP_base = r2_GP_base, mse_GP_base = mse_GP_base), f)

    #with open('Results_base.pkl', 'rb') as f:
    #    a = pickle.load(f)
    
    ################################# Pooling Approach ########################
    hyperparams, Ifilter, bounds = initialize.init('GPpool_LIN', Y_train.T, X_train, None)
    covar_c = linear.LinearCF(n_dimensions = 1) # vector of 1s
    covar_r = linear.LinearCF(n_dimensions = n_dimensions)
    likelihood = lik.GaussIsoLik()
    covar_r.X = X_train
    gp = gp_kronprod.KronProdGP(covar_r = covar_r, covar_c = covar_c, likelihood = likelihood)
    gp.setData(Y = Y_train, X_r = X_train, X_c = SP.ones((Y_train.shape[1],1)))
    
    # Training: optimize hyperparameters
    hyperparams_opt,lml_opt = optimize_base.opt_hyper(gp, hyperparams, bounds = 
                                                      bounds, Ifilter = Ifilter)
    # Testing
    Y_pred_pool, Y_pred_cov_pool = gp.predict(hyperparams_opt, Xstar_r = X_test, compute_cov = True)
    
    Y_pred_pool = Y_scaler.inverse_transform(Y_pred_pool)    
    Y_pred_cov_pool = Y_pred_cov_pool * Y_scaler.var_
    s_n2_pool = np.exp(2 * hyperparams_opt['lik']) * Y_scaler.var_    
    
    NPM_pool = normative_prob_map(Y_test, Y_pred_pool, Y_pred_cov_pool, s_n2_pool)
    abnormal_prob_pool = extreme_value_prob(NPM_pool, 0.01)
    auc_pool = roc_auc_score(labels, abnormal_prob_pool)
    
    #sns.heatmap(np.reshape(threshold_NPM(NPM_pool[140,:], 0.05),[10,10]))
    
    r2_GP_pool = compute_r2(Y_test, Y_pred_pool)
    mse_GP_pool = mean_squared_error(Y_test, Y_pred_pool, multioutput='uniform_average')
    print 'GP_Pool: R2 = %f, MSE = %f' %(np.mean(r2_GP_pool), mse_GP_pool)
    
    with open('Results_pool.pkl', 'wb') as f:
        pickle.dump(dict(Y_pred_pool = Y_pred_pool, Y_pred_cov_pool = Y_pred_cov_pool, 
                         s_n2_pool = s_n2_pool, NPM_pool = NPM_pool, 
                         abnormal_prob_pool = abnormal_prob_pool, auc_pool = auc_pool,
                         r2_GP_pool = r2_GP_pool, mse_GP_pool = mse_GP_pool), f)
    
    ################################# Kronprod Approach ########################
    hyperparams, Ifilter, bounds = initialize.init('GPkronprod_LIN', Y_train.T, 
                                                   X_train, {'n_c' : n_latent})
    covar_c = lowrank.LowRankCF(n_dimensions = n_latent)
    covar_r = linear.LinearCF(n_dimensions = n_dimensions)
    likelihood = lik.GaussIsoLik()
    covar_r.X = X_train
    covar_c.X = hyperparams['X_c']
    gp = gp_kronprod.KronProdGP(covar_r = covar_r, covar_c = covar_c, likelihood = likelihood)
    gp.setData(Y = Y_train, X_r = X_train)
    
    # Training: optimize hyperparameters
    hyperparams_opt,lml_opt = optimize_base.opt_hyper(gp, hyperparams, bounds = 
                                                      bounds, Ifilter = Ifilter)
    # Testing
    Y_pred_prod, Y_pred_cov_prod = gp.predict(hyperparams_opt, Xstar_r = X_test , compute_cov = True)
    
    Y_pred_prod = Y_scaler.inverse_transform(Y_pred_prod)
    Y_pred_cov_prod = Y_pred_cov_prod * Y_scaler.var_
    s_n2_prod = np.exp(2 * hyperparams_opt['lik']) * Y_scaler.var_
   
    NPM_prod = normative_prob_map(Y_test, Y_pred_prod, Y_pred_cov_prod, s_n2_prod)
    abnormal_prob_prod = extreme_value_prob(NPM_prod, 0.01)
    auc_prod = roc_auc_score(labels, abnormal_prob_prod)    
    
    #sns.heatmap(np.reshape(threshold_NPM(NPM_prod[140,:], 0.05),[10,10]))
    
    r2_GP_prod = compute_r2(Y_test, Y_pred_prod)
    mse_GP_prod = mean_squared_error(Y_test, Y_pred_prod, multioutput='uniform_average')
    print 'GP_Kronprod: R2 = %f, MSE = %f' %(np.mean(r2_GP_prod), mse_GP_prod) 
    
    with open('Results_prod.pkl', 'wb') as f:
        pickle.dump(dict(Y_pred_prod = Y_pred_prod, Y_pred_cov_prod = Y_pred_cov_prod, 
                         s_n2_prod = s_n2_prod, NPM_prod = NPM_prod, 
                         abnormal_prob_prod = abnormal_prob_prod, auc_prod = auc_prod,
                         r2_GP_prod = r2_GP_prod, mse_GP_prod = mse_GP_prod), f)

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
    Y_pred_sum, Y_pred_cov_sum = gp.predict(hyperparams_opt, Xstar_r = X_test, compute_cov = True)
    
    Y_pred_sum = Y_scaler.inverse_transform(Y_pred_sum)
    Y_pred_cov_sum = Y_pred_cov_sum * Y_scaler.var_
    s_n2_sum = np.diag(np.dot(hyperparams_opt['X_s'], hyperparams_opt['X_s'].T)) * Y_scaler.var_

    NPM_sum = normative_prob_map(Y_test, Y_pred_sum, Y_pred_cov_sum, s_n2_sum)
    abnormal_prob_sum = extreme_value_prob(NPM_sum, 0.01)
    auc_sum = roc_auc_score(labels, abnormal_prob_sum)    
    
    #sns.heatmap(np.reshape(threshold_NPM(NPM_sum[140,:], 0.05),[10,10]))

    r2_GP_sum = compute_r2(Y_test, Y_pred_sum)
    mse_GP_sum = mean_squared_error(Y_test, Y_pred_sum, multioutput='uniform_average')
    print 'GP_Kronsum: R2 = %f, MSE = %f' %(np.mean(r2_GP_sum), mse_GP_sum) 

    with open('Results_sum.pkl', 'wb') as f:
        pickle.dump(dict(Y_pred_sum = Y_pred_sum, Y_pred_cov_sum = Y_pred_cov_sum, 
                         s_n2_sum = s_n2_sum, NPM_sum = NPM_sum, 
                         abnormal_prob_sum = abnormal_prob_sum, auc_sum = auc_sum,
                         r2_GP_sum = r2_GP_sum, mse_GP_sum = mse_GP_sum), f)