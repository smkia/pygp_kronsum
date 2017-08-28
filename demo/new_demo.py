#!/usr/bin/python2.7

import scipy as SP
import core.gp.gp_base as gp_base
import core.gp.gp_kronprod as gp_kronprod
import core.gp.gp_kronsum as gp_kronsum
import core.covariance.linear as linear
import core.covariance.diag as diag
import core.covariance.lowrank as lowrank
import core.optimize.optimize_base as optimize_base
import core.util.initialize as initialize
import numpy as np
from  sklearn.preprocessing import StandardScaler
import core.likelihood.likelihood_base as lik
from sklearn.metrics import mean_squared_error
from core.util.utilities import data_simulation, compute_r2, MSLL
import pickle

if __name__ == "__main__":
    # simulation settings
    method = 'all'
    save_path = '/home/mrstats/seykia/Results/Pygp/Simulated_Data/'
    dataset = 'demo'
    simulation_num = 10  
    n_latent = 1
    n_tasks = 2
    n_samples = 200
    n_dimensions = 10
    train_portion = 0.5
    
    results_base = dict({'Y_pred':np.zeros([simulation_num, np.int((1-train_portion)*n_samples), n_tasks]),
                         'Y_pred_cov':np.zeros([simulation_num, np.int((1-train_portion)*n_samples),n_tasks]),
                         's_n2' : np.zeros([simulation_num, n_tasks]),
                         'r2' : np.zeros([simulation_num,]),
                         'msll' : np.zeros([simulation_num,]),
                         'mse': np.zeros([simulation_num,]) 
                        })    
    results_pool = dict({'Y_pred':np.zeros([simulation_num, np.int((1-train_portion)*n_samples),n_tasks]),
                         'Y_pred_cov':np.zeros([simulation_num, np.int((1-train_portion)*n_samples),n_tasks]),
                         's_n2' : np.zeros([simulation_num, n_tasks]),
                         'r2' : np.zeros([simulation_num,]),
                         'msll' : np.zeros([simulation_num,]),
                         'mse': np.zeros([simulation_num,]) 
                        })    
    
    results_prod = dict({'Y_pred':np.zeros([simulation_num, np.int((1-train_portion)*n_samples),n_tasks]),
                         'Y_pred_cov':np.zeros([simulation_num, np.int((1-train_portion)*n_samples),n_tasks]),
                         's_n2' : np.zeros([simulation_num, n_tasks]),
                         'r2' : np.zeros([simulation_num,]),
                         'msll' : np.zeros([simulation_num,]),
                         'mse': np.zeros([simulation_num,]) 
                        })
                        
    results_sum = dict({'Y_pred':np.zeros([simulation_num, np.int((1-train_portion)*n_samples),n_tasks]),
                         'Y_pred_cov':np.zeros([simulation_num, np.int((1-train_portion)*n_samples),n_tasks]),
                         's_n2' : np.zeros([simulation_num, n_tasks]),
                         'r2' : np.zeros([simulation_num,]),
                         'msll' : np.zeros([simulation_num,]),
                         'mse': np.zeros([simulation_num,]) 
                        })
    
    for s in range(simulation_num):    
        # Data Simulation
        X_train, X_test, Y_train, Y_test, true_param = data_simulation(n_samples, n_dimensions, n_tasks, n_latent, train_portion) 
        
        # Normalization
        #X_scaler = StandardScaler()
        #X_scaler.fit(X_train)
        #X_train = X_scaler.transform(X_train)
        #X_test = X_scaler.transform(X_test)
        Y_scaler = StandardScaler()
        Y_scaler.fit(Y_train)
        #Y_train = Y_scaler.transform(Y_train)
        
        ################################# GP_base Approach ########################
        if (method == 'base' or  method == 'all'):
            for i in range(n_tasks):
                hyperparams, Ifilter, bounds = initialize.init('GPbase_LIN', Y_train[:,i].T, X_train, None)
                covariance = linear.LinearCF(n_dimensions = X_train.shape[0])
                likelihood = lik.GaussIsoLik()
                gp = gp_base.GP(covar=covariance, likelihood=likelihood)
                gp.setData(Y = Y_train[:,i:i+1], X_r = X_train)  
            
                # Training: optimize hyperparameters
                hyperparams_opt,lml_opt = optimize_base.opt_hyper(gp, hyperparams, bounds = 
                                                              bounds, Ifilter = Ifilter)
                # Testing
                results_base['Y_pred'][s,:,i], results_base['Y_pred_cov'][s,:,i]  = gp.predict(hyperparams_opt, Xstar_r = X_test)
                results_base['s_n2'][s,i] = np.exp(2 * hyperparams_opt['lik'])
            
            #results_base['Y_pred'][s,:,:] = Y_scaler.inverse_transform(results_base['Y_pred'][s,:,:])
            #results_base['Y_pred_cov'][s,:,:] = results_base['Y_pred_cov'][s,:,:] * Y_scaler.var_
            #results_base['s_n2'][s,:] = results_base['s_n2'][s,:] * Y_scaler.var_
            
            results_base['msll'][s] = MSLL(Y_test, np.squeeze(results_base['Y_pred'][s,:,:]), 
                                            np.squeeze(results_base['Y_pred_cov'][s,:,:]), 
                                            np.squeeze(results_base['s_n2'][s,:]), Y_scaler)            
               
            results_base['r2'][s] = np.mean(compute_r2(Y_test, results_base['Y_pred'][s,:,:]))
            results_base['mse'][s] = mean_squared_error(Y_test,results_base['Y_pred'][s,:,:], multioutput='uniform_average')
            print 'Dataset: %i, GP_Base: R2 = %f, MSE = %f, MSLL = %f' %(s+1, results_base['r2'][s], results_base['mse'][s], results_base['msll'][s])
        
        ################################# Pooling Approach ########################
        if (method == 'pool' or  method == 'all'): 
            hyperparams, Ifilter, bounds = initialize.init('GPpool_LIN', Y_train.T, X_train, None)
            covar_c = linear.LinearCF(n_dimensions = 1) # vector of 1s
            covar_r = linear.LinearCF(n_dimensions = X_train.shape[0])
            likelihood = lik.GaussIsoLik()
            covar_r.X = X_train
            gp = gp_kronprod.KronProdGP(covar_r = covar_r, covar_c = covar_c, likelihood = likelihood)
            gp.setData(Y = Y_train, X_r = X_train, X_c = SP.ones((Y_train.shape[1],1)))
            
            # Training: optimize hyperparameters
            hyperparams_opt,lml_opt = optimize_base.opt_hyper(gp, hyperparams, bounds = 
                                                              bounds, Ifilter = Ifilter)
            # Testing
            results_pool['Y_pred'][s,:,:], results_pool['Y_pred_cov'][s,:,:] = gp.predict(hyperparams_opt, Xstar_r = X_test, compute_cov = True)
            
            #results_pool['Y_pred'][s,:,:] = Y_scaler.inverse_transform(results_pool['Y_pred'][s,:,:])    
            #results_pool['Y_pred_cov'][s,:,:] = results_pool['Y_pred_cov'][s,:,:] * Y_scaler.var_
            results_pool['s_n2'][s,:] = likelihood.Kdiag(hyperparams_opt['lik'],n_tasks) #* Y_scaler.var_
            
            results_pool['msll'][s] = MSLL(Y_test, np.squeeze(results_pool['Y_pred'][s,:,:]), 
                                            np.squeeze(results_pool['Y_pred_cov'][s,:,:]), 
                                            np.squeeze(results_pool['s_n2'][s,:]), Y_scaler)
                
            results_pool['r2'][s] = np.mean(compute_r2(Y_test, results_pool['Y_pred'][s,:,:]))
            results_pool['mse'][s] = mean_squared_error(Y_test, results_pool['Y_pred'][s,:,:], multioutput='uniform_average')
            print 'Dataset: %i, GP_Pool: R2 = %f, MSE = %f, MSLL = %f' %(s+1,results_pool['r2'][s], results_pool['mse'][s], results_pool['msll'][s])
        
        
        ################################# Kronprod Approach ########################
        if (method == 'prod' or  method == 'all'):              
            hyperparams, Ifilter, bounds = initialize.init('GPkronprod_LIN', Y_train.T, X_train, {'n_c' : n_latent})
            covar_c = lowrank.LowRankCF(n_dimensions = n_latent)
            covar_r = linear.LinearCF(n_dimensions = X_train.shape[0])
            likelihood = lik.GaussIsoLik()
            covar_r.X = X_train
            covar_c.X = hyperparams['X_c']
            gp = gp_kronprod.KronProdGP(covar_r = covar_r, covar_c = covar_c, likelihood = likelihood)
            gp.setData(Y = Y_train, X_r = X_train)
            
            # Training: optimize hyperparameters
            hyperparams_opt,lml_opt = optimize_base.opt_hyper(gp, hyperparams, bounds = 
                                                              bounds, Ifilter = Ifilter)
            # Testing
            results_prod['Y_pred'][s,:,:], results_prod['Y_pred_cov'][s,:,:] = gp.predict(hyperparams_opt, Xstar_r = X_test , compute_cov = True)
            
            #results_prod['Y_pred'][s,:,:] = Y_scaler.inverse_transform(results_prod['Y_pred'][s,:,:])
            #results_prod['Y_pred_cov'][s,:,:] = results_prod['Y_pred_cov'][s,:,:] * Y_scaler.var_
            results_prod['s_n2'][s,:] = likelihood.Kdiag(hyperparams_opt['lik'],n_tasks) #* Y_scaler.var_
            
            results_prod['msll'][s] = MSLL(Y_test, np.squeeze(results_prod['Y_pred'][s,:,:]), 
                                            np.squeeze(results_prod['Y_pred_cov'][s,:,:]), 
                                            np.squeeze(results_prod['s_n2'][s,:]), Y_scaler)
            
            results_prod['r2'][s] = np.mean(compute_r2(Y_test, results_prod['Y_pred'][s,:,:]))
            results_prod['mse'][s] = mean_squared_error(Y_test, results_prod['Y_pred'][s,:,:], multioutput='uniform_average')
            print 'Dataset: %i, GP_Kronprod: R2 = %f, MSE = %f, MSLL = %f' %(s+1,results_prod['r2'][s], results_prod['mse'][s], results_prod['msll'][s]) 
            
    
        ################################# Kronsum Approach ########################
        if (method == 'sum' or  method == 'all'):     
            
            hyperparams, Ifilter, bounds = initialize.init('GPkronsum_LIN', Y_train.T, 
                                                           X_train, {'n_c' : n_latent, 'n_sigma' : n_latent})
            covar_c = lowrank.LowRankCF(n_dimensions = n_latent)
            covar_s = lowrank.LowRankCF(n_dimensions = n_latent)
            covar_r = linear.LinearCF(n_dimensions = X_train.shape[0])
            covar_o = diag.DiagIsoCF(n_dimensions = X_train.shape[0])  
            
            # initialize gp and its covariance functions
            covar_r.X = X_train
            covar_o.X = X_train
            covar_o._K = SP.eye(X_train.shape[0])
            covar_s.X = hyperparams['X_s']
            covar_c.X = hyperparams['X_c']

            gp = gp_kronsum.KronSumGP(covar_c = covar_c, covar_r = covar_r, covar_s = covar_s, 
                                      covar_o = covar_o)
            gp.setData(Y = Y_train)  
            # Training: optimize hyperparameters
            hyperparams_opt,lml_opt = optimize_base.opt_hyper(gp, hyperparams, bounds = 
                                                              bounds, Ifilter = Ifilter)
            
            # Testing
            results_sum['Y_pred'][s,:,:], results_sum['Y_pred_cov'][s,:,:] = gp.predict(hyperparams_opt, Xstar_r = X_test, compute_cov = True)
            
            #results_sum['Y_pred'][s,:,:] = Y_scaler.inverse_transform(results_sum['Y_pred'][s,:,:])
            #results_sum['Y_pred_cov'][s,:,:] = results_sum['Y_pred_cov'][s,:,:] * Y_scaler.var_
            results_sum['s_n2'][s,:] = np.diag(covar_s.K(hyperparams_opt['covar_s'])) #* Y_scaler.var_
            
            results_sum['msll'][s] = MSLL(Y_test, np.squeeze(results_sum['Y_pred'][s,:,:]), 
                                            np.squeeze(results_sum['Y_pred_cov'][s,:,:]), 
                                            np.squeeze(results_sum['s_n2'][s,:]), Y_scaler)            
        
            results_sum['r2'][s] = np.mean(compute_r2(Y_test, results_sum['Y_pred'][s,:,:]))
            results_sum['mse'][s] = mean_squared_error(Y_test, results_sum['Y_pred'][s,:,:], multioutput='uniform_average')
            print 'Dataset: %i, GP_Kronsum: R2 = %f, MSE = %f, MSLL = %f' %(s+1,results_sum['r2'][s], results_sum['mse'][s], results_sum['msll'][s]) 

########################################## Saving the results#############
        with open(save_path + dataset + '_Results_base.pkl', 'wb') as f:
            pickle.dump(results_base, f)
            
        with open(save_path + dataset + '_Results_pool.pkl', 'wb') as f:
            pickle.dump(results_pool, f)
        
        with open(save_path + dataset + '_Results_prod.pkl', 'wb') as f:
                    pickle.dump(results_prod, f)
                                     
        with open(save_path + dataset + '_Results_sum.pkl', 'wb') as f:
                pickle.dump(results_sum, f)  
                             