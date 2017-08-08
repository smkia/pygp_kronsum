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
import matplotlib.pylab as plt

if __name__ == "__main__":
    
    method = 'all'
    
    simulation_num = 10
    n_samples = 150
    n_tasks = 100
    
    results_base = dict({'Y_pred':np.zeros([simulation_num, n_samples,n_tasks]),
                         'Y_pred_cov':np.zeros([simulation_num, n_samples,n_tasks]),
                         's_n2' : np.zeros([simulation_num, n_tasks]),
                         'NPM' : np.zeros([simulation_num, n_samples,n_tasks]),
                         'abnormal_prob' : np.zeros([simulation_num, n_samples]),
                         'auc' : np.zeros([simulation_num,]),
                         'r2' : np.zeros([simulation_num,]),
                         'mse': np.zeros([simulation_num,]) 
                        })    
    results_pool = dict({'Y_pred':np.zeros([simulation_num, n_samples,n_tasks]),
                         'Y_pred_cov':np.zeros([simulation_num, n_samples,n_tasks]),
                         's_n2' : np.zeros([simulation_num, n_tasks]),
                         'NPM' : np.zeros([simulation_num, n_samples,n_tasks]),
                         'abnormal_prob' : np.zeros([simulation_num, n_samples]),
                         'auc' : np.zeros([simulation_num,]),
                         'r2' : np.zeros([simulation_num,]),
                         'mse': np.zeros([simulation_num,]) 
                        })    
    
    results_prod = dict({'Y_pred':np.zeros([simulation_num, n_samples,n_tasks]),
                         'Y_pred_cov':np.zeros([simulation_num, n_samples,n_tasks]),
                         's_n2' : np.zeros([simulation_num, n_tasks]),
                         'NPM' : np.zeros([simulation_num, n_samples,n_tasks]),
                         'abnormal_prob' : np.zeros([simulation_num, n_samples]),
                         'auc' : np.zeros([simulation_num,]),
                         'r2' : np.zeros([simulation_num,]),
                         'mse': np.zeros([simulation_num,]) 
                        })
                        
    results_sum = dict({'Y_pred':np.zeros([simulation_num, n_samples,n_tasks]),
                         'Y_pred_cov':np.zeros([simulation_num, n_samples,n_tasks]),
                         's_n2' : np.zeros([simulation_num, n_tasks]),
                         'NPM' : np.zeros([simulation_num, n_samples,n_tasks]),
                         'abnormal_prob' : np.zeros([simulation_num, n_samples]),
                         'auc' : np.zeros([simulation_num,]),
                         'r2' : np.zeros([simulation_num,]),
                         'mse': np.zeros([simulation_num,]) 
                        })
                        
    for s in range (simulation_num):
        
        matContent = loadmat('./demo/dataset' + str(s+1) +'.mat')
        
        
        
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
        if (method == 'base' or  method == 'all'):
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
                results_base['Y_pred'][s,:,i], results_base['Y_pred_cov'][s,:,i]  = gp.predict(hyperparams_opt, Xstar_r = X_test)
                results_base['s_n2'][s,i] = np.exp(2 * hyperparams_opt['lik'])
            
            results_base['Y_pred'][s,:,:] = Y_scaler.inverse_transform(results_base['Y_pred'][s,:,:])
            results_base['Y_pred_cov'][s,:,:] = results_base['Y_pred_cov'][s,:,:] * Y_scaler.var_
            results_base['s_n2'][s,:] = results_base['s_n2'][s,:] * Y_scaler.var_
            
            results_base['NPM'][s,:,:] = normative_prob_map(Y_test, results_base['Y_pred'][s,:,:], 
                                            results_base['Y_pred_cov'][s,:,:], results_base['s_n2'][s,:])
            results_base['abnormal_prob'][s,:] = extreme_value_prob(results_base['NPM'][s,:,:], 0.01)
            results_base['auc'][s] = roc_auc_score(labels,results_base['abnormal_prob'][s,:])
               
            results_base['r2'][s] = np.mean(compute_r2(Y_test, results_base['Y_pred'][s,:,:]))
            results_base['mse'][s] = mean_squared_error(Y_test,results_base['Y_pred'][s,:,:], multioutput='uniform_average')
            print 'Dataset: %i, GP_Base: R2 = %f, MSE = %f' %(s+1, results_base['r2'][s], results_base['mse'][s])
            
        ################################# Pooling Approach ########################
        if (method == 'pool' or  method == 'all'): 
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
            results_pool['Y_pred'][s,:,:], results_pool['Y_pred_cov'][s,:,:] = gp.predict(hyperparams_opt, Xstar_r = X_test, compute_cov = True)
            
            results_pool['Y_pred'][s,:,:] = Y_scaler.inverse_transform(results_pool['Y_pred'][s,:,:])    
            results_pool['Y_pred_cov'][s,:,:] = results_pool['Y_pred_cov'][s,:,:] * Y_scaler.var_
            results_pool['s_n2'][s,:] = np.exp(2 * hyperparams_opt['lik']) * Y_scaler.var_    
            
            results_pool['NPM'][s,:,:] = normative_prob_map(Y_test, results_pool['Y_pred'][s,:,:], 
                                          results_pool['Y_pred_cov'][s,:,:], results_pool['s_n2'][s,:])
            results_pool['abnormal_prob'][s,:] = extreme_value_prob(results_pool['NPM'][s,:,:], 0.01)
            results_pool['auc'][s] = roc_auc_score(labels, results_pool['abnormal_prob'][s,:])
                
            results_pool['r2'][s] = np.mean(compute_r2(Y_test, results_pool['Y_pred'][s,:,:]))
            results_pool['mse'][s] = mean_squared_error(Y_test, results_pool['Y_pred'][s,:,:], multioutput='uniform_average')
            print 'Dataset: %i, GP_Pool: R2 = %f, MSE = %f' %(s+1,results_pool['r2'][s], results_pool['mse'][s])
        
        ################################# Kronprod Approach ########################
        if (method == 'prod' or  method == 'all'):              
            hyperparams, Ifilter, bounds = initialize.init('GPkronprod_LIN', Y_train.T, X_train, {'n_c' : n_latent})
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
            results_prod['Y_pred'][s,:,:], results_prod['Y_pred_cov'][s,:,:] = gp.predict(hyperparams_opt, Xstar_r = X_test , compute_cov = True)
            
            results_prod['Y_pred'][s,:,:] = Y_scaler.inverse_transform(results_prod['Y_pred'][s,:,:])
            results_prod['Y_pred_cov'][s,:,:] = results_prod['Y_pred_cov'][s,:,:] * Y_scaler.var_
            results_prod['s_n2'][s,:] = np.exp(2 * hyperparams_opt['lik']) * Y_scaler.var_
           
            results_prod['NPM'][s,:,:] = normative_prob_map(Y_test, results_prod['Y_pred'][s,:,:], 
                                                    results_prod['Y_pred_cov'][s,:,:], results_prod['s_n2'][s,:])
            results_prod['abnormal_prob'][s,:] = extreme_value_prob(results_prod['NPM'][s,:,:], 0.01)
            results_prod['auc'][s] = roc_auc_score(labels, results_prod['abnormal_prob'][s,:])    
            
            results_prod['r2'][s] = np.mean(compute_r2(Y_test, results_prod['Y_pred'][s,:,:]))
            results_prod['mse'][s] = mean_squared_error(Y_test, results_prod['Y_pred'][s,:,:], multioutput='uniform_average')
            print 'Dataset: %i, GP_Kronprod: R2 = %f, MSE = %f' %(s+1,results_prod['r2'][s], results_prod['mse'][s]) 
            
        ################################# Kronsum Approach ########################
        if (method == 'sum' or  method == 'all'):     
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
            results_sum['Y_pred'][s,:,:], results_sum['Y_pred_cov'][s,:,:] = gp.predict(hyperparams_opt, Xstar_r = X_test, compute_cov = True)
            
            results_sum['Y_pred'][s,:,:] = Y_scaler.inverse_transform(results_sum['Y_pred'][s,:,:])
            results_sum['Y_pred_cov'][s,:,:] = results_sum['Y_pred_cov'][s,:,:] * Y_scaler.var_
            results_sum['s_n2'][s,:] = np.diag(np.dot(hyperparams_opt['X_s'], hyperparams_opt['X_s'].T)) * Y_scaler.var_
        
            results_sum['NPM'][s,:,:] = normative_prob_map(Y_test, results_sum['Y_pred'][s,:,:], 
                                         results_sum['Y_pred_cov'][s,:,:], results_sum['s_n2'][s,:])
            results_sum['abnormal_prob'][s,:] = extreme_value_prob(results_sum['NPM'][s,:,:], 0.01)
            results_sum['auc'][s] = roc_auc_score(labels, results_sum['abnormal_prob'][s,:])    
            
            results_sum['r2'][s] = np.mean(compute_r2(Y_test, results_sum['Y_pred'][s,:,:]))
            results_sum['mse'][s] = mean_squared_error(Y_test, results_sum['Y_pred'][s,:,:], multioutput='uniform_average')
            print 'Dataset: %i, GP_Kronsum: R2 = %f, MSE = %f' %(s+1,results_sum['r2'][s], results_sum['mse'][s]) 
                             
    ########################################## Saving the results#############
    with open('Results_base.pkl', 'wb') as f:
        pickle.dump(results_base, f)
        
    with open('Results_pool.pkl', 'wb') as f:
        pickle.dump(results_pool, f)
    
    with open('Results_prod.pkl', 'wb') as f:
                pickle.dump(results_prod, f)
                                 
    with open('Results_sum.pkl', 'wb') as f:
            pickle.dump(results_sum, f)  
                             
    ######################################### Ploting ########################
#    import pickle
#    with open('Results_base.pkl', 'rb') as f:
#        Results_base = pickle.load(f)
#    with open('Results_pool.pkl', 'rb') as f:
#        Results_pool = pickle.load(f)
#    with open('Results_prod.pkl', 'rb') as f:
#        Results_prod = pickle.load(f)
#    with open('Results_sum.pkl', 'rb') as f:
#        Results_sum = pickle.load(f)
#    abn_samples =  np.squeeze(np.array(np.nonzero (labels)))
#    ax = []
#    fig = plt.figure()
#    for i in range(8):
#        sns.heatmap(np.reshape(threshold_NPM(Results_base['NPM_base'][abn_samples[i],:], 0.05),[10,10]), ax = fig.add_subplot(4,8,1+i))
#        sns.heatmap(np.reshape(threshold_NPM(Results_pool['NPM_pool'][abn_samples[i],:], 0.05),[10,10]), ax = fig.add_subplot(4,8,9+i))
#        sns.heatmap(np.reshape(threshold_NPM(Results_prod['NPM_prod'][abn_samples[i],:], 0.05),[10,10]), ax = fig.add_subplot(4,8,17+i))
#        sns.heatmap(np.reshape(threshold_NPM(Results_sum['NPM_sum'][abn_samples[i],:], 0.05),[10,10]), ax = fig.add_subplot(4,8,25+i))
#    
#    plt.show()