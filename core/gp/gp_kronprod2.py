import logging as LG
import scipy as SP
import scipy.linalg as LA
import copy
from gplvm import GPLVM

import core.likelihood.likelihood_base as likelihood_base
from core.util.utilities import fast_dot, fast_kron, ravel, unravel
    
class KronProdGP(GPLVM):
    """GPLVM for kronecker type covariance structures
    This class assumes the following model structure
    vec(Y) ~ GP(0, Kx \otimes Kd + \sigma^2 \unit)
    """

    __slots__ = ['covar_c','covar_r']
    
    def __init__(self, covar_r = None, covar_c = None, likelihood = None, prior = None, basis = None):
        assert isinstance(likelihood,likelihood_base.GaussIsoLik), 'likelihood is not implemented yet'
        self.covar_r = covar_r
        self.covar_c = covar_c
        self.likelihood = likelihood
        self.prior = prior
        self.basis = basis
        self._covar_cache = None
        self.Y = None
        

    def setData(self, Y, Y_hat, X = None, X_r = None, X_c = None, **kwargs):
        """
        set data
        Y:    Outputs [n x t]
        """
        assert Y.ndim == 2, 'Y must be a two dimensional vector'
        self.Y = Y
        self.n = Y.shape[0]
        self.t = Y_hat.shape[1]
        self.nt = self.n * self.t
        if X_r is not None:
            X = X_r
        if X is not None:
            self.covar_r.X = X
        if X_c is not None:
            self.covar_c.X = X_c
        self._invalidate_cache()
        
    def _update_inputs(self, hyperparams):
        """ update the inputs from gplvm model """
        if 'X_c' in hyperparams.keys():
            self.covar_c.X = hyperparams['X_c']
        if 'X_r' in hyperparams.keys():
            self.covar_r.X = hyperparams['X_r']

    def predict(self, hyperparams, Xstar_r, compute_cov = False):
        """
        predict on Xstar
        """
        self._update_inputs(hyperparams)
        KV = self.get_covariances(hyperparams)
        
        self.covar_r.Xcross = Xstar_r
        
        Kstar_r = self.covar_r.Kcross(hyperparams['covar_r'])
        Kstar_c = self.covar_c.K(hyperparams['covar_c'])  
        BKstar_cB = SP.dot(self.basis, SP.dot(Kstar_c.T, self.basis.T))

        KinvY = SP.dot(KV['U_r'], SP.dot(KV['Ytilde'], SP.dot(KV['U_c'].T, KV['Binv'])))
        
        Ystar = SP.dot(Kstar_r.T,  SP.dot(KinvY, BKstar_cB))
        
        Ystar_covar = []
        if compute_cov:
            R_star_star = SP.exp(2 * hyperparams['covar_r']) * fast_dot(Xstar_r, Xstar_r.T)
            R_tr_star = Kstar_r
            C = BKstar_cB         
            temp = fast_kron(fast_dot(C, fast_dot(KV['Binv'].T ,KV['U_c'])), fast_dot(R_tr_star.T, KV['U_r']))
            Ystar_covar = SP.diag(fast_kron(C, R_star_star)) - SP.sum((1./KV['S'] * temp).T * temp.T, axis = 0)            
            
            Ystar_covar = unravel(Ystar_covar, Xstar_r.shape[0], self.Y.shape[1])
            
        return Ystar, Ystar_covar
  
    
    def get_covariances(self,hyperparams):
        """
        INPUT:
        hyperparams:  dictionary
        OUTPUT: dictionary with the fields
        Kr:     kernel on rows
        Kc:     kernel on columns
        Knoise: noise kernel
        """
        if self._is_cached(hyperparams):
            return self._covar_cache
        if self._covar_cache==None:
            self._covar_cache = {}
            
        if not(self._is_cached(hyperparams,keys=['covar_c'])):
            K_c = self.covar_c.K(hyperparams['covar_c'])
            S_c,U_c = LA.eigh(K_c)
            self._covar_cache['K_c'] = K_c
            self._covar_cache['U_c'] = U_c
            self._covar_cache['S_c'] = S_c
        else:
            K_c = self._covar_cache['K_c']
            U_c = self._covar_cache['U_c']
            S_c = self._covar_cache['S_c']
            
        if not(self._is_cached(hyperparams,keys=['covar_r'])):
            K_r = self.covar_r.K(hyperparams['covar_r'])
            S_r,U_r = LA.eigh(K_r)
            self._covar_cache['K_r'] = K_r
            self._covar_cache['U_r'] = U_r
            self._covar_cache['S_r'] = S_r
        else:
            K_r = self._covar_cache['K_r']
            U_r = self._covar_cache['U_r']
            S_r = self._covar_cache['S_r']

        Binv = SP.linalg.pinv(self.basis)       
        S = SP.kron(S_c,S_r) + self.likelihood.Kdiag(hyperparams['lik'],self.nt)
        #UYUB = SP.dot(U_r.T, SP.dot(self.Y, SP.dot(Binv.T, SP.dot(U_c, SP.dot(U_c.T, Binv)))))
        UYUB = SP.dot(U_r.T, SP.dot(self.Y, SP.dot(Binv.T, U_c)))
        YtildeVec = (1./S) * ravel(UYUB)
        self._covar_cache['Binv'] = Binv
        self._covar_cache['S'] = S
        self._covar_cache['UYUB'] = UYUB
        self._covar_cache['Ytilde'] = unravel(YtildeVec,self.n,self.t)
        UBinv = SP.dot(U_c.T, Binv)
        self._covar_cache['UBinvB'] = SP.dot(UBinv, self.basis)
        self._covar_cache['UBinvBinvU'] = SP.dot(UBinv, UBinv.T)
        self._covar_cache['S_c_tilde'] = SP.diag(SP.dot(self._covar_cache['UBinvB'],SP.dot(K_c, self._covar_cache['UBinvB'].T)))
        self._covar_cache['hyperparams'] = copy.deepcopy(hyperparams)
        return self._covar_cache
        

    def _LML_covar(self,hyperparams):
        """
        log marginal likelihood
        """
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LML_covar')
            return 1E6
        except ValueError:
            LG.error('value error in _LML_covar')
            return 1E6
 
        lml_quad = 0.5*(KV['Ytilde']*KV['UYUB']).sum()
        lml_det =  0.5 * SP.log(KV['S']).sum()
        lml_const = 0.5*self.n*self.t*(SP.log(2*SP.pi))
        lml = lml_quad + lml_det + lml_const

        return lml

    def _LMLgrad_covar(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood with respect to the
        hyperparameters of the covariance function
        """
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_covar')
            return {'covar_r':SP.zeros(len(hyperparams['covar_r'])),'covar_c':SP.zeros(len(hyperparams['covar_c'])),'covar_r':SP.zeros(len(hyperparams['covar_r']))}
        except ValueError:
            LG.error('value error in _LMLgrad_covar')
            return {'covar_r':SP.zeros(len(hyperparams['covar_r'])),'covar_c':SP.zeros(len(hyperparams['covar_c'])),'covar_r':SP.zeros(len(hyperparams['covar_r']))}
 
        RV = {}
        Si = unravel(1./KV['S'],self.n,self.t)
    
        if 'covar_r' in hyperparams:
            theta = SP.zeros(len(hyperparams['covar_r']))
            for i in range(len(theta)):
                Kgrad_r = self.covar_r.Kgrad_theta(hyperparams['covar_r'],i)
                d = (KV['U_r']*SP.dot(Kgrad_r,KV['U_r'])).sum(0)                                
                LMLgrad_det = SP.dot(d,SP.dot(Si,KV['S_c_tilde']))                
                UdKU = SP.dot(KV['U_r'].T,SP.dot(Kgrad_r,KV['U_r']))
                SYUdKU = SP.dot(UdKU,(KV['Ytilde']*SP.tile(KV['S_c_tilde'][SP.newaxis,:],(self.n,1))))                
                #SYUdKU = SP.dot(UdKU,SP.dot(KV['Ytilde'],KV['S_c_tilde']))
                LMLgrad_quad = - (KV['Ytilde']*SYUdKU).sum()
                LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
                theta[i] = LMLgrad          
            RV['covar_r'] = theta

        if 'covar_c' in hyperparams:
            theta = SP.zeros(len(hyperparams['covar_c']))
            for i in range(len(theta)):
                Kgrad_c = self.covar_c.Kgrad_theta(hyperparams['covar_c'],i)
                #d = (KV['U_c']*SP.dot(Kgrad_c,KV['U_c'])).sum(0)
                S_c_tilde_grad = SP.dot(KV['UBinvB'],SP.dot(Kgrad_c, KV['UBinvB'].T))
                
                LMLgrad_det = SP.dot(KV['S_r'],SP.dot(Si, SP.diag(S_c_tilde_grad)))
                
                SYUdKU = SP.dot((KV['Ytilde']*SP.tile(KV['S_r'][:,SP.newaxis],(1,self.t))),S_c_tilde_grad.T)
                LMLgrad_quad = -SP.sum(KV['Ytilde']*SYUdKU)
                LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
                theta[i] = LMLgrad
            RV['covar_c'] = theta

        return RV

    def _LMLgrad_lik(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood with respect to
        the hyperparameters of the likelihood function
        """
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LML_covar')
            return {'lik':SP.zeros(len(hyperparams['lik']))}
        except ValueError:
            LG.error('value error in _LML_covar')
            return {'lik':SP.zeros(len(hyperparams['lik']))}
        
        YtildeVec = ravel(KV['Ytilde'])
        Kd_diag = self.likelihood.Kdiag_grad_theta(hyperparams['lik'],self.n,0)

        temp = SP.kron(SP.dot(KV['U_c'].T, SP.dot(KV['Binv'], SP.dot(KV['Binv'].T, KV['U_c']))),SP.diag(Kd_diag)) 
        LMLgrad_det = SP.diag(SP.dot(SP.diag(1./KV['S']),temp)).sum() # Needs more optimization
        
        sigma_grad = 2 * SP.exp(2 * hyperparams['lik'])
        LMLgrad_quad = - (sigma_grad * YtildeVec * 
            ravel(SP.dot(SP.eye(self.n), SP.dot(KV['Ytilde'],KV['UBinvBinvU'].T)))).sum()
        LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
        
        return {'lik':SP.array([LMLgrad])}

    def _LMLgrad_x(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood with respect to
        the latent factors
        """
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LML_covar')
            RV = {}
            if 'X_r' in hyperparams:
                RV['X_r'] = SP.zeros(hyperparams['X_r'].shape)
            if 'X_c' in hyperparams:
                RV['X_c'] = SP.zeros(hyperparams['X_c'].shape)
            return RV
        except ValueError:
            LG.error('value error in _LML_covar')
            RV = {}
            if 'X_r' in hyperparams:
                RV['X_r'] = SP.zeros(hyperparams['X_r'].shape)
            if 'X_c' in hyperparams:
                RV['X_c'] = SP.zeros(hyperparams['X_c'].shape)
            return RV
       
        RV = {}
        if 'X_r' in hyperparams:
            LMLgrad = SP.zeros((self.n,self.covar_r.n_dimensions))
            LMLgrad_det = SP.zeros((self.n,self.covar_r.n_dimensions))
            LMLgrad_quad = SP.zeros((self.n,self.covar_r.n_dimensions))

            SS = SP.dot(unravel(1./KV['S'],self.n,self.t),KV['S_c_tilde'])
            UY = SP.dot(KV['U_r'],KV['Ytilde'])
            UYSYU = SP.dot(UY,SP.dot(SP.diag(KV['S_c_tilde']),UY.T))
            for d in xrange(self.covar_r.n_dimensions):
                Kd_grad = self.covar_r.Kgrad_x(hyperparams['covar_r'],d)
                # calculate gradient of logdet
                URU = SP.dot(Kd_grad.T,KV['U_r'])*KV['U_r']
                LMLgrad_det[:,d] = 2*SP.dot(URU,SS.T)
                # calculate gradient of squared form
                LMLgrad_quad[:,d] = -2*(UYSYU*Kd_grad).sum(0)
            LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
            RV['X_r'] = LMLgrad
            
        if 'X_c' in hyperparams:
            LMLgrad = SP.zeros((self.t,self.covar_c.n_dimensions))
            LMLgrad_quad = SP.zeros((self.t,self.covar_c.n_dimensions))
            LMLgrad_det = SP.zeros((self.t,self.covar_c.n_dimensions))

            SS = SP.dot(KV['S_r'],unravel(1./KV['S'],self.n,self.t))
            UY = SP.dot(KV['UBinvB'],KV['Ytilde'].T)
            UYSYU = SP.dot(UY,SP.dot(SP.diag(KV['S_r']),UY.T))
            for d in xrange(self.covar_c.n_dimensions):
                Kd_grad = self.covar_c.Kgrad_x(hyperparams['covar_c'],d)
                # calculate gradient of logdet
                UCU = SP.dot(Kd_grad.T,KV['UBinvB'])*KV['UBinvB']
                LMLgrad_det[:,d] = 2*SP.dot(SS,UCU.T)
                # calculate gradient of squared form
                LMLgrad_quad[:,d] = -2*(UYSYU*Kd_grad).sum(0)
                
            LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
            RV['X_c'] = LMLgrad
    
        return RV
