from ReVel.LLEs.LLE import LLE
import numpy as np
from sklearn.linear_model import Ridge
import scipy
import tqdm


class SHAP(LLE):
    '''
    SHAP is a Local Linear Explanation method, proposed in :cite:`shap`. We
    also develop a variation of the SHAP method, called Loacal-SHAP, where we don't
    consider the neighbours with less than half the features.
    
    Parameters
    ----------
    sigma : float
        The sigma parameter define if the method is local or global. If sigma is
        greater than 0.5, the method is local. If sigma is less than 0.5, the
        method is global. The default value is 0.5.
    
    
    '''
    def __init__(self,sigma=0.5, **kwargs):
        self.local = (sigma > 0.5)
        super().__init__(**kwargs)
    
    def generate_neighbour(self,n_features:int)->np.ndarray:
        '''
        Generate a neighbour of the instance. The neighbour generation is done over
        the feature space. The generation of the neighbour is done randomly but each neighbour is
        generated with a probability proportional to their weight on the SHAP regressor.
        
        If the method is global, the neighbour is generated with 50% probability to have less than half the features
        and 50% probability to have more than half the features. If the method is local, the neighbour is generated
        with more than half the features.

        
        Parameters
        ----------
        n_features: int
            Number of features to be considered. It is needed to generate the neighbour of size n_features.
        
        Returns
        -------
        list
            A neighbour of the instance of size n_features.
        '''
        w = np.random.random()
        w = max(w,(4*n_features+4)/(n_features*n_features)+0.00001)
        
        if not self.local:
            
            if np.random.random() < 0.5:
                choosed = int((n_features - np.sqrt(n_features*n_features - 4*(n_features-1)/w))/2)
            else:
                choosed = int((n_features + np.sqrt(n_features*n_features - 4*(n_features-1)/w))/2)
        else:
            choosed = int((n_features - np.sqrt(n_features*n_features - 4*(n_features-1)/w))/2)
        lista = np.array([i for i in range(n_features)])
        np.random.shuffle(lista)
        lista = lista[:choosed]
        return lista
    
    def kernel(self, V,n_features:int)->np.ndarray:
        r'''
        The kernel function is used to weight the examples. For each example V,
        the asociated weight is computed as
        
        :math:`w = \dfrac{N-1}{|k|(N-\sum_{i=1}^{N}V_i){N\choose{|k|}}}`
        
        where :math:`|k|=\sum_{i=1}^{N}V_i`, that is, the number of features in the example.
        
        where :math:`N` is the number of features. Note that V is a vector of size with 0 and 1 values so
        the sum of V is the number of features that are not ocluded.
        
        Parameters
        ----------
        V: numpy.ndarray
            Vector of size N with 0 and 1 values.
        n_features: int
            Number of features.
        
        Returns
        -------
        float
            Weight associated to the feature vetor V.
        '''
        w = (n_features-1)/((n_features-np.sum(V,axis=-1))*np.sum(V,axis=-1))
        Mchoosez = scipy.special.comb(n_features,np.sum(V,axis=-1),exact=False)
        w = w / Mchoosez
        return w
    
    
    
    def regression(self, instance, model_forward, segments, examples)->Ridge:
        '''
        Regression method for the SHAP method. It is a ridge regression with the
        weights computed using the kernel function.
        
        Parameters
        ----------
        instance: numpy.ndarray
            Instance to be explained.
        model_forward: 
            Black-box model to explain the instance. It must accept an object
            similar to instance and return a prediction.
        segments: numpy.ndarray
            Segments of the instance.
        examples: numpy.ndarray
            Examples over the feature space over which the regression is performed.
            Those examples must be the neighbours over the feature space of the instance.
        
        Returns
        -------
        sklearn.linear_model.Ridge
            The explanation proposed by the method.
        '''
        logits = []
        n_pixels = []
        n_features = len(np.unique(segments))

        if len(examples) > self.max_examples:
            examples = np.random.choice(examples,self.max_examples,replace=False)
        
        examples,weights = np.unique(np.array([tuple(e) for e in examples ],dtype=object),return_counts=True)
        
        for example in tqdm.tqdm(examples):
            perturbacion = self.perturbation.perturbation(instance,self.perturbation.fn_neutral_image(instance),segments,example)
            logit = model_forward(perturbacion).to("cpu").detach().numpy()[0]
            logits.append(np.copy(logit))

            oclusion_pixels = np.any([i == segments for i in example],axis=0)
            n_pixels.append(np.copy(oclusion_pixels))

        X_Vector = np.array([[0.0 if k in examples[i] else 1.0 for k in range(n_features)] for i in range(len(examples))])
        
        Y_logits = logits

        regressor_logits = Ridge(fit_intercept=True)
        X_Vector,Y_logits = np.array(X_Vector),np.array(Y_logits)
        
        regressor_logits.fit(X_Vector,Y_logits,sample_weight= self.kernel(X_Vector,n_features=n_features))



        return regressor_logits