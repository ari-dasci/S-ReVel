from ReVel.LLEs.LLE import LLE
import numpy as np
import tqdm
from sklearn.linear_model import Ridge,LinearRegression

class LIME(LLE):
    '''
    LIME is a Local Linear Explanation method, proposed in :cite:`lime`. 
    
    Parameters
    ----------
    sigma: int
        Sigma parameter of the kernel. It controls the size of the neighbourhood in
        terms of how weighted the examples are.
    '''
    def __init__(self,sigma:int=3,**kwargs):
        
        self.sigma = sigma
        super().__init__(**kwargs)
        
    def generate_neighbour(self,n_features:int)->np.ndarray:
        '''
        Generate a neighbour of the instance. The neighbour generation is done over
        the feature space. The generation of the neighbour is done randomly but each neighbour is
        generated with a probability proportional to their weight on the LIME regressor.
        
        Parameters
        ----------
        n_features: int
            Number of features to be considered. It is needed to generate the neighbour of size n_features.
        
        Returns
        -------
        np.array
            A neighbour of the instance of size n_features. 
        
        '''
        p = np.random.random()
        w = int(self.sigma*np.sqrt(-np.log(p)))
        
        lista = np.array([i for i in range(n_features)])
        np.random.shuffle(lista)
        lista = lista[:min(w,len(lista))]
        
        return lista
    def kernel(self, V)->np.ndarray:
        '''
        Kernel function for the LIME regressor. It is calculated by the following formula
        
        :math:`K(V) = e^{-\dfrac{||1_{N}-V||^2}{ \sigma^2} }`
        
        where :math:`1_{N}` is the vector of ones of size N and :math:`d` is the vector of the vector of
        features.
        
        Parameters
        ----------
        V: np.array
            Vector of features of the neighbour. The :math:`V_{i}` must be 1 if the feature is 
            present in the neighbour and 0 if it is not.

        Returns
        -------
        np.array
            Weight of the neighbour.
        '''
        
        distance = np.linalg.norm((1-V),axis=-1)/self.sigma
        
        return np.exp(distance)
    
    def regression(self, instance, model_forward, segments, examples)->Ridge:
        '''
        Regression method for the LIME method. It is a ridge regression with the kernel function
        defined by LIME method.
        
        Parameters
        ----------
        instance:
            Instance to be explained.
        model_forward:
            Black-box model to explain the instance. It must accept an object
            similar to instance and return a prediction.
        segments:
            Segments on which the instance is divided into features. It must have
            the same shape as instance except for the last dimension.
        examples:
            Examples to be used in the regression. They must be generated in the feature space.

        Returns
        -------
        sklearn.linear_model.Ridge
            The explanation proposed by the method
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
        
        #indices = [(i,j) for i in range(len(logits)) for j in range(i+1,len(logits))]

        X_Vector = np.array([[0.0 if k in examples[i] else 1.0 for k in range(n_features)] for i in range(len(examples))])
        
        Y_logits = logits
        
        regressor_logits = Ridge(fit_intercept=True)
        X_Vector,Y_logits = np.array(X_Vector),np.array(Y_logits)

        regressor_logits.fit(X_Vector,Y_logits,sample_weight= self.kernel(X_Vector))
        
        return regressor_logits