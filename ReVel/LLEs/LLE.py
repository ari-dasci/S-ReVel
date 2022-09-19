import numpy as np
from ReVel.perturbations.perturbation import Perturbation
from sklearn.linear_model import LinearRegression

class LLE(object):
    '''
        Abstract class for Local Linear Explanations.
        
        A Local Linear explanation is defined by a regression model and
        a perturbation strategy of the input. This perturbation strategy defines
        the neighbourhood construction.
        
        Parameters
        ----------
        perturbation: Perturbation
            Perturbation class with the strategy used to construct the neighbourhood.
        max_examples: int
            Maximum number of examples to be used in the regression.
    '''
    def __init__(self,perturbation:Perturbation=None,max_examples:int = 1000,**kwargs):
        self.max_examples   = max_examples
        self.perturbation   = perturbation
        
    def generate_neighbour(self,n_features:int)->np.ndarray:
        '''
        Abstract method for generating a neighbour. 
        
        Parameters
        ----------
        n_features: int
            Number of features to be considered

        Returns
        -------
        np.array 
            A neighbour of the instance.
        '''
        raise NotImplementedError("Subclass must implement abstract method")
        
    def regression(self,instance,model_forward, segments, examples)->LinearRegression:
        '''
        Abstract method for regression. 
        
        Parameters
        ----------
        instance:
            Instance to be explained.
        model_forward:
            black-box model to explain the instance. It must accept an object
            similar to instance and return a prediction.
        segments:
            Segments on which the instance is divided into features. It must have 
            the same shape as instance except for the last dimension.
        examples:
            Examples to be used in the regression.
        
        Returns
        -------
        sklearn.linear_model.LinearRegression
            The explanation proposed by the method.
        
        '''
        raise NotImplementedError("Subclass must implement abstract method")
    def kernel(self,V)->np.ndarray:
        '''
        Abstract method for kernel. This function is used to weight the neighbour
        respect to the distance to the original instance.
        
        Parameters
        ----------
        V: float
            Distance to the original instance.

        Returns
        -------
        np.array
            Weight of the neighbour.
        '''
        raise NotImplementedError("Subclass must implement abstract method")
    
    def perturb(self,img,neutral,segments,indexes)->np.ndarray:
        '''
        Abstract method for perturbation. This function is used to perturb the
        original instance to generate a neighbour. It is a wrapper for the
        perturbation object which has the strategy to generate the neighbour.
        
        Parameters
        ----------
        img:
            Original instance.
        neutral:
            Neutral instance. 
        segments:
            Segments on which the instance is divided into features. It must have
            the same shape as instance except for the last dimension.
        indexes:
            Indexes of the features to be perturbed.
        
        Returns
        -------
        np.array
            The perturbed instance obtained by removing the features of the indexes 
            on the img and adding the features of the indexes on the neutral. Dims (H,W,C).
            
        
        '''
        return self.perturbation.perturbation(img,neutral,segments,indexes)

    def __call__(self,model_forward,instance)->LinearRegression:
        '''
        Abstract method for calling the LLE. This function is used to generate
        the LLE. This function propose the segmentation of the instance by the
        perturbation object.

        Parameters
        ----------
        model_forward:
            black-box model forward method. It must accept an object similar to
            instance and return a prediction.
        instance:
            Instance to be explained.

        Returns
        -------
        sklearn.linear_model.LinearRegression
            The explanation proposed by the LLE.
        '''
        
        if np.array(instance).shape[0] == 3:
            segments = self.perturbation.segmentation_fn(np.array(instance).swapaxes(0,1).swapaxes(1,2))
        else: 
            segments = self.perturbation.segmentation_fn(np.array(instance))
        
        
        return self.explain_instance(instance,model_forward,segments=segments)

    def explain_instance(self,instance,model_forward,segments)->LinearRegression:
        '''
        Method for explaining an instance. This function is used to generate
        the LLE.

        Parameters
        ----------
        model_forward:
            black-box model forward method. It must accept an object similar to
            instance and return a prediction.
        instance:
            Instance to be explained.
        segments:
            Segments on which the instance is divided into features. It must have
            the same shape as instance except for the last dimension.
        Returns
        -------
        sklearn.linear_model.LinearRegression
            The explanation proposed by the method.
        '''
        examples = []
        n_features = len(np.unique(segments))
        while len(examples) < self.max_examples:
            lista = self.generate_neighbour(n_features)
            inside = False
            for j, cluster in enumerate(examples):
                same_shape = cluster.shape[0] == lista.shape[0]
                if same_shape:
                    equal = np.all(cluster == lista)
                    if equal:
                        inside = True
                        break
            if not inside and len(lista) > 0:
                examples.append(lista)
        
        examples = np.array(examples,dtype=object)
            
        return self.regression(instance,model_forward,segments,examples)