import numpy as np
import scipy
from sklearn.metrics import r2_score
import matplotlib.cm as cm
import pandas as pd
from sklearn.linear_model import LinearRegression

from ReVel.LLEs import LLE

class ReVel:
    '''
        Base module for the ReVel framework.
        
        The framework is composed of five different metrics and some auxiliary functions, such as the
        derivation of the softmax function and the visualization of the results.
        
        The metrics are:
        
        .. autosummary::
            ReVel.revel.ReVel.local_concordance
            ReVel.revel.ReVel.local_fidelity
            ReVel.revel.ReVel.prescriptivity
            ReVel.revel.ReVel.conciseness
            ReVel.revel.ReVel.robustness
        
        This module has the following parameters:
        
        Parameters
        ----------
        model_f:
            Black-box function :math:`B:\mathcal{X} -> \mathcal{Y}` from the initial space
            :math:`\mathcal{X}` to the target space :math:`\mathcal{Y}`.
        model_g: 
            Black-box function :math:`B_{\mathcal{F}}:\mathcal{F} -> \mathcal{Y}` from the feature space
            :math:`\mathcal{F}` to the target space :math:`\mathcal{Y}`.
        instance: np.ndarray
            Instance :math:`x \in X` to be explained.
        lle: LLE
            LLE explanation generator to be used. lle returns a linear regressor :math:`Af+b` 
            Theoretically, lle is a linear function :math:`L:\mathcal{F} -> \mathcal{Y}, L(f) = Af+b, f \in \mathcal{F}`.
            with :math:`A` the coefficients and :math:`b` the intercept.
        n_classes: int,
            Number of classes in the instance.
        segments: np.ndarray,
            Segmentation of the instance.
        '''
        
    def __init__(self,model_f,model_g,instance:np.ndarray,lle:LLE,n_classes:int,segments:np.ndarray):
        
        self.model_g = model_g
        self.model_f = model_f
        self.LLE = lle
        self.instance = instance
        self.n_classes = n_classes
        self.segments = segments
        self.n_features = len(np.unique(segments))
    
    def train_on_sample(self) -> LinearRegression :
        '''
        Train the LLE model on the instance. It is a wrapper of the LLE train method.
        
        Return
        ------
        sklearn.linear_model.LinearRegression
            Explanation proposed by LLE method for the instance.
        
        '''
        return self.LLE(model_forward=self.model_f,instance=self.instance)
    
    def softmax_derivative(self,lle_representation:LinearRegression)->np.ndarray:
        '''
        Calculate the derivative of the softmax function with respect to the logits. To calculate
        this derivative, we use the chain rule. Let :math:`L(f) = Af+x` be the LLE model and
        :math:`S=softmax(L(f))` the softmax evaluation. Then, the derivative is given by
        
        :math:`D(L(f)) = (S-I)·S^t`
        
        Parameters
        ----------
        lle_representation: LLE
            LLE on which to calculate the softmax derivative.
            
        Returns
        -------
        np.ndarray 
            Derivative of the softmax function with respect to the logits. Dims (C,C).
        '''
        coeficients = np.copy(lle_representation.coef_) # dim :Features(F)*Classes(C)
        bias = np.copy(lle_representation.intercept_) # dim : C
        
        # Predicción de nuestro modelo interpretable en logits
        # values es un vector de dimension C (nº clases)
        
        values = bias + np.sum(coeficients,axis=1) # dim: C
        # Calculo de softmax y su derivada

        S = scipy.special.softmax(values) # dim: C
        
        C = len(S)
        S = np.expand_dims(S,axis=0)
        S = np.repeat(S,C,axis=0) # dim: C*C

        derivative = (np.eye(C)-S)*(np.transpose(S))
        

        return derivative
    
    def importance_matrix(self,lle_representation:LinearRegression)->np.ndarray:
        '''
        Computation of the importance matrix :math:`\mathcal{A}` of the LLE model.
        It is a matrix of dimensions FxC where :math:`a_{i,j}` is the importance of the feature i to the class j.
        
        Let :math:`A_l` be :math:`A` from the LLE model and :math:`A_p = \dfrac{ \delta }{ \delta X}(softmax(L(X)))` the
        derivative of the softmax function with respect to the feature space. Then, the importance matrix is given by
        :math:`\mathcal{A} = sign(A_l)·\sqrt{|A_p|·|A_l|}`
        
        Parameters
        ----------
        lle_representation: LLE
            LLE on which to calculate the importance matrix.
        
        Returns
        -------
        np.ndarray  
            Importance matrix :math:`\mathcal{A}` of the LLE model. Dims (F,C).
        
        '''
        importance_log = np.copy(lle_representation.coef_)
        
        derivative = self.softmax_derivative(lle_representation) # dim : C*C

        importance_prob = np.matmul(derivative,importance_log) # C*F
        
        importance = np.sign(importance_log)*np.sqrt(np.abs(importance_log)*np.abs(importance_prob))

        return importance
    def feature_importance(self,lle_representation:LinearRegression,order=2)->np.ndarray:
        '''
        Calculate the feature importance of the LLE model. For each features, the importance 
        is the norm of the vector of the importance matrix of this specific feature.

        Parameters
        ----------
        lle_representation: LLE
            LLE on which to calculate the feature importance.
        order: int,
            Order of the norm used to calculate the feature importance.
        Returns
        -------
        np.ndarray
            Feature importances of the LLE model V with dimension F. Dims (F,).
        '''
        importance = self.importance_matrix(lle_representation)
        importance = self.importance_matrix(lle_representation)
        max_abs = np.max(np.abs(importance))
        importance = np.abs(importance/max_abs)
        features = np.swapaxes(importance,0,1)
        return features
        
    def local_concordance(self,lle_representation:LinearRegression,order=1)->np.ndarray:
        '''
        Calculate the local concordance of the LLE model. 
        The local concordance is calculated as the distance between the output predicted
        by the black-box model and the output predicted by the LLE model on the instance itself.
        This distance is normalized by the maximun distance between the two possible probabilities vectors 
        so the local concordance is between 0 and 1. That is
        
        :math:`LocalConcordance(X) = \dfrac{|L(X) - B(X)|_{order}}{\max(|U-V|_{order})}`,
        
        where :math:`U` and :math:`V` are two vectors of probabilities where the maximundistance is achived.
        In practive, :math:`U` and :math:`V` are (1,0,0,...) and (0,1,0,...) respectively.
        
        Parameters
        ----------
        lle_representation: LLE
            LLE on which to calculate the local concordance.
        order: int,
            Order of the norm used to calculate the local concordance.
        
        Returns
        -------
        np.ndarray
            Local concordance of the LLE model.
        '''
        
        X = np.ones(shape=self.n_features)
        u = scipy.special.softmax(lle_representation.predict(np.array([X])))
        v = scipy.special.softmax(self.model_g(X).detach().cpu().numpy())
        
        #print(u[0,np.argsort(u[0])[-10:]],v[0,np.argsort(u[0])[-10:]])
        #print(u[0,np.argsort(v[0])[-10:]],v[0,np.argsort(v[0])[-10:]])
        a = np.array([1 if i == 0 else 0 for i in range(len(u))])
        b = np.array([1 if i == 1 else 0 for i in range(len(u))])
        dist = np.linalg.norm(u-v,ord=order)
        max_dist = np.linalg.norm(a-b,ord=order)
        return 1-(dist/max_dist)

    def local_fidelity(self,neighbourhood,lle_representation:LinearRegression,order = 2) ->np.ndarray:
        
        '''
        Calculate the local fidelity of the LLE model. It is calculated as the distance between 
        the output predicted by the black-box model and the output predicted by the LLE model 
        on the neighbourhood of the instance. Actually, the local fidelity is the generalization 
        of the local concordance over a neighbourhood of the instance and it is calculated as
        the mean of the local concordance of each neigbour of the neighbourhood.
        
        Parameters
        ----------
        neighbourhood: list of int
            List of the indices of the neighbours of the instance.
        lle_representation: LLE
            LLE on which to calculate the local fidelity.
        order: int,
            Order of the norm used to calculate the local fidelity.
        
        Returns
        -------
        np.ndarray
            Local fidelity of the LLE model.

        '''
        distances = []
        for neighbour in neighbourhood:
            BB = scipy.special.softmax(self.model_g(neighbour).detach().cpu().numpy()[0])
            WB = scipy.special.softmax(lle_representation.predict(np.array([neighbour]))[0])
            distances.append(np.linalg.norm(BB-WB,ord=order))

        a = np.array([1 if i == 0 else 0 for i in range(len(BB))])
        b = np.array([1 if i == 1 else 0 for i in range(len(BB))])
        max_dist = np.linalg.norm(a-b,ord=order)
        return np.mean([1-dist/max_dist for dist in distances])
    
    
    def prescriptivity(self,lle_representation:LinearRegression,order=2) -> np.ndarray:

        '''
        Calculate the prescriptivity of the LLE model. It is calculated as the distance between
        the output predicted by the black-box model and the output predicted by the LLE model
        on the neighbour :math:`X_{h}` of the intance :math:`X` with the minimum distance but with 
        different class of the instance itself. This distance is normalized by the maximun distance
        between the two possible probabilities vectors so the prescriptivity is between 0 and 1.
        
        :math:`X_h` is computed by the following procedure
        
        .. highlight:: python
        .. code-block:: python
        
            X_h = X
            while L(X_h) has the same class as L(X);
                X_h discard the most positive important feature for class
            return X_h
        
        :math:`X_h` is the closest neighbour of :math:`X` which has not the same class as :math:`X`.
        As L is linear, the distance between L(X_h) and L(X) is constantly increasing.
        
        Parameters
        ----------
        lle_representation: LLE
            LLE on which to calculate the prescriptivity.

        order: int,
            Order of the norm used to calculate the prescriptivity.
        
        Returns
        -------
        np.ndarray
            Prescriptivity of the LLE model.
        '''
        importance = self.importance_matrix(lle_representation=lle_representation)
        X = np.array([1 for _ in range(importance.shape[1])])
        prediction = self.model_g(X).cpu().detach().numpy()[0]
        clase = np.argmax(prediction)
        importance = importance[clase]
        importance_indexes = np.argsort(importance)[::-1]
        j = 0
        prediction_minus = prediction
        while clase == np.argmax(prediction_minus) and j < len(importance_indexes):
            X[importance_indexes[j]] = 0
            j+=1
            prediction_minus = lle_representation.predict(np.array([X]))[0]
        u = scipy.special.softmax(self.model_g(X).detach().cpu().numpy())
        v = scipy.special.softmax(lle_representation.predict(np.array([X])))
        a = np.array([1 if i == 0 else 0 for i in range(len(X))])
        b = np.array([1 if i == 1 else 0 for i in range(len(X))])

        distance = np.linalg.norm(u-v,ord=order)
        max_distance =  np.linalg.norm(a-b,ord=order)
        return 1-distance/max_distance
    
    def conciseness(self,lle_representation:LinearRegression,order=2)->np.ndarray:
        '''
        Calculate the conciseness of the LLE model. 
        
        Let :math:`\mathcal{A}` be the importance matrix with dimensions FxC.
        We define the feature importance of feature :math:`i` as the norm of the vector :math:`\mathcal{A}_{i}`, that is,
        :math:`I_i = \|\mathcal{A}_{i}\|_{order}`. We normalize those norms by the maximum norm :math:`I_i`,that is,
        :math:`I_{i}^{norm} = \dfrac{I_i}{I_{max}}`. Then, the conciseness is calculated as
        
        :math:`Conciseness = \dfrac{N-\sum_{i=1}^{C}I_{i}^{norm}}{N-1}`

                
        Parameters
        ----------
        lle_representation: LLE
            LLE on which to calculate the conciseness.
        order: int,
            Order of the norm used to calculate the conciseness.

        Returns
        -------
        np.ndarray
            Conciseness of the LLE model.
        '''
        
        features = self.feature_importance(lle_representation,order)

        
        normas = np.array([np.linalg.norm(features[i],ord=order) for i in range(len(features))])
        normas = normas/np.max(normas)
        
        sum_normas = np.sum(normas)
        
        return (len(features)-sum_normas)/(len(features)-1)
    
    def robustness(self,lles)-> np.ndarray:
        '''
        Calculate the robustness of the LLE model. Let :math:`\mathcal{A}_i` be the importance matrix with dimensions FxC
        of the :math:`i`-th explanation given on *lles*. For two different explanations :math:`\mathcal{A}_i` and :math:`\mathcal{A}_j`,
        we define the similarity between the explanations as a combination of two similarities. The first similarity is the
        cosine similarity(:math:`sim_{cos}(\mathcal{A}_i,\mathcal{A}_j)`) between the importance matrices :math:`\mathcal{A}_i` and :math:`\mathcal{A}_j`. 
        The second similarity is calculated as
        
        :math:`sim_{norm}(  \mathcal{A}_i,\mathcal{A}_j) = 1-\dfrac{\|\mathcal{A}_i-\mathcal{A}_j\|_{order}}{max(\|\mathcal{A}_i\|_{order},\|\mathcal{A}_j\|_{order})}`
        
        The similarity is calculated as :math:`sim(\mathcal{A}_i,\mathcal{A}_j) = sim_{cos}(\mathcal{A}_i,\mathcal{A}_j) * sim_{norm}(\mathcal{A}_i,\mathcal{A}_j)`.
        
        The robustness of a LLE model is defined by the expected value of the similarity between two explanations of 
        the LLE model and it is aproximated by the mean of the similarities between all the explanations given in lles.
        
        Parameters
        ----------
        lles: list of LLE
            List of LLE on which to calculate the robustness.

        Returns
        -------
        np.ndarray
            Robustness of the LLE model.
        '''
        weights = []
        for lle in lles:
            importance = self.importance_matrix(lle_representation=lle)
            weights.append(np.copy(importance.flatten()))
        def d(u,v):
            similarity_factor = np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
            norm_factor = 1-np.abs(np.linalg.norm(v)-np.linalg.norm(u))/max(np.linalg.norm(v),np.linalg.norm(u))
            return similarity_factor*norm_factor
        ejemplos = [d(weights[i],weights[j])  for i in range(len(weights)) for j in range(i,len(weights))]
        return np.mean(ejemplos)
    
    def importance_mask(self, lle_representation:LinearRegression,segments,id_class)->   np.ndarray:
        '''
        Calculate the importance mask of the LLE model to a certain class over the segments. 
        That is, we build a mask with the same shape as segments where each element is the importance of
        the corresponding feature in the LLE model. We normalize the importance to be between 0 and 1 and a 0 valued importance
        to be 0,5.
        
        This function is used as a helper for the visualization functions, so it is not necessary to use it directly.
        
        Parameters
        ----------
        lle_representation: LLE
            LLE on which to calculate the importance mask.
        segments: numpy.ndarray
            Segments over which to calculate the importance mask.
        id_class: int
            Class for which to calculate the importance mask.

        Returns
        -------
        np.ndarray
            Importance mask of the LLE model to the class id_class. Dims (H,W)
        '''
        imp_matrix = self.importance_matrix(lle_representation=lle_representation)
        max_abs = np.max(np.abs(imp_matrix))
        imp_matrix = imp_matrix/max_abs
        imp_matrix = (imp_matrix+1)/2 # C*F
        
        
        imp_vector_class = np.array([imp_matrix[id_class][i] for i in range(len(imp_matrix[id_class]))])
        mask = np.zeros(segments.shape)
        for f, val in enumerate(imp_vector_class):
            mask = np.where(segments==f,val,mask)
        
        return mask
    def coloured_importance_mask(self, lle_representation:LinearRegression,segments,id_class)->   np.ndarray:
        '''
        Transforms the importance mask of the LLE model to a certain class over the segments to a coloured mask.
        The colours are from the cmap "RdYlGn", where red is assigned to the importance of the feature with the highest 
        negative importance, green to the feature with the highest positive importance and yellow to the features with
        clossest importance to 0. 
        
        Parameters
        ----------
        lle_representation: LLE
            LLE on which to calculate the importance mask.
        segments: numpy.ndarray
            Segments over which to calculate the importance mask.
        id_class: int
            Class for which to calculate the importance mask.

        Returns
        -------
        np.ndarray
            Coloured importance mask of the LLE model to the class id_class. Dims (H,W,3)
        
        '''
        mask_j = self.importance_mask(lle_representation=lle_representation,segments=segments,id_class=id_class)
        colored = cm.get_cmap("RdYlGn")
        mask_j = colored(mask_j)
        mask_j = mask_j[:,:,:3]
        
        return (mask_j+self.instance/256.0)/2
    
    def evaluate(self,times:int = 5)-> pd.DataFrame:
        '''
        Evaluate the LLE method over a certain instance a certain number of *times*. For this purpose,
        we calculate *times* different LLE models and we calculate all the metrics for each of them. We also calculate
        the robustness of all the LLE explanations proposal.
        
        Parameters
        ----------
        times: int
            Number of times to evaluate the LLE method.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with the metrics of the LLE method.
        '''
        lles = []
        concis = []
        fidel = []
        concord = []
        prescript = []
        for i in range(times):
            lle_representation = self.train_on_sample()
            neigbour = [[0 if j==i else 1 for j in range(self.n_features) ] for i in range(self.n_features)]
            concis.append(self.conciseness(lle_representation))
            fidel.append(self.local_fidelity(neigbour,lle_representation))
            concord.append(self.local_concordance(lle_representation))
            prescript.append(self.prescriptivity(lle_representation))
            lles.append(lle_representation)
        
        # Guardar los resultados en un dataframe con 'times' filas
        # con los resultados de cada ejemplo
        robustness = self.robustness(lles)
        df = pd.DataFrame(columns=['conciseness','local_fidelity','local_concordance','prescriptivity','robustness'])
        df['conciseness'] = concis
        df['local_fidelity'] = fidel
        df['local_concordance'] = concord
        df['prescriptivity'] = prescript
        df['robustness'] = robustness
        
        return df
        