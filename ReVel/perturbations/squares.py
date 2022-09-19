from typing import Tuple
from ELBAF.perturbations.perturbation import Perturbation
from ELBAF.perturbations.utils import get_input_transform
import numpy as np
import torch

class SquarePerturbation(Perturbation):
    '''
    Perturbation that perturbs the input image by a square segmentation. This class is used to perturb the input image 
    
    Parameters
    -----------
    
    dim: int
        dimension of the square segmentation in which the image is segmented.
        
    '''
    def __init__(self,dim:int,**kwargs):
        self.dim = dim
        if kwargs.get('get_input_transform',None) is None:
            kwargs['get_input_transform'] = get_input_transform
        super().__init__(segmentation_fn=self.square_segmentation,
                         fn_neutral_image=self.neutral,
                         **kwargs)
        
    def square_segmentation(self,img):
        '''
        This function returns the segmentation of the image divided on squares of size (H/self.dim,W/self.dim)
        
        Parameters
        -----------
        
        img: 
            original image. Dims (H,W,C)
        
        Returns
        --------
        
        The segmentation of the image divided on squares of size (H/self.dim,W/self.dim). Each square
        of the matrix is a diferent integer.
        
        '''
        img = np.array(img)
        rango = img.shape[0]//self.dim
        segments = np.zeros(shape=img.shape[0:2])
        for i in range(self.dim+1):
            for j in range(self.dim+1):
                segments[i*rango:(i+1)*rango,j*rango:(j+1)*rango] = i+j*self.dim
        
        return segments
    
    def neutral(self,image):
        '''
        This function returns the neutral image. The neutral image
        is an image with all pixels set to the mean value of the image for
        each channel.
        
        Parameters
        -----------
        
        image:
            original image. Dims (H,W,C)
        
        Returns
        --------
        
        The neutral image with same shape as the original image with the mean value of each channel.
        '''
        if isinstance(image,torch.Tensor):
            image = image.numpy()
            
        return np.zeros(image.shape) + np.mean(image)
        