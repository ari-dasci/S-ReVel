from typing import Tuple
from .perturbation import Perturbation
from .utils import get_input_transform
import numpy as np
import torch
from skimage.segmentation import quickshift

class QuickshiftPerturbation(Perturbation):
    '''
    Perturbation that perturbs the input image by quickshift unsupervised segmentation.
    
    Parameters
    -----------
    
    kernel_size: float
        size of the kernel used to compute the segmentation with quickshift
    max_dist: float
        maximum distance between points in the segmentation
    ratio: float
        ratio of the segmentation
    '''
    def __init__(self,kernel_size:float=4,max_dist:float=200.0,ratio:float=0.2,**kwargs):
        self.kernel_size = kernel_size
        self.max_dist = max_dist
        self.ratio = ratio
        super().__init__(segmentation_fn     =   self.quick_shift_segmentation,
                         fn_neutral_image    =   self.neutral,
                         get_input_transform =   get_input_transform,
                         **kwargs)
    def quick_shift_segmentation(self,img):
        '''
        Quickshift segmentation function. First introduced on :cite:`quickshift`. It is a wrapper 
        of the skimage.segmentation.quickshift function.
        
        Parameters
        -----------
        img:
            original image. Dims (H,W,C)
            
        Returns
        --------
        The segmentation of the image. Dims (H,W)
        '''
        img = np.array(img)
        if img.shape[-1] != 3:
            raise ValueError("The original image must have 3 on the last dimension")
        
            
        
        segments = quickshift(img, kernel_size=self.kernel_size, max_dist=self.max_dist, ratio=self.ratio)
        
        return segments
    def neutral(self,image):
        '''
        This function returns the neutral image. The neutral image
        is an image with all pixels set to the mean value of the image for
        each channel.
        Parameters
        -----------
        image:
            original image. Dims (H,W,3)
            
        Returns
        --------
        The neutral image with same shape as the original image with the mean value of each channel.
        '''
        #check if original image has 3 on the last dimension
        if np.array(image).shape[-1] != 3:
            raise ValueError("The original image must have 3 on the last dimension")
        if isinstance(image,torch.Tensor):
            image = image.numpy()
            
        return np.zeros(image.shape) + np.mean(image)