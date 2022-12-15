import cv2
import torch
import numpy as np


class Perturbation:
    '''
    Initialize the abstract Perturbation class. This class is used to perturb the input image 
    
    This class is used as a base class for the perturbations. It assumes that the needed 
    functions are implemented in the child class.
    
    Parameters
    ---------
    segmentation_fn: 
        function that returns the segmentation of an image
    fn_neutral_image: 
        function that returns the neutral image used to replace the
        original image when the perturbation is applied
    num_classes: 
        number of classes. It determines the target space
    final_size: 
        desired size of the final image
    '''
    
    def __init__(self,
            segmentation_fn,
            fn_neutral_image,
            num_classes:int,
            get_input_transform,
            final_size,**kwargs):
        
        self.segmentation_fn = segmentation_fn
        self.fn_neutral_image = fn_neutral_image
        self.after_transform = get_input_transform(final_size)
        self.num_classes = num_classes
        self.final_size = final_size

    def perturbation(self,img,neutral,segments,indexes):
        '''
        This function is used to perturb the input image. For this purpose, the function
        recives the original image, the neutral image and the segmentation of the image.
        We change the segmentation indexes to avoid the original image and replace them with the neutral image.
        
        Parameters
        ----------
        img: 
            original image. Dims (H,W,3)
        neutral: 
            neutral image. Dim (H,W,3)
        segments:
            segmentation of the image. Dims (H,W)
        indexes:
            indexes of the image that we want to replace with the neutral image. Integer or list of integers
        
        Returns
        -------
        Perturbed image. Dim (H,W,C)
        '''
        #check if original image and the neutral image has 3 on the last dimension
        if np.array(img).shape[-1] != 3 or np.array(neutral).shape[-1] != 3:
            #if not, raise an error
            raise ValueError("The original image and the neutral image must have 3 on the last dimension")
        
        if isinstance(indexes,np.ndarray) or isinstance(indexes,list) or isinstance(indexes,tuple):
            conditions = np.array([segments == indx for indx in indexes],dtype=object)
            condition = np.any(conditions,axis=0)
        else:
            condition = (segments == indexes )
        
        if len(np.array(img).shape) == 3:
            condition = np.expand_dims(condition,-1)
            condition = np.repeat(condition,3,axis=-1)
        
        return np.where(condition,neutral,img )

    def transform(self,img):
        '''
        This function is used to transform the input image to the input space.
        
        Parameters
        ----------
        img:    
            original image. Dims (H,W,C) or (H,W)
        Returns
        -------
        Image preprocessed. Dims (H,W,C)
        '''
        if len(np.array(img).shape) == 2:
            img = np.expand_dims(img,-1)
            img = np.repeat(img,3,axis=-1)
        
        segments = self.segmentation_fn(img)
        
        neutral = self.fn_neutral_image(np.array(img))
        
        perturbation = self.perturbation(img,neutral,segments,-1)
        return self.after_transform(perturbation)
    
    def target_transform(self,target:int):
        '''
        This function is used to transform the target to the target space.
        Usually, it depends on the dataset if we want a one-hot vector or a
        categorical vector. In this case, we transform the target, coded as an int,
        to a one-hot vector of size self.num_classes.
        
        Parameters
        ----------
        target:
            target of the image. Integer
        Returns
        -------
        Target in the target space. Dims (self.num_classes)
        
        '''
        t = torch.zeros(self.num_classes)
        t[target] = 1
        target = t
        return target
        
    def __call__(self, img,target):
        '''
        This function is used to transform the input image and the target to the input space.
        
        Parameters
        ----------
        img:
            original image to transform. Dims (H,W,C)
        target:
            target of the image. Integer
        Returns
        -------
        Image and target preprocessed. Dims ((H,W,C), (self.num_classes))
        
        '''
        t = torch.zeros(self.num_classes)
        t[target] = 1
        return img,t