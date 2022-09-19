import torchvision
import cv2
import numpy as np

def get_input_transform(final_size=(32,32)):     
    
    def swap_Array(img):
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img,-1)
            img = np.repeat(img,3,-1)
            
        img = np.swapaxes(img,-1,-2)
        img = np.swapaxes(img,-2,-3)
        
        return img
    def resize(img):
        img = cv2.resize(img,final_size,interpolation=cv2.INTER_CUBIC)
        return img
    
    transf = torchvision.transforms.Compose([resize,
        swap_Array
    ])
    return transf