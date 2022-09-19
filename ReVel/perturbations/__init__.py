'''
Perturbations
+++++++++++++

Module for the perturbation of instances to generate neighbors. 

'''
from typing import Tuple
from .squares import SquarePerturbation
from .quickshift import QuickshiftPerturbation
from .perturbation import Perturbation

def get_perturbation(name:str,**kwargs):
    if name == 'square':
        return SquarePerturbation(**kwargs)
    elif name == 'quickshift':
        return QuickshiftPerturbation(**kwargs)                              
    else:
        raise ValueError('Perturbation {} not found'.format(name))
