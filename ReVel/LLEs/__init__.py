'''
Local Linear Explanation (LLE)
++++++++++++++++++++++++++++++

Local Linear Explanation (LLE) abstraction module. This module provides the abstract
class LLE and three concrete implementations: LIME, RANDOM and SHAP.

.. autosummary::
    ReVel.LLEs.LIME
    ReVel.LLEs.SHAP
    ReVel.LLEs.RANDOM
'''
from .LLE import LLE
from .lime import LIME
from .shap import SHAP
from .random import RANDOM


def get_xai_model(**kwargs):
    if kwargs.get('name') == 'LIME':
        return LIME(**kwargs)
    elif kwargs.get('name') == 'RANDOM':
        return RANDOM(**kwargs)
    elif kwargs.get('name') == 'SHAP':
        return SHAP(**kwargs)
    else:    
        raise ValueError('XAI model {} not found'.format(kwargs.get('name')))