Introduction
=================================

ReVEL(Robust Evaluation VEctorized Loca-linear-explanation) is a Python
library that provides series of tools for Explinable Artificial Intelligence(XAI) 
from the perspective of the audition of black-box. Specifically, it provides 
a framework for the generation and evaluation of Local Linear Explanations(LLEs).

Installation
------------

To install ReVEL we recomend to create an environment with conda as following

.. code-block:: bash

   conda create -n revel python=3.8
   conda activate revel

After that, we can install the library using the following commands

.. code-block:: bash

   git clone https://github.com/ari-dasci/ReVel.git
   cd ./ReVel
   pip install .

First example
-------------

After downloading the library, we can start using it. For example, on `the following
jupyter notebook`_ we generate a LLE for an image of the imagenet dataset and evaluate it.

.. _the following jupyter notebook: notebooks/fisrt-steps.ipynb











   
