from .CIFAR10 import get_CIFAR10_data
from .CIFAR100 import get_CIFAR100_data
from .EMNIST import get_EMNIST_data
from .FashionMNIST import get_FashionMNIST_data

def load_data(dataset:str="CIFAR10",perturbation = None,train=True,dir="./"):
    if dir is None:
        dir = "./"
    if dataset == "CIFAR10":
        return get_CIFAR10_data(perturbation= perturbation,train=train,dir=dir)
    elif dataset == "CIFAR100":
        return get_CIFAR100_data(perturbation= perturbation,train=train,dir=dir)  
    elif dataset == "EMNIST":
        return get_EMNIST_data(perturbation= perturbation,train=train,dir=dir)
    elif dataset == "FashionMNIST":
        return get_FashionMNIST_data(perturbation= perturbation,train=train,dir=dir)
    else:
        raise Exception("Dataset not supported")