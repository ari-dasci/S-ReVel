from .CIFAR10 import get_CIFAR10_data
from .CIFAR100 import get_CIFAR100_data
from .EMNIST import get_EMNIST_data
from .FashionMNIST import get_FashionMNIST_data
from .StandfordCars import get_StandfordCars_data
from .Flowers import get_Flowers_data
from .FGVC import get_FGVC_data
from .OxfordIIITPet import get_OxforIIITPet_data
from .Food101 import get_Food101_data
from .Imagenet import get_Imagenet_data
import ssl

def load_data(dataset:str="CIFAR10",perturbation = None,train=True,dir="./"):
    
    ssl._create_default_https_context = ssl._create_unverified_context
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
    elif dataset == "StandfordCars":
        return get_StandfordCars_data(perturbation= perturbation,train=train,dir=dir)
    elif dataset == "Flowers":
        return get_Flowers_data(perturbation= perturbation,train=train,dir=dir)
    elif dataset == "FGVC":
        return get_FGVC_data(perturbation= perturbation,train=train,dir=dir)
    elif dataset == "OxfordIIITPet":
        return  get_OxforIIITPet_data(perturbation= perturbation,train=train,dir=dir)
    elif dataset == "Food101":
        return get_Food101_data(perturbation= perturbation,train=train,dir=dir)
    elif dataset == "ImageNet":
        return get_Imagenet_data(perturbation= perturbation,train=train,dir=dir)
    else:
        raise Exception("Dataset not supported")