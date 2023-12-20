import torchvision
import torch
def get_Food101_data(perturbation, train=True, dir=None):
    if dir is None:
        dir = "./"
    testIMG = torchvision.transforms.Lambda(lambda x:perturbation.transform(x))
    testTarget = perturbation.target_transform
    root = f"{dir}/Food101/"
    if train:
        Dataset = torchvision.datasets.Food101(root=root+"train",
            split="train",
            download=True,
            transform=testIMG,
            target_transform=testTarget)
        
        
    else:
        Dataset = torchvision.datasets.Food101(root=root+"test",
            split="test",
            download=True,
            transform=testIMG,
            target_transform=testTarget)
    return Dataset