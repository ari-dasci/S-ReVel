import torchvision
import torch
def get_Flowers_data(perturbation, train=True, dir=None):
    if dir is None:
        dir = "./"
    testIMG = torchvision.transforms.Lambda(lambda x:perturbation.transform(x))
    testTarget = perturbation.target_transform
    root = f"{dir}/Flowers/"
    if train:
        Train = torchvision.datasets.Flowers102(root=root+"train",
            split="train",
            download=True,
            transform=testIMG,
            target_transform=testTarget)
        Val = torchvision.datasets.Flowers102(root=root+"val",
            split="val",
            download=True,
            transform=testIMG,
            target_transform=testTarget)
        Dataset = torch.utils.data.ConcatDataset([Train, Val])
    else:
        Dataset = torchvision.datasets.Flowers102(root=root+"test",
            split="test",
            download=True,
            transform=testIMG,
            target_transform=testTarget)
    return Dataset
    