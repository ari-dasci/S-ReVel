import torchvision
def get_Imagenet_data(perturbation, train=True, dir=None):
    if dir is None:
        dir = "./"
    testIMG = torchvision.transforms.Lambda(lambda x:perturbation.transform(x))
    testTarget = perturbation.target_transform
    root = f"{dir}/Imagenet/"
    if train:
        Dataset = torchvision.datasets.ImageNet(root=root+"train",
            split="train",
            transform=testIMG,
            target_transform=testTarget)
    else:
        Dataset = torchvision.datasets.ImageNet(root=root+"val",
            split="val",
            transform=testIMG,
            target_transform=testTarget)
    return Dataset
    