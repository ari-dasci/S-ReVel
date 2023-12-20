import torchvision
def get_CIFAR100_data(perturbation, train=True, dir="./data"):
    testIMG = torchvision.transforms.Lambda(lambda x:perturbation.transform(x))
    testTarget = perturbation.target_transform
    root = f"{dir}/CIFAR100/"
    if train:
        root += "train"
    else:
        root += "test"
    Dataset = torchvision.datasets.CIFAR100(root=root,
        download=True,
        train=train,
        transform=testIMG,
        target_transform=testTarget)
    return Dataset