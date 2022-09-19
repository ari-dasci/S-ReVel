import torchvision
def get_CIFAR10_data(perturbation, train=True, dir=None):
    if dir is None:
        dir = "./"
    testIMG = torchvision.transforms.Lambda(lambda x:perturbation.transform(x))
    testTarget = perturbation.target_transform
    root = f"{dir}/CIFAR10/"
    if train:
        root += "train"
    else:
        root += "test"
    Dataset = torchvision.datasets.CIFAR10(root=root,
        download=True,
        train=False,
        transform=testIMG,
        target_transform=testTarget)
    return Dataset
    