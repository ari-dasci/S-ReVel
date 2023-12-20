import torchvision
def get_FashionMNIST_data(perturbation, train=True, dir="./data"):
    testIMG = torchvision.transforms.Lambda(lambda x:perturbation.transform(x))
    testTarget = perturbation.target_transform
    root = f"{dir}/FashionMNIST/"
    if train:
        root += "train"
    else:
        root += "test"
    Dataset = torchvision.datasets.FashionMNIST(root=root,
        download=True,
        train=train,
        transform=testIMG,
        target_transform=testTarget)
    return Dataset