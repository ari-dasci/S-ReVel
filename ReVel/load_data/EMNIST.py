import torchvision
def get_EMNIST_data(perturbation, train=True, dir="./data"):
    testIMG = torchvision.transforms.Lambda(lambda x:perturbation.transform(x))
    testTarget = perturbation.target_transform
    root = f"{dir}/EMNIST/"
    if train:
        root += "train"
    else:
        root += "test"
    Dataset = torchvision.datasets.EMNIST(root=root,
        download=True,
        train=False,
        transform=testIMG,
        target_transform=testTarget,
        split="balanced")
    return Dataset