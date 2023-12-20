import torchvision
def get_OxforIIITPet_data(perturbation, train=True, dir=None):
    if dir is None:
        dir = "./"
    testIMG = torchvision.transforms.Lambda(lambda x:perturbation.transform(x))
    testTarget = perturbation.target_transform
    root = f"{dir}/OxforIIITPet/"
    if train:
        Dataset = torchvision.datasets.OxfordIIITPet(root=root+"trainval",
            split="trainval",
            download=True,
            transform=testIMG,
            target_transform=testTarget)
    else:
        Dataset = torchvision.datasets.OxfordIIITPet(root=root+"test",
            split="test",
            download=True,
            transform=testIMG,
            target_transform=testTarget)
    return Dataset
    