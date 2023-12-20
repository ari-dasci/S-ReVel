import torchvision
def get_FGVC_data(perturbation, train=True, dir=None):
    if dir is None:
        dir = "./"
    testIMG = torchvision.transforms.Lambda(lambda x:perturbation.transform(x))
    testTarget = perturbation.target_transform
    root = f"{dir}/FGVC/"
    if train:
        Dataset = torchvision.datasets.FGVCAircraft(root=root+"trainval",
        download=True,
        split="trainval",
        transform=testIMG,
        target_transform=testTarget)
    else:
        Dataset = torchvision.datasets.FGVCAircraft(root=root+"test",
        download=True,
        split="test",
        transform=testIMG,
        target_transform=testTarget)
    return Dataset
    