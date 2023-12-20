import torchvision
def get_StandfordCars_data(perturbation, train=True, dir=None):
    if dir is None:
        dir = "./"
    testIMG = torchvision.transforms.Lambda(lambda x:perturbation.transform(x))
    testTarget = perturbation.target_transform
    root = f"{dir}/StandfordCars/"
    if train:
        Dataset = torchvision.datasets.StanfordCars(root=root+"train",
            split="train",
            download=True,
            transform=testIMG,
            target_transform=testTarget)
    else:
        Dataset = torchvision.datasets.StanfordCars(root=root+"test",
            split="test",
            download=True,
            transform=testIMG,
            target_transform=testTarget)
    return Dataset
    