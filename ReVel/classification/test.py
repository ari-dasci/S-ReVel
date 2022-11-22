import argparse
from cProfile import label
import torch
from ReVel.load_data import load_data
from ReVel.perturbations import get_perturbation
torch.set_num_threads(3)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision
import torch.optim as optim
import tqdm
from efficientnet_pytorch import EfficientNet

n_classes = {'CIFAR10':10,   'CIFAR100':100, 'EMNIST':47,'FashionMNIST':10}
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', metavar='D', type=str,default="CIFAR10",
                        help='Dataset to train on.')
    parser.add_argument('--epochs', metavar='E', type=int,default=1000,
                        help='Number of epochs trained')

    parser.add_argument('--seed', metavar='S', type=int,default=3141516,
                        help='Seed for random number generator')
    args = parser.parse_args()

    torch.random.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu" 

    perturbation = get_perturbation("square",dim=4,num_classes=n_classes[args.dataset],final_size=(224,224))
    Test   = load_data(args.dataset,perturbation=perturbation,train=False)
    
    num_classes = n_classes[args.dataset]

    classifier = EfficientNet.from_name('efficientnet-b2',num_classes=num_classes)
    state_dict = torch.load(f"./models/classifier_{args.dataset}.pt")
    classifier.load_state_dict(state_dict)

    batch_size = 8
    TestLoader = DataLoader(Test,batch_size=batch_size,shuffle=False)
    
    
    classifier.to(device)
    
    def loss_f(ypred,y_label):
        return F.cross_entropy(ypred,torch.argmax(y_label,1))
    
    accIterator = tqdm.tqdm(enumerate(TestLoader),total=len(TestLoader))
    total = 0
    classifier.eval()
    m = 0
    for i, data in accIterator :
        inputs, labels = data
        
        inputs      = inputs.float().to(device)
        labels      = labels.to(device)

        # forward + backward + optimize
        outputs     = classifier(inputs)
        
        labels      = torch.argmax(labels,axis=-1)
        outputs     = torch.argmax(outputs,axis=-1)
        
        result   = (outputs == labels).float()
        m       += torch.sum(result)
        total   += len(inputs)
        accIterator.set_postfix({"Running Val acc":(m/total).cpu().numpy()})
    print(f"Accuracy:{m*1.0/total:.4f}")
    