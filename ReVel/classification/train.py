import argparse
from cProfile import label
import torch
from ReVel.load_data import load_data
torch.set_num_threads(3)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision
import torch.optim as optim
import tqdm
from efficientnet_pytorch import EfficientNet


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

    Train   = load_data(args.dataset,train=True)
    Train, Val = random_split(Train, [int(len(Train)*0.9), len(Train)-int(len(Train)*0.9)])
    num_classes = len(Train.dataset.classes) 

    classifier = EfficientNet.from_name('efficientnet-b2',num_classes=num_classes)
    state_dict = classifier.state_dict()
    pretrained = EfficientNet.from_pretrained('efficientnet-b2',
            weights_path=f'./Damax/classification/pretrained_model/efficientnet-b2-8bb594d6.pth')
    pretrained_state_dict = pretrained.state_dict()

    pretrained_state_dict["_fc.weight"] = state_dict['_fc.weight']
    pretrained_state_dict["_fc.bias"] = state_dict['_fc.bias']
    classifier.load_state_dict(pretrained_state_dict)

    batch_size = 8
    TrainLoader = DataLoader(Train,batch_size=batch_size,shuffle=True)
    ValLoader = DataLoader(Val,batch_size=batch_size,shuffle=False)
    
    
    classifier.to(device)
    
    def loss_f(ypred,y_label):
        return F.cross_entropy(ypred,torch.argmax(y_label,1))
    
    optimizer = optim.Adam(classifier.parameters(), lr=0.001,weight_decay=0.01,amsgrad=True)

    epochs = range(args.epochs)
    
    best_loss = float('inf')
    for epoch in epochs:
        batches = tqdm.tqdm(enumerate(TrainLoader),total = len(TrainLoader))
        batches.set_description(f'Epoch {epoch+1}')
        running_loss = 0.0
        total, m = 0,0
        classifier.train()
        for i, data in batches:
            inputs, labels = data
            inputs = inputs.float().to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = classifier(inputs)
            
            labelsAcc = torch.argmax(labels,axis=-1)
            outputsAcc = torch.argmax(outputs,axis=-1)
            result = (outputsAcc == labelsAcc).float()
            
            m += torch.sum(result)
            total+=len(result)
            
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_reported = loss.item()
            running_loss += loss_reported
            batches.set_postfix({"Running loss": running_loss*batch_size/total,
                "Running Acc":(m/total).cpu().numpy()})
        accIterator = tqdm.tqdm(enumerate(ValLoader),total=len(ValLoader))
        total,m = 0, 0.0
        classifier.eval()
        last_loss = 0.0
        for i, data in accIterator :
            
            data = data
            inputs, labels = data
            total +=len(inputs)
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = classifier(inputs)
            last_loss += loss_f(outputs, labels).item()

            labels = torch.argmax(labels,axis=-1)
            outputs = torch.argmax(outputs,axis=-1)
            result = (outputs == labels).float()
            m += torch.sum(result)
            accIterator.set_postfix({"Running Val loss": last_loss*batch_size/total,
                "Running Val Acc":(m/total).cpu().numpy()})
        if best_loss is None or best_loss > last_loss:
            best_loss = last_loss
            torch.save(classifier.state_dict(), f"./models/classifier_{args.dataset}.pt")
            print(f"New best model: val_loss {best_loss}")