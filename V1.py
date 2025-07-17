import os
import platform
import time
import copy
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import shufflenet_v2_x0_5
from tqdm import tqdm

def get_data_loaders(batch_size=128, num_workers=0, pin_memory=False):
    
    mean = [
        
        0.4377, 
        
        0.4438, 
        
        0.4728
        
        ]
    
    std  = [
        
        0.1980, 
        
        0.2010, 
        
        0.1970
        
        ]

    train_tf = transforms.Compose([

        transforms.RandomCrop(32, padding=4),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(mean, std),

    ])

    val_tf = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(mean, std),

    ])

    train_set = torchvision.datasets.SVHN(

        root='./data', 
        
        split='train', 
        
        download=True, 
        
        transform=train_tf
        
        )
    
    extra_set = torchvision.datasets.SVHN(
        
        root='./data', 
        
        split='extra', 
        
        download=True, 
        
        transform=train_tf
        
        )
    
    train_set = torch.utils.data.ConcatDataset(
        
        [
            
        train_set, 
         
        extra_set
        
        ]
        
        )

    val_set = torchvision.datasets.SVHN(

        root='./data', 
        
        split='test', 
        
        download=True, 

        transform=val_tf
        
        )

    train_loader = DataLoader(

        train_set, 
        
        batch_size=batch_size, 
        
        shuffle=True,
        
        num_workers=num_workers, 
        
        pin_memory=pin_memory
        
        )
    
    val_loader = DataLoader(

        val_set, 
        
        batch_size=batch_size, 
        
        shuffle=False,

        num_workers=num_workers, 
        
        pin_memory=pin_memory
        
        )

    return train_loader, val_loader

def build_model(
        
        num_classes=10, 
        
        pretrained=True
        
        ):
    
    # use ShuffleNetV2 0.5x as a lightweight backbone
    
    model = shufflenet_v2_x0_5(
        
        pretrained=pretrained

        )
    
    # adjust final FC for 10 classes

    in_feats = model.fc.in_features
    model.fc = nn.Linear(
        
        in_feats, 
        
        num_classes
        
        )

    return model

def train_one_epoch(
        
        model, 
        
        loader, 
        
        criterion, 
        
        optimizer, 
        
        device
        
        ):
    
    model.train()

    running_loss = 0.0
    running_corrects = 0
    total = 0

    loop = tqdm(
        
        loader, 
        
        desc='  Training', 
        
        leave=False
        
        )
    
    for inputs, targets in loop:
    
        inputs = inputs.to(device)

        targets = targets.to(device).long()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(
            
            outputs, 
            
            targets
            
            )
        
        loss.backward()
        
        optimizer.step()

        _, preds = torch.max(
            
            outputs,
             
              1

            )
        
        batch_acc = (preds == targets).float().mean().item()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == targets).sum().item()
        total += inputs.size(0)

        loop.set_postfix(loss=f'{loss.item():.4f}', acc=f'{batch_acc:.4f}')

    return running_loss / total, running_corrects / total

def validate(
        
        model, 
        
        loader, 
        
        criterion, 
        
        device
        
        ):

    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    loop = tqdm(
        
        loader, 
        
        desc='  Validating', 
        
        leave=False
        
        )
    
    with torch.no_grad():

        for inputs, targets in loop:
        
            inputs = inputs.to(device)
            targets = targets.to(device).long()

            outputs = model(inputs)

            loss = criterion(

                outputs, 
                
                targets
                
                )
            
            _, preds = torch.max(
                
                outputs, 
                
                1
                
                )
            
            batch_acc = (preds == targets).float().mean().item()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == targets).sum().item()
            total += inputs.size(0)

            loop.set_postfix(loss=f'{loss.item():.4f}', acc=f'{batch_acc:.4f}')

    return running_loss / total, running_corrects / total

def main():

    # Hyperparameters

    num_epochs    = 30

    batch_size    = 128

    learning_rate = 0.01

    momentum      = 0.9

    weight_decay  = 1e-4

    step_size     = 10

    gamma         = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Win10 multiprocessing guard
    is_windows = platform.system() == "Windows"
    num_workers = 0 if is_windows else 4
    pin_memory  = device.type == "cuda"

    train_loader, val_loader = get_data_loaders(

        batch_size=batch_size,

        num_workers=num_workers,

        pin_memory=pin_memory

    )

    model     = build_model(
        
        num_classes=10, 
        
        pretrained=True
        
        ).to(device)
    
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(

        model.parameters(),

        lr=learning_rate,

        momentum=momentum,

        weight_decay=weight_decay

    )
    scheduler = lr_scheduler.StepLR(

        optimizer, 
        
        step_size=step_size, 
        
        gamma=gamma
    
    )

    # CSV logger
    csv_path = 'training_log.csv'

    with open(
        
        csv_path, 
        
        'w', 
        
        newline=''
        
        ) as csvfile:

        writer = csv.writer(csvfile)
        
        writer.writerow(
            
            [
                
                'epoch',
                
                'train_loss',
                
                'train_acc',
                
                'val_loss',
                
                'val_acc'
                
                ]
            
            )

        best_acc = 0.0
        since = time.time()

        for epoch in range(
            
            1,
            
            num_epochs+1
            
            ):
            
            print(f'\nEpoch {epoch}/{num_epochs}')
            
            train_loss, train_acc = train_one_epoch(

                model, 

                train_loader, 

                criterion, 

                optimizer, 

                device
                
                )
            
            val_loss, val_acc = validate(
                
                model, 
                
                val_loader, 

                criterion, 

                device
                
                )

            print(f'  Train loss: {train_loss:.4f}  acc: {train_acc:.4f}')
            print(f'  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}')

            writer.writerow(
                
                [epoch,
                 
                             f'{train_loss:.4f}', f'{train_acc:.4f}',
                             f'{val_loss:.4f}', f'{val_acc:.4f}'
                             
                ])
            
            csvfile.flush()

            # save best immediately
            if val_acc > best_acc:

                best_acc = val_acc

                torch.save(
                    
                    model.state_dict(), 
                    
                    'shufflenet_v2_x0_5_svhn_best.pth'
                    
                    )
                
                print(f'  ▶ New best model saved (val_acc={best_acc:.4f})')

            scheduler.step()

        elapsed = time.time() - since
        
        print(f'\nTraining complete in {elapsed//60:.0f}m {elapsed%60:.0f}s')
        print(f'Best validation accuracy: {best_acc:.4f}')
        print(f'Checkpointed best model to shufflenet_v2_x0_5_svhn_best.pth')
        print(f'Full log available at {csv_path}')

if __name__ == '__main__':
    main()
