import platform, time, csv
import torch, torch.nn as nn, torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T
import torchvision.datasets as D
from torchvision.models import shufflenet_v2_x0_5
from tqdm import tqdm

def get_data_loaders(
        
        batch_size=128, 

        num_workers=4, 

        pin_memory=True

        ):
    
    mean, std = (

        [0.4377,0.4438,0.4728],

        [0.1980,0.2010,0.1970]
        
        )
    
    train_tf = T.Compose([

        T.RandomCrop(32, padding=4),

        T.RandAugment(num_ops=2, magnitude=9),

        T.RandomRotation(10),

        T.ToTensor(),

        T.Normalize(mean, std),

        T.RandomErasing(p=0.5),

    ])

    val_tf = T.Compose(
        
        [

        T.ToTensor(),

        T.Normalize(
            
            mean, 
            
            std

            ),

    ]
    
    )

    train = D.SVHN(
        
        './data', 

        split='train', 

        download=True, 

        transform=train_tf

        )
    
    extra = D.SVHN(

        './data', 
        
        split='extra', 

        download=True, 

        transform=train_tf

        )
    
    train = ConcatDataset(

        [
            
            train, 
            
            extra
            
            ]

        )
    
    val   = D.SVHN(

        './data', 

        split='test',  

        download=True, 

        transform=val_tf
        
        )

    return (

        DataLoader(

            train, 

            batch_size, 

            shuffle=True,

            num_workers=num_workers, 

            pin_memory=pin_memory

            ),
            
        DataLoader(val,   batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory)
    )

def build_model(num_classes=10, pretrained=True):
    model = shufflenet_v2_x0_5(pretrained=pretrained)
    # replace final fc
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, scheduler):
    model.train()
    running_loss = running_corrects = total = 0
    pbar = tqdm(loader, desc='  Train ', leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device).long()
        optimizer.zero_grad()
        with autocast():
            logits = model(xb)
            loss   = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler: scheduler.step()

        preds = logits.argmax(1)
        running_loss     += loss.item() * xb.size(0)
        running_corrects += (preds==yb).sum().item()
        total            += xb.size(0)
        pbar.set_postfix(loss=f'{loss.item():.4f}',
                         acc =f'{(preds==yb).float().mean():.4f}')
    return running_loss/total, running_corrects/total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = running_corrects = total = 0
    pbar = tqdm(loader, desc='  Valid ', leave=False)
    with torch.no_grad():
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device).long()
            logits = model(xb)
            loss   = criterion(logits, yb)
            preds  = logits.argmax(1)
            running_loss     += loss.item() * xb.size(0)
            running_corrects += (preds==yb).sum().item()
            total            += xb.size(0)
            pbar.set_postfix(loss=f'{loss.item():.4f}',
                             acc =f'{(preds==yb).float().mean():.4f}')
    return running_loss/total, running_corrects/total

def main():
    # ---- Config ----
    epochs       = 20
    batch_size   = 128
    init_lr      = 0.1    # peak LR for OneCycle
    weight_decay = 5e-4
    swa_start    = 15     # start SWA on epoch 16
    # ----------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Data
    is_win = platform.system()=='Windows'
    trn, val = get_data_loaders(
        batch_size,
        num_workers=0 if is_win else 4,
        pin_memory=(device.type=='cuda')
    )

    # Model / loss / optim
    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(
        model.parameters(), lr=init_lr,
        momentum=0.9, weight_decay=weight_decay,
        nesterov=True
    )

    # OneCycleLR: warmup + anneal in 1 cycle
    steps_per_epoch = len(trn)
    scheduler = OneCycleLR(
        optimizer, max_lr=init_lr,
        epochs=epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.1, div_factor=25.0, final_div_factor=1e4
    )

    # SWA
    swa_model     = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=init_lr*0.1)

    scaler = GradScaler()
    best_acc = 0.0

    # CSV log
    with open('metrics.csv','w',newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['epoch','trn_loss','trn_acc','val_loss','val_acc'])
        t0 = time.time()

        for ep in range(1, epochs+1):
            print(f'\nEpoch {ep}/{epochs}')
            trn_l, trn_a = train_one_epoch(
                model, trn, criterion, optimizer,
                device, scaler,
                scheduler if ep<=swa_start else None
            )
            val_l, val_a = validate(model, val, criterion, device)
            print(f'  → Train: loss={trn_l:.4f}, acc={trn_a:.4f}')
            print(f'  → Valid: loss={val_l:.4f}, acc={val_a:.4f}')
            wr.writerow([ep, f'{trn_l:.4f}', f'{trn_a:.4f}',
                         f'{val_l:.4f}', f'{val_a:.4f}'])
            f.flush()

            # SWA update vs. regular scheduler
            if ep > swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            # save best standard
            if val_a > best_acc:
                best_acc = val_a
                torch.save(model.state_dict(), 'best_standard.pth')
                print(f'  ▶ New best STANDARD model: {best_acc:.4f}')

        # finalize SWA
        update_bn(trn, swa_model)
        swa_l, swa_a = validate(swa_model, val, criterion, device)
        print(f'\nSWA accuracy: {swa_a:.4f}')
        torch.save(swa_model.module.state_dict(), 'best_swa.pth')

        elapsed = time.time()-t0
        m,s = int(elapsed//60), int(elapsed%60)
        print(f'\nDone in {m}m{s}s — std={best_acc:.4f}, swa={swa_a:.4f}')

if __name__=='__main__':
    main()
