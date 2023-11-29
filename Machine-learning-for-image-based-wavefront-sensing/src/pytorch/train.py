import os
import time
import src.pytorch.utils as utils
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from src.pytorch.dataset import psf_dataset, splitDataLoader, ToTensor, Normalize
from src.pytorch.utils_visdom import VisdomWebServer
import aotools
from src.pytorch.criterion import *
from torchmetrics import MeanMetric

def train(model, device, dataset, optimizer, criterion, split=[0.9, 0.1], batch_size=32, 
          n_epochs=1, model_dir='./', random_seed=None, visdom=False, factor=1):
    

    # Create directory if doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Logging
    log_path = os.path.join(model_dir, 'logs.log')
    utils.set_logger(log_path)
    
    # Visdom support
    if visdom:
        vis = VisdomWebServer()
   
    # Dataset
    dataloaders = {}
    dataloaders['train'], dataloaders['val'] = splitDataLoader(dataset, split=split, 
                                                             batch_size=batch_size, random_seed=random_seed)

    # ---
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [150, 300])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    #scheduler = CosineWithRestarts(optimizer, T_max=40, eta_min=1e-7, last_epoch=-1)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7, last_epoch=-1)
    
    # Metrics
    metrics_path = os.path.join(model_dir, 'metrics.json')
    
    metrics = {
        'model': model_dir,
        'optimizer': optimizer.__class__.__name__,
        'criterion': criterion.__class__.__name__,
        'scheduler': scheduler.__class__.__name__,
        'dataset_size': int(len(dataset)),
        'train_size': int(split[0]*len(dataset)),
        'test_size': int(split[1]*len(dataset)),
        'n_epoch': n_epochs,
        'batch_size': batch_size,
        'learning_rate': [],
        'train_loss': [],
        'val_loss': [],
        'zernike_train_loss': [],
        'zernike_val_loss': []
    }
    
    # Zernike basis
    z_basis = torch.as_tensor(aotools.zernikeArray(100+1, 128, norm='rms'), dtype=torch.float32)
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    # Training
    since = time.time()
    dataset_size = {
        'train':int(split[0]*len(dataset)),
        'val':int(split[1]*len(dataset))
    }
    
    
    best_loss = 0.0
    
    for epoch in range(n_epochs):
        
        logging.info('-'*30)
        epoch_time = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            zernike_loss =0.0
            
            for _, sample in enumerate(dataloaders[phase]):
                # GPU support 
                inputs = sample['image'].to(device)
                phase_0 = sample['zernike'][:,1:].to(device)*factor
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward: track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    # Network return phase and phase coeffs
                    #print(inputs.shape)
                    #print(phase.shape)
                    phase_estimation = model(inputs)
                    #print(phase_estimation.shape, phase.shape)
                    loss = criterion(torch.squeeze(phase_estimation), torch.squeeze(phase_0))
                    #loss_supervised = criterion(torch.squeeze(phase_estimation), torch.squeeze(zernike))
                    #loss_supervised_meter.update(loss_supervised.detach())
                    #loss = loss_supervised.mean()
                    # backward
                    if phase == 'train':
                        loss.backward()
                        #return loss
                        optimizer.step()   
                    
                    running_loss += 1* loss.item() * inputs.size(0)
            #running_loss = loss_supervised_meter.compute()     

            logging.info('[%i/%i] %s loss MSE: %f' % (epoch+1, n_epochs, phase, running_loss / dataset_size[phase]))
            
            # Update metrics
            metrics[phase+'_loss'].append(running_loss / dataset_size[phase])
            #metrics[phase+'_loss'].append(running_loss.cpu().detach().item())
            #metrics['zernike_'+phase+'_loss'].append(zernike_loss / dataset_size[phase])
            if phase=='train':
                metrics['learning_rate'].append(get_lr(optimizer))
                
            # Adaptive learning rate
            if phase == 'val':
                scheduler.step()
                # Save weigths
                if epoch == 0 or running_loss < best_loss:
                    best_loss = running_loss
                    model_path = os.path.join(model_dir, 'model.pth')
                    torch.save(model.state_dict(), model_path)
                # Save metrics
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4) 
                # Visdom update 
                if visdom:
                    vis.update(metrics)
                    
        logging.info('[%i/%i] Time: %f s' % (epoch + 1, n_epochs, time.time()-epoch_time))
        
    time_elapsed = time.time() - since    
    logging.info('[-----] All epochs completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            
        
def get_lr(optimizer):
    for p in optimizer.param_groups:
        lr = p['lr']
    return lr  
