from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader, random_split
import os
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional  as F
import wandb
from pathlib import Path

import json
import numpy as np
import sys        
sys.path.append('/home/huyentn2/project/nano_count/segmentation_unet/')
from dataset import BasicDataset, PramDataset , Gaussian_Masks_Dataset  
import logging
from tqdm import tqdm
from utils.evaluate import evaluate,  eval_fold_mode, evaluate_gauss  #get_prediction, 
import pdb
import random


# from torch._six import int_classes as _int_classes
int_classes = int
from torch import Tensor
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
from sklearn.model_selection import KFold

T_co = TypeVar('T_co', covariant=True)

# original radius
# dir_img = Path('/home/huyentn2/project/nano_count/segmentation_unet/data/img_patch/')
# dir_mask = Path('/home/huyentn2/project/nano_count/segmentation_unet/data/binary_mask/')


#1/3 radius
# dir_img = Path('/home/huyentn2/project/nano_count/segmentation_unet/data/PRAM_annotation/mask_generation/test_radius/img_patch/')
# dir_mask = Path('/home/huyentn2/project/nano_count/segmentation_unet/data/PRAM_annotation/mask_generation/test_radius/binary_mask/')


#point
dir_img = Path('/home/huyentn2/project/nano_count/segmentation_unet/data/PRAM_annotation/mask_generation/test_point/img_patch/')
dir_mask = Path('/home/huyentn2/project/nano_count/segmentation_unet/data/PRAM_annotation/mask_generation/test_point/binary_mask/')


gauss_dir = Path('/home/huyentn2/project/nano_count/segmentation_unet/data/PRAM_annotation/mask_generation/test_point/binary_mask/')


# dir_checkpoint = Path('/home/huyentn2/huyen/project/project_nano_count/segmentation_unet_/checkpoints')
# dir_checkpoint = Path('/home/huyentn2/huyen/project/project_nano_count/segmentation_unet_/all_checkpts/checkpoints_seashell')
# if not os.path.exists(dir_checkpoint):
#     os.makedirs(dir_checkpoint)
# split_dir = '/home/huyentn2/huyen/project/project_nano_count/segmentation_unet_/data/split_2/'  # this has been messed between runs
# split_dir_5fold  = '/home/huyentn2/huyen/project/project_nano_count/segmentation_unet_/data/split_5fold_2/'  
# split_29_3 = '/home/huyentn2/huyen/project/project_nano_count/segmentation_unet_/data/split_29_3/'

# gauss_dir = '/home/huyentn2/huyen/project/data_nano/gauss_mask/'


full_size_dir = '/home/huyentn2/project/nano_count/segmentation_unet/data/PRAM_annotation/combined/'




class Sampler(Generic[T_co]):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source: Optional[Sized]) -> None:
        pass

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError



class SubsetRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices), generator=self.generator))

    def __len__(self):
        return len(self.indices) 





def train_w_gauss_kernel(
        save_dir,
        k_fold,
        model,
        device,
        train_loader,
        val_loader,
        n_train,
        experiment,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        use_gauss: bool = False,
    ):

    # 3. Create data loaders
    # loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP

    # pdb.set_trace()
    # torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=False)
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    # TypeError: __init__() got an unexpected keyword argument 'foreach'
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    if not use_gauss:
        criterion = nn.CrossEntropyLoss() if model.module.n_classes > 1 else nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.MSELoss(reduction='mean')
    global_step = 0

    test_score_list = []
    train_score_list = []
    epoch_loss_list = []

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        num_batch = 0
        i = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                i+=1

                images, true_masks = batch['image'], batch['mask']

                # assert images.shape[1] == model.module.n_channels, \
                #     f'Network has been defined with {model.module.n_channels} input channels, ' \
                #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                if use_gauss:
                    true_masks = true_masks.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

                
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)

                    # pdb.set_trace()

                    # if use_gauss:
                    #     masks_pred = masks_pred[:,1,:,:].unsqueeze(dim=1)

                    if model.module.n_classes == 1:
                        # pdb.set_trace()
                        loss = criterion(masks_pred.squeeze(1), true_masks.squeeze(1))
                    else:
                        loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                num_batch += 1
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not torch.isinf(value).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not torch.isinf(value.grad).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        if use_gauss:
                            err_mean, err_std, abs_mean_err, loss = evaluate_gauss(model, val_loader, device, amp)
                        # scheduler.step(val_score)

                        if i % 2 == 0:
                            logging.info('Validation mean error: {}'.format(err_mean))
                        # pdb.set_trace()
                        if use_gauss:
                            try:
                                experiment.log({
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    'validation mean error': abs_mean_err,
                                    'images': wandb.Image(images[0].cpu()),
                                    'masks': {
                                        'true': wandb.Image(true_masks[0].float().cpu()),
                                        'pred': wandb.Image(masks_pred[0].float().cpu()),
                                    },
                                    'step': global_step,
                                    'epoch': epoch,
                                    **histograms
                                })
                            except:
                                pass
        epoch_loss_list.append(epoch_loss/num_batch)
        logging.info('Loss at epoch {}: {}\n'.format(epoch, epoch_loss/num_batch))  

        if use_gauss:
            # pdb.set_trace()
            err_mean, err_std, abs_mean_err, loss = evaluate_gauss(model, val_loader, device, amp)

            logging.info('Test loss: {}'.format(loss))
            logging.info('Test absolute mean error: {}'.format(abs_mean_err))
            logging.info('Test error mean: {}'.format(err_mean))
            logging.info('Test error devition: {}'.format(err_std))

            # print(f"{'Test'}:\n"
            #   f"\tAverage loss: {epoch_loss/num_batch:3.4f}\n"
            #   f"\tMean error: {err_mean:3.3f}\n"
            #   f"\tMean absolute error: {abs_mean_err:3.3f}\n"
            #   f"\tError deviation: {err_std:3.3f}")

            # test_score_list.append(test_score.cpu())
            # err_mean, err_std, abs_mean_err = evaluate_gauss(model, train_loader, device, amp)
            # train_score_list.append(train_score.cpu())


        if save_checkpoint:
            dir_checkpoint = str(Path(save_dir) / 'fold{}'.format(k_fold))
            if not os.path.exists(dir_checkpoint):
                os.makedirs(dir_checkpoint)

            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = mask_values
            torch.save(state_dict, str(Path(save_dir) / 'fold{}'.format(k_fold) /'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


    # with open(split_29_3 + 'epoch-loss_{}_fold.npy'.format(k_fold), 'wb') as f:
    #     np.save(f, np.asarray(epoch_loss_list))

    return test_score_list, train_score_list, epoch_loss_list





def train(
        save_dir,
        k_fold,
        model,
        device,
        train_loader,
        val_loader,
        n_train,
        experiment,
        mask_values,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        use_gauss: bool = False,
    ):
    

    # 3. Create data loaders
    # loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP

    # pdb.set_trace()
    # torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=False)
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    # TypeError: __init__() got an unexpected keyword argument 'foreach'
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    if not use_gauss:
        criterion = nn.CrossEntropyLoss() if model.module.n_classes > 1 else nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.MSELoss(reduction='sum')
    global_step = 0



    test_score_list = []
    train_score_list = []
    epoch_loss_list = []


    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.module.n_channels, \
                    f'Network has been defined with {model.module.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                if not use_gauss:
                    true_masks = true_masks.to(device=device, dtype=torch.long)
                else:
                    true_masks = true_masks.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

                

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)

                    # if use_gauss:
                    #     masks_pred = masks_pred[:,1,:,:].unsqueeze(dim=1)

                    if model.module.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    else:
                        loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not torch.isinf(value).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not torch.isinf(value.grad).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        if not use_gauss:
                            val_score = evaluate(model, val_loader, device, amp)
                        else:
                            val_score = evaluate_gauss(model, val_loader, device, amp)
                        # scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        # pdb.set_trace()
                        if not use_gauss:
                            try:
                                experiment.log({
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    'validation Dice': val_score,
                                    'images': wandb.Image(images[0].cpu()),
                                    'masks': {
                                        'true': wandb.Image(true_masks[0].float().cpu()),
                                        'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                    },
                                    'step': global_step,
                                    'epoch': epoch,
                                    **histograms
                                })
                            except:
                                pass
                        else:
                            try:
                                experiment.log({
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    'validation Dice': val_score,
                                    'images': wandb.Image(images[0].cpu()),
                                    'masks': {
                                        'true': wandb.Image(true_masks[0].float().cpu()),
                                        'pred': wandb.Image(masks_pred[1][0].float().cpu()),
                                    },
                                    'step': global_step,
                                    'epoch': epoch,
                                    **histograms
                                })
                            except:
                                pass

                                
        if not use_gauss:
            test_score = evaluate(model, val_loader, device, amp)
            test_score_list.append(test_score.cpu())
            train_score = evaluate(model, train_loader, device, amp)
            train_score_list.append(train_score.cpu())

        epoch_loss_list.append(epoch_loss)
        logging.info('Loss at epoch {}: {}'.format(epoch, epoch_loss))

        if save_checkpoint:
            dir_checkpoint = str(Path(save_dir) / 'fold{}'.format(k_fold))
            if not os.path.exists(dir_checkpoint):
                os.makedirs(dir_checkpoint)

            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = mask_values
            torch.save(state_dict, str(Path(save_dir) / 'fold{}'.format(k_fold) /'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


    # with open(split_29_3 + 'epoch-loss_{}_fold.npy'.format(k_fold), 'wb') as f:
    #     np.save(f, np.asarray(epoch_loss_list))

    return test_score_list, train_score_list, epoch_loss_list




def get_idx(index_dict, files, train_id, val_id):

    train_f = [files[i] for i in train_id]
    val_f = [files[i] for i in val_id]

    # pdb.set_trace()
    train_indices = []
    val_indices = []
    for key in index_dict:
        if key in train_f:
            for i in index_dict[key]:
                train_indices.append(i)
        elif key in val_f:
            for i in index_dict[key]:
                val_indices.append(i)

    return train_indices, val_indices


def train_kfold_model(
        save_dir,
        num_fold, 
        eval,
        n_units,
        model,
        device,
        experiment,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        use_gauss: bool = False,
        sigma: float = None, 
        ):
    
    try:
        if not use_gauss:
            dataset = PramDataset(dir_img, dir_mask, img_scale)
        else:
            dataset = Gaussian_Masks_Dataset(sigma, dir_img, gauss_dir, img_scale)

    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # pdb.set_trace()
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    
    # split dataset for k-fold cross validation
    fold = KFold(num_fold, shuffle=True, random_state=10)

    files = os.listdir(full_size_dir)

    files = [file.replace(".tiff", "") for file in files]


    index_dict = {}
    sum = 0
    for file in files:
        temp = []
        for id in dataset.ids:
            if file == id[:-2]:
                # print(1)
                temp.append(dataset.ids.index(id))
            else:
                continue
        if len(temp) == 0:
            continue
        index_dict[file] = temp
        sum += len(temp)


    new_files = list(index_dict.keys())

    # pdb.set_trace()
    # for k,(tr_idx, test_idx) in enumerate(fold.split(new_files)):
    #     pdb.set_trace()

        
        # if k == n_units:
        #     with open(split_dir_5fold + 'trainset-index_{}fold.npy'.format(k), 'wb') as f:
        #         np.save(f, np.asarray(tr_idx))        
        #     with open(split_dir_5fold + 'testset-index_{}fold.npy'.format(k), 'wb') as f:
        #         np.save(f, np.asarray(test_idx))     

    for k,(tr_idx, test_idx) in enumerate(fold.split(new_files)):
        # print(k)
        # print(test_idx)
        
        if k != n_units:
            continue

        train_idx, val_idx = get_idx(index_dict, new_files, tr_idx, test_idx)

        # save the index for later inference
        with open(save_dir + 'trainset-index_{}fold.npy'.format(k), 'wb') as f:
            np.save(f, np.asarray(tr_idx))        
        with open(save_dir + 'testset-index_{}fold.npy'.format(k), 'wb') as f:
            np.save(f, np.asarray(test_idx))   

        n_train = len(train_idx)
        n_val = len(val_idx)

        with open(save_dir + 'trainset-index_{}fold.npy'.format(k), 'wb') as f:
            np.save(f, np.asarray(train_idx))        
        with open(save_dir + 'testset-index_{}fold.npy'.format(k), 'wb') as f:
            np.save(f, np.asarray(val_idx))   


        train_loader = DataLoader(dataset, **loader_args, sampler = SubsetRandomSampler(train_idx))

        # pdb.set_trace()

        val_loader = DataLoader(dataset, **loader_args, sampler = SubsetRandomSampler(val_idx))



        if eval:
            if not use_gauss:

                score = eval_fold_mode(save_dir, model, val_loader, k, device, use_gauss)

                print('Validation Dice of fold {} score: {}'.format(k, score))
                return
            
            elif use_gauss:

                score = eval_fold_mode(save_dir, model, val_loader, k, device, use_gauss)
                print('Validation absolute mean error of fold {}: {}'.format(k, score))
                return

        # (Initialize logging)
        # experiment = wandb.init(project='U-Net-{}-fold'.format(k), resume='allow', anonymous='must')
        experiment.config.update(
            dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
        )

        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Images scaling:  {img_scale}
            Mixed Precision: {amp}
            K-Fold:          {k}
        ''')    

        if not use_gauss:
            test_score_list, train_score_list, epoch_loss_list = train(
                                                                    save_dir,
                                                                    n_units,
                                                                    model,
                                                                    device,
                                                                    train_loader,
                                                                    val_loader,
                                                                    n_train,
                                                                    experiment,
                                                                    dataset.mask_values,
                                                                    epochs,
                                                                    batch_size,
                                                                    learning_rate,
                                                                    val_percent,
                                                                    save_checkpoint,
                                                                    img_scale,
                                                                    amp,
                                                                    weight_decay,
                                                                    momentum,
                                                                    gradient_clipping,
                                                                    use_gauss,
                                                                )     
            with open(save_dir + 'epoch-loss_{}_fold.npy'.format(n_units), 'wb') as f:
                np.save(f, np.asarray(epoch_loss_list))


        elif use_gauss:

            train_w_gauss_kernel(
                                save_dir,
                                n_units,
                                model,
                                device,
                                train_loader,
                                val_loader,
                                n_train,
                                experiment,
                                epochs,
                                batch_size,
                                learning_rate,
                                val_percent,
                                save_checkpoint,
                                img_scale,
                                amp,
                                weight_decay,
                                momentum,
                                gradient_clipping,
                                use_gauss,
                            )  






def train_scheduler(
        k_fold,
        model,
        device,
        train_loader,
        val_loader,
        train_val_loader,
        n_train,
        experiment,
        mask_values,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
    ):
    

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP

    # pdb.set_trace()
    # torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=False)
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    # TypeError: __init__() got an unexpected keyword argument 'foreach'
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.module.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0



    test_score_list = []
    train_score_list = []
    epoch_loss_list = []


    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.module.n_channels, \
                    f'Network has been defined with {model.module.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.module.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    else:
                        loss = criterion(masks_pred, true_masks)


                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not torch.isinf(value).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not torch.isinf(value.grad).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        train_val_score = evaluate(model, train_val_loader, device, amp)
                        scheduler.step(train_val_score)
                        
                        logging.info('Validation Dice score: {}'.format(train_val_score))
                        logging.info('Test Dice score: {}'.format(val_score))
                        # pdb.set_trace()
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': train_val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass


        test_score = evaluate(model, val_loader, device, amp)
        test_score_list.append(test_score.cpu())

        train_score = evaluate(model, train_loader, device, amp)
        train_score_list.append(train_score.cpu())

        epoch_loss_list.append(epoch_loss)
        logging.info('Loss at epoch {}: {}'.format(epoch, epoch_loss))

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = mask_values
            torch.save(state_dict, str(dir_checkpoint / 'fold{}'.format(k_fold) /'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

    return test_score_list, train_score_list, epoch_loss_list



@torch.inference_mode()
def eval_single(
        input_dir,
        save_dir,
        model,
        device,
        img_scale: float = 0.5,
        ):   

    from PIL import Image
    def preprocess(pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH))
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if img.ndim == 2:
            img = img[np.newaxis, ...]   #  img = (24,24)   -> img = (1,24,24)   append new dim at the beginning
        else:
            img = img.transpose((2, 0, 1))
        if (img > 1).any():
            img = img / 255.0

        return torch.as_tensor(img.copy()).float().contiguous()

    import pdb
    # pdb.set_trace()

    image = preprocess(Image.open(input_dir), img_scale)

    image = image.unsqueeze(dim=0).to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
    # with torch.no_grad():
    mask_pred = model(image)

    binary_pred = F.one_hot(mask_pred.argmax(dim=1), model.module.n_classes).permute(0, 3, 1, 2).float()

    mask_pred = F.sigmoid(mask_pred)

    # pdb.set_trace()
    img_name = input_dir.split("/")[-1]

    save_img = mask_pred.detach().cpu().numpy().squeeze()[1]  # particles bright

    print("here")
    Image.fromarray((save_img*255).astype('uint8')).save(save_dir + "/mask2_" + img_name)

    return save_img







@torch.inference_mode()
def eval_single_count(
            input_dir,
            save_dir,
            model,
            device,
            img_scale: float = 0.5,
            ):
    

    from PIL import Image
    def preprocess(pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH))
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if img.ndim == 2:
            img = img[np.newaxis, ...]   #  img = (24,24)   -> img = (1,24,24)   append new dim at the beginning
        else:
            img = img.transpose((2, 0, 1))
        if (img > 1).any():
            img = img / 255.0

        return torch.as_tensor(img.copy()).float().contiguous()

    # import pdb
    # pdb.set_trace()

    image = preprocess(Image.open(input_dir), img_scale)

    image = image.unsqueeze(dim=0).to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
    # with torch.no_grad():
    mask_pred = model(image)


    return mask_pred.cpu()



