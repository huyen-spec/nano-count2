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
from dataset import BasicDataset, PramDataset, Gaussian_Masks_Dataset      
import logging
from tqdm import tqdm
from utils.evaluate import evaluate
import pdb
import random


# from torch._six import int_classes as _int_classes
int_classes = int
from torch import Tensor
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
from sklearn.model_selection import KFold

T_co = TypeVar('T_co', covariant=True)

dir_img = Path('/home/huyentn2/project/nano_count/segmentation_unet/data/img_patch/')
dir_mask = Path('/home/huyentn2/project/nano_count/segmentation_unet/data/binary_mask/')
dir_checkpoint = Path('/home/huyentn2/project/nano_count/segmentation_unet/checkpoints')
if not os.path.exists(dir_checkpoint):
    os.makedirs(dir_checkpoint)
split_dir = '/home/huyentn2/project/nano_count/segmentation_unet/data/split/'
gauss_mask = '/home/huyentn2/project/nano_count/segmentation_unet/data/gauss_mask/'


def train_model(
        model,
        device,
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
    # 1. Create dataset
    try:
        dataset = PramDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)


    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    name_list = {}
    for i, idx in enumerate(val_set.indices):
        name_list[i] = dataset.ids[i]

    with open(split_dir + 'data_names.json', 'w') as fp:
        json.dump(name_list, fp)


    # save the split indices

    with open(split_dir + 'train-seed_{}.npy'.format(0), 'wb') as f:
        np.save(f, np.asarray(train_set.indices))

    with open(split_dir+ 'val-seed_{}.npy'.format(0), 'wb') as f:
        np.save(f, np.asarray(val_set.indices))

    # dictionary ={}

    # for i in range(len(dataset.ids)):
    #     dictionary[i] = dataset.ids[i]
        
    # with open("image_names.json", "w") as outfile:
    #     json.dump(dictionary, outfile)
    

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
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
    ''')    

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
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
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

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')




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



def train(
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
                        # scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
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

    # pdb.set_trace()

    with open(split_dir + 'test-dice_{}_fold.npy'.format(k_fold), 'wb') as f:
        np.save(f, np.asarray(test_score_list))

    with open(split_dir + 'train-dice_{}_fold.npy'.format(k_fold), 'wb') as f:
        np.save(f, np.asarray(train_score_list))

    with open(split_dir + 'epoch-loss_{}_fold.npy'.format(k_fold), 'wb') as f:
        np.save(f, np.asarray(epoch_loss_list))

    return test_score_list, train_score_list, epoch_loss_list


def train_gauss(
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
        weight_decay: float = 1e-5,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
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

    # criterion = nn.CrossEntropyLoss() if model.module.n_classes > 1 else nn.BCEWithLogitsLoss()

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
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.module.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    else:
                        loss = criterion(F.sigmoid(masks_pred), true_masks)


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
                        # scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))

                        pdb.set_trace()
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

    # pdb.set_trace()

    with open(split_dir + 'test-dice_{}_fold.npy'.format(k_fold), 'wb') as f:
        np.save(f, np.asarray(test_score_list))

    with open(split_dir + 'train-dice_{}_fold.npy'.format(k_fold), 'wb') as f:
        np.save(f, np.asarray(train_score_list))

    with open(split_dir + 'epoch-loss_{}_fold.npy'.format(k_fold), 'wb') as f:
        np.save(f, np.asarray(epoch_loss_list))

    return test_score_list, train_score_list, epoch_loss_list


def train_kfold_model(
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
        use_gauss: bool = True,
        ):

    if use_gauss:
        try:
            dataset = Gaussian_Masks_Dataset(dir_img, gauss_mask, img_scale)
        except (AssertionError, RuntimeError, IndexError):
            dataset = BasicDataset(dir_img, dir_mask, img_scale)

    else:
        try:
            dataset = PramDataset(dir_img, dir_mask, img_scale)
        except (AssertionError, RuntimeError, IndexError):
            dataset = BasicDataset(dir_img, dir_mask, img_scale)        

    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    
    random_seed = 10
    fold = KFold(5, shuffle=True, random_state=random_seed)

    for k,(tr_idx, test_idx) in enumerate(fold.split(dataset)):
        if k == n_units:
            with open(split_dir + 'trainset-index_{}fold.npy'.format(k), 'wb') as f:
                np.save(f, np.asarray(tr_idx))        
            with open(split_dir + 'testset-index_{}fold.npy'.format(k), 'wb') as f:
                np.save(f, np.asarray(test_idx))     

    for k,(tr_idx, test_idx) in enumerate(fold.split(dataset)):
        # print(k)
        # print(test_idx)
        
        if k != n_units:
            continue

        n_train = len(tr_idx)
        n_val = len(test_idx)

        train_loader = DataLoader(dataset, **loader_args,
                                               sampler = SubsetRandomSampler(tr_idx)
                                            )
        val_loader = DataLoader(dataset, **loader_args,
                                               sampler = SubsetRandomSampler(test_idx)
                                            )

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

        # pdb.set_trace()
        if dataset.mask_values != None:

            test_score_list, train_score_list, epoch_loss_list = train(
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
                                                                )
        
        else:
            test_score_list, train_score_list, epoch_loss_list = train_gauss(
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
                                                                )
        
