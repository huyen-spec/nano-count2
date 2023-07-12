import sys
sys.path.insert(0, '/home/huyentn2/huyen/project/')
# from UNET import UNet
from utils.UNET import UNet
import PIL
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional  as F
import matplotlib.pyplot as plt
import pdb
import numpy as np
from PIL import Image
import cv2
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader, random_split
# from evaluate import evaluate


import wandb
import logging
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff, dice_coeff


# img_rgb = mask_pred[0][0].squeeze().cpu().numpy()
# PIL.Image.fromarray((img_rgb*255).astype('uint8')).save('/home/huyentn2/huyen/project/save1.png')
# /home/huyentn2/huyen/project/UNET/evaluate

dir_img = Path('/home/huyentn2/huyen/project/data_nano/img_patch/')
dir_mask = Path('/home/huyentn2/huyen/project/data_nano/binary_mask/')
dir_checkpoint = Path('/home/huyentn2/huyen/project/project_nano_count/segmentation_unet/checkpoints/')


# save_mask = Path()
# save_img = Path()

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        it = 0 
        id = 0
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            f, axarr = plt.subplots(3,2,  figsize=(10, 10))
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.module.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.module.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format

                # pdb.set_trace()
                mask_hold = mask_pred
                mask_true = F.one_hot(mask_true, net.module.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.module.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background

                # pdb.set_trace()

                # t = 0
                # axarr[0,0].imshow(mask_pred[t][0].squeeze().cpu().numpy())
                # axarr[0,1].imshow(mask_true[t][0].squeeze().cpu().numpy())

                # axarr[1,0].imshow(mask_pred[t+1][0].squeeze().cpu().numpy())
                # axarr[1,1].imshow(mask_true[t+1][0].squeeze().cpu().numpy())

                # axarr[2,0].imshow(mask_pred[t+2][0].squeeze().cpu().numpy())
                # axarr[2,1].imshow(mask_true[t+2][0].squeeze().cpu().numpy())

                # it+=1
                # f.savefig("prediction_{}.png".format(t*3+it))

                # draw_pred(mask_pred,mask_true)
                draw_pred(F.sigmoid(mask_hold),mask_true, id)
                id+=1

                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
            
    net.train()

    print("Dice:{}".format(dice_score / max(num_val_batches, 1)))
    
    return dice_score / max(num_val_batches, 1)

def draw_pred(pred,mask, id):
    for i in range(10):
        img_rgb = pred[i][0].squeeze().cpu().numpy()
        img_rgb_true = mask[i][0].squeeze().cpu().numpy()
        PIL.Image.fromarray((img_rgb*255).astype('uint8')).save('/home/huyentn2/huyen/project/UNET/evaluate/Binary-Segmentation-Evaluation-Tool/test_data/rs1/{}.png'.format(id*10+i))
        PIL.Image.fromarray((img_rgb_true*255).astype('uint8')).save('/home/huyentn2/huyen/project/UNET/evaluate/Binary-Segmentation-Evaluation-Tool/test_data/gt/{}.png'.format(id*10+i))


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):

    # pdb.set_trace()

    mask_file = list(mask_dir.glob(idx  + mask_suffix + '.*'))[0]

    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')



class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        # pdb.set_trace()

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        # pdb.set_trace()
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH))
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i  #  check if the value in last dim, eg: channel has all pixels across channels equal v, if yes assign that pos = v on mask
            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]   #  img = (24,24)   -> img = (1,24,24)   append new dim at the beginning
            else:
                img = img.transpose((2, 0, 1))
            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        # mask_file = list(self.mask_dir.glob(name + '.*'))
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))


        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class PramDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        # super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
        super().__init__(images_dir, mask_dir, scale, mask_suffix='')





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



def inference(model,img_scale,val_percent,batch_size):

    # PATH = "/home/huyentn2/huyen/project/checkpoints/checkpoint_epoch5.pth"
    # model.load_state_dict(torch.load(PATH))

    try:
        dataset = PramDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # pdb.set_trace()
    val_score = evaluate(model, val_loader, device, amp=False)

    return val_score




if __name__ == "__main__":

    annotations_dir = "/home/huyentn2/huyen/project/data_nano/binary_mask/"
    img_dir = "/home/huyentn2/huyen/project/data_nano/img_patch/"

    dataset = PramDataset(img_dir, annotations_dir, scale = 1.0)
    train_dl = DataLoader(dataset, batch_size=64, shuffle=False)

    # for batch in train_dl:  
    #     # pdb.set_trace()
    #     sample_image = batch['image'][0]    
    #     sample_label = batch['mask'][0]
        

    arg_dict = {}
    arg_dict['classes'] = 2
    arg_dict['epochs'] =  15
    arg_dict['batch_size'] = 10
    arg_dict['learning_rate'] = 1e-5
    arg_dict['scale'] = 1
    arg_dict['val'] = 10.0
    arg_dict['amp'] = False
    arg_dict['bilinear'] = False


    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # logging.info(f'Using device {device}')


    model = UNet(n_channels=1, n_classes= arg_dict['classes'], bilinear=  arg_dict['bilinear'])
    

    PATH = "/home/huyentn2/huyen/project/checkpoints/checkpoint_epoch5.pth"

    # PATH = "/home/huyentn2/huyen/project/nano_direct_gt/checkpoints/checkpoint_epoch15.pth"
    #module??

    # model.load_state_dict(torch.load(PATH))
    checkpoint = torch.load(PATH)
    mask_values = checkpoint.pop('mask_values')
    model.load_state_dict(checkpoint)


    model = model.to(memory_format=torch.channels_last)
    model = torch.nn.DataParallel(model).cuda()

    # pdb.set_trace()

    model.to(device=device)
    train_model(
        model=model,
        epochs=arg_dict['epochs'],
        batch_size=arg_dict['batch_size'],
        learning_rate=arg_dict['learning_rate'],
        device=device,
        img_scale=arg_dict['scale'],
        val_percent=arg_dict['val'] / 100,
        amp=arg_dict['amp']
    )

    inference(model,arg_dict['scale'],val_percent=arg_dict['val'] / 100, batch_size=arg_dict['batch_size'])




