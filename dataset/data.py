import numpy as np
import logging
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import torch
from scipy.ndimage import gaussian_filter
import pdb

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

        # import pdb
        # pdb.set_trace()

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
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'name': name
        }


class PramDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        # super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
        super().__init__(images_dir, mask_dir, scale, mask_suffix='')



# class Gaussian_Masks_Dataset(BasicDataset_):
#     def __init__(self, images_dir, gauss_dir, scale=1):
#         # super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
#         super().__init__(images_dir, gauss_dir, scale, mask_suffix='')
#         self.mask_values = None


class Gaussian_Masks_Dataset(Dataset):
    def __init__(self, Sigma: float,  images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.sigma = Sigma

        # pdb.set_trace()

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        # logging.info('Scanning mask files to determine unique values')
        # with Pool() as p:
        #     unique = list(tqdm(
        #         p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
        #         total=len(self.ids)
        #     ))

        # pdb.set_trace()
        # self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        # logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, sigma, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH))
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        # pdb.set_trace()

        if is_mask:
            if img.ndim == 2:
                img = img[np.newaxis, ...]   #  img = (24,24)   -> img = (1,24,24)   append new dim at the beginning
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            # print(img.shape)
            # print(img)
                mask = gaussian_filter(img, sigma=sigma)
                return mask*255.0

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]   #  img = (24,24)   -> img = (1,24,24)   append new dim at the beginning
            else:
                img = img.transpose((2, 0, 1))
            if (img > 1).any():
                img = img / 255.0
                img = (img -  np.min(img)) /(np.max(img) - np.min(img))

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

        # pdb.set_trace()
        img = self.preprocess(img, self.sigma, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.sigma, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous(),
            'name': name
        }
    