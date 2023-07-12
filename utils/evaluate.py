import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torch.nn.functional  as F
import matplotlib.pyplot as plt
import os
import sys        
sys.path.append('/home/huyentn2/huyen/project/project_nano_count/segmentation_unet/')
from dataset import BasicDataset, PramDataset  

from utils.dice_score import dice_coeff, multiclass_dice_coeff
import pdb
import numpy as np
import json

# with open(file, 'r') as f:
#     data = json.load(f)

dir_img = Path('/home/huyentn2/huyen/project/data_nano/img_patch/')
dir_mask = Path('/home/huyentn2/huyen/project/data_nano/binary_mask/')
dir_checkpoint = Path('/home/huyentn2/huyen/project/project_nano_count/segmentation_unet/checkpoints/')



def draw_pred(pred,mask, id):
    for i in range(10):
        img_rgb = pred[i][0].squeeze().cpu().numpy()
        img_rgb_true = mask[i][0].squeeze().cpu().numpy()
        Image.fromarray((img_rgb*255).astype('uint8')).save('/home/huyentn2/huyen/project/UNET/evaluate/Binary-Segmentation-Evaluation-Tool/test_data/rs1/{}.png'.format(id*10+i))
        Image.fromarray((img_rgb_true*255).astype('uint8')).save('/home/huyentn2/huyen/project/UNET/evaluate/Binary-Segmentation-Evaluation-Tool/test_data/gt/{}.png'.format(id*10+i))




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
            # f, axarr = plt.subplots(3,2,  figsize=(10, 10))
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


                # draw_pred(F.sigmoid(mask_hold),mask_true, id)
                id+=1

                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
            
    net.train()

    print("Dice:{}".format(dice_score / max(num_val_batches, 1)))
    
    return dice_score / max(num_val_batches, 1)



def inference(model,img_scale,val_percent,batch_size):  # infer the entire validation set of the dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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







@torch.inference_mode()
def evaluate_fold(net, dataloader, device, amp):

    assert net.module.n_classes == 1    
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    imgs_name = []
    gray_m = []
    bin_m = []
    inv_gray_mask = []

    # iterate over the validation set
    
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        it = 0 
        id = 0
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            # f, axarr = plt.subplots(3,2,  figsize=(10, 10))

            image, mask_true, names = batch['image'], batch['mask'], batch['name']
            imgs_name.extend(names)

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
                id+=1

                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
            
            for j in range(mask_pred.shape[0]):

                gray_m.append(F.sigmoid(mask_hold)[j,:][0].cpu().numpy())    # particle dark 
                bin_m.append(mask_pred[j,:][0].cpu().numpy())
                inv_gray_mask.append((F.sigmoid(mask_hold)[j,:][1].cpu().numpy()))  # particle bright
            
    net.train()

    print("Dice:{}".format(dice_score / max(num_val_batches, 1)))
    
    return dice_score / max(num_val_batches, 1) , imgs_name, gray_m, bin_m, inv_gray_mask





@torch.inference_mode()
def evaluate_fold_gauss(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)

    imgs_name = []
    gray_m = []
    count_pred = []

    err = np.array([], dtype=np.float32)

    true_count = np.array([], dtype=np.float32)

    # iterate over the validation set
    
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        it = 0 
        id = 0
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            # f, axarr = plt.subplots(3,2,  figsize=(10, 10))

            image, mask_true, names = batch['image'], batch['mask'], batch['name']
            imgs_name.extend(names)

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

            # predict the mask
            mask_pred = net(image)

            if net.module.n_classes == 1:
                err_batch = np.array((mask_true.sum(dim=-1).sum(dim=-1) - mask_pred.sum(dim=-1).sum(dim=-1)).squeeze(1).cpu())/255.0
                count = mask_pred.sum(dim=-1).sum(dim=-1).squeeze(1).cpu()/255.0
                true_c = mask_true.sum(dim=-1).sum(dim=-1).squeeze(1).cpu()/255.0
            
            elif net.module.n_classes > 1:
                err_batch = np.array((mask_true[:,0,:].sum(dim=-1).sum(dim=-1) - mask_pred[:,0,:].sum(dim=-1).sum(dim=-1)).cpu())/255.0
                count = mask_pred[:,0,:].sum(dim=-1).sum(dim=-1).cpu()/255.0
                true_c = mask_pred[:,0,:].sum(dim=-1).sum(dim=-1).cpu()/255.0

            for j in range(mask_pred.shape[0]):
                gray_m.append(np.asarray(mask_pred[j,0,:].cpu()))
                count_pred.append(count[j].cpu().numpy().item())
                err = np.concatenate((err, err_batch), axis=None)
                true_count = np.concatenate((true_count, true_c), axis=None)


    # get the metric
    # pdb.set_trace()
    err_mean = err.mean()
    err_std = err.std()
    abs_mean_err = abs(err).mean()

    standardized_err = err/ np.clip(true_count, 0.5, 10e5)   # in case no particle in the image, but unlikely
    ab_standardized_err = abs(err)/ np.clip(true_count, 0.5, 10e5)   # in case no particle in the image, but unlikely
    
    standardized_err_mean = standardized_err.mean()
    ab_standardized_err_mean = ab_standardized_err.mean()
    ab_standardized_err_std = ab_standardized_err.std()

    print(f"{'Validation metrics'}:\n"
      f"\tMean error: {err_mean:3.3f}\n"
      f"\tMean absolute error: {abs_mean_err:3.3f}\n"
      f"\tError deviation: {err_std:3.3f}\n"
      f"\tstandardized_err_mean: {standardized_err_mean:3.3f}\n"
      f"\tab_standardized_err_mean: {ab_standardized_err_mean:3.3f}\n"
      f"\tab_standardized_err_std: {ab_standardized_err_std:3.3f}\n"
      f"\tab_standardized_err_min: {np.min(ab_standardized_err):3.3f}\n")
    
    return abs_mean_err, imgs_name, gray_m, count_pred





def eval_fold_mode(save_dir, model, val_loader, k, device, use_gauss):

    if not use_gauss:

        val_score, imgs_name, gray_mask, bin_mask, inv_gray_mask = evaluate_fold(model, val_loader, device, amp=False)

        # pdb.set_trace()

        bin_dir = str(Path(save_dir)/ 'save_result'/ 'binary_pred' / 'fold{}'.format(k))
        gray_dir = str(Path(save_dir)/ 'save_result'/ 'gray_pred' / 'fold{}'.format(k))
        gray_dir_inv = str(Path(save_dir)/ 'save_result'/ 'gray_pred_inv' / 'fold{}'.format(k))

        if not os.path.exists(gray_dir):
            os.makedirs(gray_dir)

        if not os.path.exists(bin_dir):
            os.makedirs(bin_dir)

        if not os.path.exists(gray_dir_inv):
            os.makedirs(gray_dir_inv)

        for i,img in enumerate(imgs_name):
            # img_bin = bin_mask[i].squeeze(dim=0)[i,:].cpu().numpy()
            # img_gr = gray_mask[i].squeeze(dim=0)[i,:].cpu().numpy()
            img_bin = bin_mask[i]
            img_gr = gray_mask[i]
            img_gr_inv = inv_gray_mask[i]
            Image.fromarray((img_bin*255).astype('uint8')).save(bin_dir + "/" + img+ ".png")
            Image.fromarray((img_gr*255).astype('uint8')).save(gray_dir + "/" + img+ ".png")
            Image.fromarray((img_gr_inv*255).astype('uint8')).save(gray_dir_inv + "/" + img+ ".png")

        return val_score
    
    elif use_gauss:

        abs_mean_err, imgs_name, gray_m, count_pred = evaluate_fold_gauss(model, val_loader, device, amp=False)

        save_result_dict = {}

        gray_dir = str(Path(save_dir)/ 'save_result'/ 'mask_pred' / 'fold{}'.format(k))

        save_dict_dir = str(Path(save_dir)/ 'save_result'/ 'count_dict' / 'fold{}'.format(k))

        if not os.path.exists(gray_dir):
            os.makedirs(gray_dir)

        if not os.path.exists(save_dict_dir):
            os.makedirs(save_dict_dir)

        for i,img in enumerate(imgs_name):
            img_gray = gray_m[i]
            img_gray = (img_gray-np.min(img_gray)) / (np.max(img_gray) - np.min(img_gray))
            save_result_dict[img] = count_pred[i]
            
            Image.fromarray((img_gray*255).astype('uint8')).save(gray_dir + "/" + img+ ".png")

        file = save_dict_dir + '/count_prediction.json' 
        with open( file, 'w') as f: 
            json.dump(save_result_dict, f)
        return abs_mean_err



def get_num_NP(mask_pred, min_dist, max_fil_thr, bin_thr, range_rad, type_count):
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max

    def filter_count(im, min_dist, max_fil_thr):

        # image_max = ndi.maximum_filter(im, size=5, mode='constant')
        # Comparison between image_max and im to find the coordinates of local maxima

        coordinates = peak_local_max(im*255, min_distance=min_dist, threshold_abs=max_fil_thr)
        return len(coordinates)
    

    def direct_count(im,  range):
        # import pdb
        # pdb.set_trace()
        range_flg = True if type(range) is tuple else False
        try:
            min_r, max_r = range[0], range[1]
        except:
            print("range should be a tuple")
            range_flg = False
        if range_flg:
            im = limit_NP(im, min_r, max_r)
        im = ndi.gaussian_filter(im, 2)
        # if thrh:
        # im[im<10]=0
        blobs, number_of_blobs = ndi.label(im)
        # print(number_of_blobs)
        return number_of_blobs, blobs
    

    def limit_NP(im, min_r, max_r):
        import numpy as np

        blobs, number_of_blobs = ndi.label(im)
        mask = np.zeros((blobs.shape))
        # np.unique(blobs)
        # mask_pred_1.sum()/ len(np.unique(blobs))
        l = []
        for i in range(1,number_of_blobs):    
            l.append((blobs== i).sum())

        mean_area = np.mean(np.asarray(l))
        std_area = np.std(np.asarray(l))
        s = 1   #scale_factor to ommit out-of-distribution NP, too big radius

        for i in range(1,number_of_blobs):    
            reg = blobs== i #np.where(blobs== i, 1, 0)
            area = reg.sum()
            if area > min_r and area < max_r:
                # import pdb
                # pdb.set_trace()
                mask[blobs== i] = im[blobs== i]
                # if area > mean_area- s*std_area and area < mean_area + s*std_area:
                #     mask[blobs== i] = 1
        return mask


    mask_pred_0 = (1-F.sigmoid(mask_pred)).squeeze()[0,:].float()
    mask_pred_00 = torch.where(mask_pred_0> bin_thr, 1, 0).numpy()
    mask_pred_0 = mask_pred_0.numpy()

    mask_pred_1 = F.sigmoid(mask_pred).squeeze()[1,:].float()
    mask_pred_11 = torch.where(mask_pred_1> bin_thr, 1, 0).numpy()
    mask_pred_1 = mask_pred_1.numpy()

    mask_soft = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float().squeeze().numpy()

    # import pdb
    # pdb.set_trace()
    if type_count =='0':
        count_0 = filter_count((mask_pred_0), min_dist, max_fil_thr)
        return count_0

    if type_count =='1':
        # pdb.set_trace()
        count_1 = filter_count(mask_pred_1, min_dist, max_fil_thr)
        return count_1
    
    # BINARIZE
    if type_count =='2':
        # pdb.set_trace()
        count_2, blobs = direct_count(mask_soft[1,:], range_rad)
        return count_2, blobs
        # return count_2
    
    # BINARIZE
    if type_count =='3':
        count_3, blobs = direct_count((mask_pred_00)*255, range_rad)
        return count_3

    if type_count =='4':
        count_4, blobs = direct_count(mask_pred_11*255, range_rad)
        return count_4
    





@torch.inference_mode()
def evaluate_gauss(net, dataloader, device, amp):
    criterion = torch.nn.MSELoss(reduction='mean')
    net.eval()
    num_val_batches = len(dataloader)
    # dice_score = 0
    loss_batch = 0

    # iterate over the validation set

    err = np.array([], dtype=np.float32)
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        # it = 0 
        # id = 0
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            # f, axarr = plt.subplots(3,2,  figsize=(10, 10))
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

            # predict the mask
            mask_pred = net(image)
            loss = criterion(mask_pred, mask_true)
            loss_batch += loss
            if net.module.n_classes == 1:
                err_batch = np.array((mask_true.sum(dim=-1).sum(dim=-1) - mask_pred.sum(dim=-1).sum(dim=-1)).squeeze(1).cpu())/255.0
                err = np.concatenate((err, err_batch), axis=None)

    net.train()

    # pdb.set_trace()
    err_mean = err.mean()
    err_std = err.std()
    abs_meab_err = abs(err).mean()
    
    return err_mean, err_std, abs_meab_err, loss_batch/ (err.shape[0])