
# from dataset.data import load_image
# import dataset as dat
from dataset import *
from utils import *


# dats = BasicDataset()


# model = UNet()
# train_model()
# inference()

import torch
from torch.utils.data import DataLoader, random_split
# import wandb
import logging
import wandb




# # img_rgb = mask_pred[0][0].squeeze().cpu().numpy()
# # PIL.Image.fromarray((img_rgb*255).astype('uint8')).save('/home/huyentn2/huyen/project/save1.png')
# # /home/huyentn2/huyen/project/UNET/evaluate


# # save_mask = Path()
# # save_img = Path()



if __name__ == "__main__":

    # annotations_dir = "/home/huyentn2/project/nano_count/segmentation_unet/data/binary_mask/"
    # img_dir = "/home/huyentn2/project/nano_count/segmentation_unet/data/img_patch/"

    # dataset = PramDataset(img_dir, annotations_dir, scale = 1.0)
    # train_dl = DataLoader(dataset, batch_size=64, shuffle=False)

    # for batch in train_dl:  
    #     # pdb.set_trace()
    #     sample_image = batch['image'][0]    
    #     sample_label = batch['mask'][0]
        

    arg_dict = {}
    arg_dict['classes'] = 2
    arg_dict['epochs'] =  3
    arg_dict['batch_size'] = 6
    arg_dict['learning_rate'] = 1e-5
    arg_dict['scale'] = 1
    arg_dict['val'] = 10.0
    arg_dict['amp'] = False
    arg_dict['bilinear'] = False


    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # logging.info(f'Using device {device}')


    model = UNet(n_channels=1, n_classes= arg_dict['classes'], bilinear=  arg_dict['bilinear'])
    

    # PATH = "/home/huyentn2/huyen/project/checkpoints/checkpoint_epoch5.pth"

    # PATH = "/home/huyentn2/huyen/project/nano_direct_gt/checkpoints/checkpoint_epoch15.pth"
    #module??

    # model.load_state_dict(torch.load(PATH))
    # checkpoint = torch.load(PATH)
    # mask_values = checkpoint.pop('mask_values')
    # model.load_state_dict(checkpoint)


    model = model.to(memory_format=torch.channels_last)
    model = torch.nn.DataParallel(model).cuda()

    # pdb.set_trace()
    # k in range(1,2,3,4) passed by arg parse
    experiment = wandb.init(project='U-Net-{}-fold'.format(k), resume='allow', anonymous='must')

    model.to(device=device)
    fold = True
    if fold:
        train_kfold_model(
                    model=model,
                    experiment = experiment,
                    epochs=arg_dict['epochs'],
                    batch_size=arg_dict['batch_size'],
                    learning_rate=arg_dict['learning_rate'],
                    device=device,
                    img_scale=arg_dict['scale'],
                    val_percent=arg_dict['val'] / 100,
                    amp=arg_dict['amp'],
                    )


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

    # inference(model,arg_dict['scale'],val_percent=arg_dict['val'] / 100, batch_size=arg_dict['batch_size'])




