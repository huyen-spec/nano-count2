
# from dataset.data import load_image
# import dataset as dat
from dataset import *
from utils import *


# dats = BasicDataset()


# model = UNet()
# train_model()
# inference()

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
# import wandb
import logging
import wandb
import argparse
import pdb



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, dest='seed', default=10)
    parser.add_argument('--classes', type = int, dest='classes', default=2)
    parser.add_argument('--epochs', type = int, dest='epochs', default=15)
    parser.add_argument('--batch_size', type = int, dest='batch_size', default=4)
    parser.add_argument('--learning_rate', type = float,  dest='learning_rate', default=1e-5)
    parser.add_argument('--scale', type = float,  dest='scale', default=1.0)
    parser.add_argument('--val', type = float,  dest='val', default=10.0)
    parser.add_argument('--amp',  dest='amp', action='store_true')
    parser.add_argument('--bilinear',  dest='bilinear', action='store_true')
    parser.add_argument('--use_gauss', dest='use_gauss', action='store_true')
    parser.add_argument('--kfold', dest='train_fold', action='store_true')
    parser.add_argument('--num_fold', type = int, dest='num_fold', default=10)
    parser.add_argument('--eval', dest='eval', action='store_true')
    parser.add_argument('--filter_sigma', type = float, dest='filter_sigma', default=3.0)

    args = parser.parse_args()


    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)


    # pdb.set_trace()


    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # save_dir = "save_path/"  # evalution

    save_dir = '/home/huyentn2/project/nano_count/segmentation_unet/checkpoints_27_4/'

    # print out wrong cases, look at ab_standardized_err_std value
    # do validation, now just train and test
    # prediction is sometimes negative??

    for n_units in range(args.num_fold):    # 10 folds
        if not args.eval:
            experiment = wandb.init(project="U-Net-28-4", config={"n_units": n_units}, reinit=True, anonymous='must')

        if args.use_gauss:
            model = UNet(n_channels=1, n_classes= 1, bilinear=  args.bilinear)
            # model = UNet2(input_filters=1,
            #                 filters=64,
            #                 N=2)

        else:
            model = UNet(n_channels=1, n_classes= args.classes, bilinear=  args.bilinear)

        model = model.to(memory_format=torch.channels_last)
        model = torch.nn.DataParallel(model).cuda()

        if not args.eval:
            model.to(device=device)
            train_kfold_model(
                        save_dir,
                        num_fold=args.num_fold,
                        eval=args.eval, 
                        n_units = n_units,
                        model=model,
                        experiment = experiment,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        device=device,
                        img_scale=args.scale,
                        val_percent=args.val / 100,
                        amp=args.amp,
                        use_gauss= args.use_gauss,
                        sigma = args.filter_sigma,
                        )
            
        elif args.eval:
            if not args.use_gauss:
                checkpoint = torch.load("/home/huyentn2/project/nano_count/segmentation_unet/checkpoints_7_4/fold9/checkpoint_epoch15.pth")

                # checkpoint = torch.load("/home/huyentn2/project/nano_count/segmentation_unet/all_checkpoints/checkpoints/fold1/checkpoint_epoch20.pth")

                # mask_values = checkpoint.pop('mask_values')

                model.load_state_dict(checkpoint)
                model.to(device=device)

                # get output of one single image at a time 

                # input_dir = "/home/huyentn2/project/nano_count/segmentation_unet/data/img_patch/Tianle1ng_0208_3.png"
                # save_single_dir = "/home/huyentn2/project/nano_count/segmentation_unet"
                # eval_single(input_dir, save_single_dir, model, device, args.scale)


                # mask_pred = eval_single_count(input_dir,
                #             save_dir,
                #             model,
                #             device,
                #             img_scale=1.0,
                #             )
                # min_dist = 10
                # max_fil_thr = 120
                # bin_thr = 0.5
                # range_rad = (10,200)
                # type_count = "4"
                # print(get_num_NP(mask_pred, min_dist, max_fil_thr, bin_thr,  range_rad, type_count))

                # pdb.set_trace()


                # doing batch eval 
                train_kfold_model(
                            save_dir,
                            num_fold=args.num_fold,
                            eval=args.eval, 
                            n_units = n_units,
                            model=model,
                            experiment = None,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            learning_rate=args.learning_rate,
                            device=device,
                            img_scale=args.scale,
                            val_percent=args.val / 100,
                            amp=args.amp,
                            use_gauss= args.use_gauss,
                            sigma = args.filter_sigma,
                            )
                
            if args.use_gauss:

                checkpoint = torch.load("/home/huyentn2/project/nano_count/segmentation_unet/checkpoints_27_4/fold8/checkpoint_epoch15.pth")
                model.load_state_dict(checkpoint)
                model.to(device=device)              
                

                train_kfold_model(
                            save_dir,
                            num_fold=args.num_fold,
                            eval=args.eval, 
                            n_units = n_units,
                            model=model,
                            experiment = None,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            learning_rate=args.learning_rate,
                            device=device,
                            img_scale=args.scale,
                            val_percent=args.val / 100,
                            amp=args.amp,
                            use_gauss= args.use_gauss,
                            sigma = args.filter_sigma,
                            )





















































    # for n_units in [0, 1, 2, 3, 4]:
    #     experiment = wandb.init(project="'U-Net", config={"n_units": n_units}, reinit=True, anonymous='must')

    #     model = UNet(n_channels=1, n_classes= args.classes, bilinear=  args.bilinear)
    #     model = model.to(memory_format=torch.channels_last)
    #     model = torch.nn.DataParallel(model).cuda()


    #     # pdb.set_trace()
    #     model.to(device=device)
    #     # fold = True
    #     if args.kfold:
    #         train_kfold_model(
    #                     n_units = n_units,
    #                     model=model,
    #                     experiment = experiment,
    #                     epochs=args.epochs,
    #                     batch_size=args.batch_size,
    #                     learning_rate=args.learning_rate,
    #                     device=device,
    #                     img_scale=args.scale,
    #                     val_percent=args.val / 100,
    #                     amp=args.amp,
    #                     use_gauss = args.use_gauss,
    #                     )

    # else:
    # train_model(
    #     model=model,
    #     epochs=arg_dict['epochs'],
    #     batch_size=arg_dict['batch_size'],
    #     learning_rate=arg_dict['learning_rate'],
    #     device=device,
    #     img_scale=arg_dict['scale'],
    #     val_percent=arg_dict['val'] / 100,
    #     amp=arg_dict['amp']
    # )

    # inference(model,arg_dict['scale'],val_percent=arg_dict['val'] / 100, batch_size=arg_dict['batch_size'])




