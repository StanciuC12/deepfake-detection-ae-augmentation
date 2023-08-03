import torch
import time
import os
import copy
from torchvision import transforms
import torch.nn as nn
import numpy as np
from util import AverageMeter
from sklearn.metrics import roc_auc_score
import pandas as pd
from capsulenet import CapsuleNet, CapsuleLoss
from data_loader_pictures_fullvideo import DeepFakeDataset
from cnn import CNN
import torchvision.models as models
# import torchfunc
#
import sys
sys.path.insert(0, r"C:\Users\Crispy\PycharmProjects\AdversarialAttacks2")
import torchattacks
from robustbench.utils import load_model, clean_accuracy

torch.autograd.set_detect_anomaly(True)


test_setup = pd.DataFrame(columns=['dataset_adr', 'train_file_path', 'dataset'])
test_setup.loc[0] = [r'E:\ff++\saved_images', r'train_test_split.xlsx', 'FF++']
test_setup.loc[1] = [r'D:\saved_celefdf_all', r'train_test_celebdf_corect.xlsx', 'celebDF']
test_setup.loc[2] = [r'D:\saved_img', r'test_train_dfdc_final.xlsx', 'dfdc']

modelses = {'orig_data': r'D:\saved_model\Xception_orig_data__fullface_epoch_14_param_FF++_83_2023.pkl',
            'augm': r'D:\saved_model\Xception_augm_fullface_epoch_103_param_FF++_33_1636.pkl',
            'AE': r'D:\saved_model\Xception_fullface_epoch_77_param_FF++_33_1446.pkl'}


########################################################################################################################
# adversarial attacks



########################################################################################################################






########################################################################################################################
# COMPRESS
#
# from PIL import Image
#
# def compress(filepath, format='JPEG', quality=10, name=''):
#
#     image = Image.open(filepath)
#
#     image.save(filepath.split('.')[0] + f'{name}{format}_q{quality}.{format}',
#                format,
#                optimize=True,
#                quality=quality)
#
#     return
#
# def lst(x):
#
#     try:
#         return [x for x in os.listdir(x) if '_compressed_' not in x and 'fullface' in x][2:18]
#     except:
#         return pd.NA
#
# for idx in [2, 1, 0]:
#     print(idx)
#
#     df = pd.read_excel(os.path.join(test_setup.loc[idx, 'dataset_adr'], test_setup.loc[idx, 'train_file_path']))
#     df = df.loc[df['TrainTest'] == 1].reset_index()
#     if idx == 2:
#         df['VideoName'] = df['VideoName'].apply(lambda x: x.split('.')[0])
#     df['VideoName'] = df['VideoName'].apply(lambda x: os.path.join(test_setup.loc[idx, 'dataset_adr'], x))
#     df['images'] = df['VideoName'].apply(lambda x: lst(x))
#     df = df.loc[df['images'].notna()]
#     df = df.loc[df['images'].apply(lambda x: len(x) > 0)]
#     df = df.reset_index()
#
#     for quality in [10, 20, 30, 50, 80]:
#         format = 'JPEG'
#         name = '_compressed_V2_'
#
#         print('Compressing...')
#         for i in range(len(df)):
#
#             images = df.loc[i, 'images']
#             addr = df.loc[i, 'VideoName']
#             print(f'{i}/{len(df)} - {addr}')
#
#             for image in images:
#                 compress(os.path.join(addr, image), format, quality, name)




        # TODO: delete _compressed_


########################################################################################################################





#transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(1), saturation=(0.5, 1.5), hue=(-0.1, 0.1)).__class__

# transf_appl_vect = [transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(1), saturation=(0.5, 1.5), hue=(-0.1, 0.1)),
#                transforms.RandomRotation((10, 30)),
#                transforms.RandomAffine(degrees=(10, 40), translate=(0.1, 0.3), scale=(0.5, 1)),
#                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(20, 25)),
#                transforms.RandomResizedCrop(size=(299, 299), ratio=(0.5, 0.9)),
#                transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
#                transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
#                transforms.RandomAutocontrast(p=1),
#                transforms.RandomAdjustSharpness(sharpness_factor=2, p=1),
#                ]
#
# for transf_appl in transf_appl_vect[::-1]:
#
#     transf_name = str(transf_appl.__class__).split('.')[-1][0:-2]


for ji in range(3):

    # dataset_adr = r'D:\saved_celefdf_all' # r'E:\saved_img'
    # train_file_path = r'train_test_celebdf_corect.xlsx'
    # dataset = 'celebDF'

    dataset_adr = test_setup.loc[ji, 'dataset_adr']
    train_file_path = test_setup.loc[ji, 'train_file_path']
    dataset = test_setup.loc[ji, 'dataset']

    img_type = 'fullface'
    dataset_model = 'FF++'
    model_type = 'Xception'
    test_batch_size = 1

    for m in modelses:
        model_param_adr = modelses[m] #r'E:\saved_model\capsule_features_fullface_epoch_7_param_celebDF_172_1647.pkl'

        print('Model:', m)
        transf = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(1), saturation=(0.5, 1.5), hue=(-0.1, 0.1)),
            #transforms.RandomRotation((0, 30)),
            #transforms.RandomAffine(degrees=(10, 40), translate=(0.1, 0.3), scale=(0.5, 1)),
            #transforms.GaussianBlur(kernel_size=(3, 3), sigma=(20, 25)),
            #transf_appl,
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            # Resnet and VGG19 expects to have data normalized this way (because pretrained)
        ])

        print('Getting images...')
        # getting ALL images
        data_test = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf,
                                    batch_size=16, train=False, image_type=img_type, dataset=dataset,
                                    images_per_folder=4, image_doesnt_contain='compressed', labels_repeat=True)
        data_test.shuffle(random_state=1)
        # img, labels = data_test[0]
        # labels = labels.tolist()
        # i_not16 = []
        # for i in range(len(img)):
        #     if(len(img[i]) != 16):
        #         i_not16.append(i - len(i_not16))
        #
        # for i in i_not16:
        #     del img[i]
        #     del labels[i]
        #
        # img_stack = torch.stack([torch.stack(x) for x in img])
        # labels = torch.Tensor(labels)
        #
        # del img
        #
        # labels_all = labels.repeat_interleave(16)
        # img_all = img_stack.reshape(-1, 3, 299, 299)
        #
        # del img_stack
#######################################################################################################################


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Device: ', device)

        #model = CapsuleNet(architecture=model_type, dataset=dataset_model)
        model = CNN(pretrained=True, finetuning=True, frozen_params=0, architecture='Xception')
        model.to(device)
        ########################################################
        # model = models.efficientnet_b4(weights=True)
        # for params in model.parameters():
        #     params.requires_grad = True
        # model.classifier[1] = nn.Sequential(nn.Linear(in_features=1792, out_features=1), nn.Sigmoid())
        # model.to(device)

        ##########################################################
        print("# parameters:", sum(param.numel() for param in model.parameters()))

        epoch_done = None
        if model_param_adr:
            model.load_state_dict(torch.load(model_param_adr))
            epoch_done = int(model_param_adr.split('_')[-5])

        params = list(model.parameters())

        criterion = nn.BCELoss()
        ###########################################
        # from timeCaps.ae import UNet
        #
        # model_redo = UNet(n_channels=3, n_classes=3)
        # model_redo.to('cuda')
        # model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_unet_fullface_epoch_6_param_all_232_1143.pkl'))

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        denorm = transforms.Normalize(
                mean=[-m / s for m, s in zip(mean, std)],
                std=[1.0 / s for s in std],
                # always_apply=True,
                # max_pixel_value=1.0
        )
        transf_orig = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ##############################################


        #########################################################################################################
        batch_size = 128

        resnet_adr = r'D:\\saved_model\\ResNet\\ff\\2\\resnet-50_fullface_epoch_5_param_FF_276_1028.pkl'
        resnet = CNN(pretrained=True, finetuning=False, frozen_params=100, architecture='resnet-50')
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.to(device)
        resnet.load_state_dict(torch.load(resnet_adr))


        print('ADVERSARIALES...')
        attack = torchattacks.PGD(resnet, eps=8 / 255, alpha=2 / 255, steps=4)  #model1
        attack.set_normalization_used(mean=mean, std=std)
        save = f"{ji}_resnet_transfer.pt"
        if save not in os.listdir():
            attack.save(data_loader=data_test, save_path=save)
            print('ATTACK \n', attack)

        print('LOADING...')
        adv_loader = attack.load(load_path=f"{ji}_resnet_transfer.pt")
        print('LOADED')
        dataiter = iter(adv_loader)
        adv_images, labels = next(dataiter)


        acc = clean_accuracy(model, adv_images.to(device), labels.to(device))
        print('Acc: %2.2f %%' % (acc * 100))


        #########################################################################################################

        # Testing

        print('Starting TEST')
        test_losses = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        times = AverageMeter()
        test_predictions_vect = []
        test_targets_vect = []
        data_test.shuffle(random_state=1)
        test_df = copy.copy(data_test.label_df)
        model.eval()
        t_start = time.time()

        adv_images = []
        labels = []
        for datalen in range(len(dataiter)-1):

            adv, lab = next(dataiter)
            adv_images.append(adv)
            labels.append(lab)

        labels = torch.concat(labels)
        adv_images = torch.concat(adv_images)
        print(len(labels)//4, 'VIDEOS')
        for i in range(len(adv_images) // 16):

            t1 = time.time()
            print(f'{i}/{len(adv_images) // 16}')
            # data, targets = data_test[i]
            # data, targets = data.to(device), targets.to(device)
            data, targets = adv_images[i*16:i*16 + 16], labels[i*16:i*16 + 16]
            targets = targets[0::4]
            data, targets = data.to(device), targets.to(device)
            outputs_total = []
            loss_total = []

            with torch.no_grad():
                try:
                    outputs_gpu = model(data)
                    outputs = outputs_gpu.to('cpu').flatten()
                except Exception as e:
                    print(e)
                    print(f'Failed in test i={i}')
                    continue


            # test_losses.update(loss.item(), data.size(0))
            outputs = outputs.reshape([4, 4])
            test_predictions_vect.append(outputs.mean(axis=1))
            test_targets_vect.append(targets.to('cpu').detach())

            if len(test_predictions_vect) > 1:
                # print(f'Predictions: {test_predictions_vect}')
                # print(f'Targets: {test_targets_vect}')

                try:
                    loss = criterion(torch.stack(test_predictions_vect).flatten().type(torch.DoubleTensor), torch.stack(test_targets_vect).flatten().type(torch.DoubleTensor))
                    print(f'Loss: {loss}')
                except Exception as e:
                    print(e)

            try:
                if len(torch.unique(torch.cat(test_targets_vect).flatten())) > 1:
                        auc_test = roc_auc_score(torch.stack(test_targets_vect).flatten().type(torch.DoubleTensor), torch.stack(test_predictions_vect).flatten().type(torch.DoubleTensor))
                else:
                        auc_test = '-'
            except Exception as e:
                print(e)
                auc_test = 'failed'

            print(f'AUC: {auc_test}')

            t2 = time.time()
            duration_1_sample = t2 - t1
            times.update(duration_1_sample, 1)
            print('Est time/Epoch: ' + str(int(times.avg * (len(data_test)-i) // 3600)) + 'h' +
                              str(int((times.avg * (len(data_test)-i) - 3600 * (times.avg * (len(data_test)-i) // 3600)) // 60)) + 'm')


        print('\n\n\n' + 'ADVERSARIAL_model1' + ' ' + dataset + ' ' + m +
                '\n' + model_param_adr +
                '\n========================================================' +
              '\n' + img_type +
              '\n TEST Epoch: ' + str(epoch_done) + '\n TEST Loss: ' + str(loss)  +
              '\n TEST AUC total: ' + str(auc_test) +
              '\n========================================================\n')

        f = open(r"D:\saved_model\test_results_" + model_type + '_' + img_type + '_' + dataset + ".txt", "a")
        f.write('\n\n\n' + 'ADVERSARIAL_model1' + ' ' + dataset + ' Model:' + m +
                '\n' + model_param_adr +
                '\n========================================================' +
              '\n' + img_type +
              '\n TEST Epoch: ' + str(epoch_done) + '\n TEST Loss: ' + str(loss)  +
              '\n TEST AUC total: ' + str(auc_test) +
              '\n========================================================\n')
        f.close()

        # try:
        #     test_df['GT'] = torch.stack(test_targets_vect).flatten().numpy()
        #     test_df['Pred'] = torch.stack(test_predictions_vect).flatten().numpy()
        #     test_df.to_excel(r'D:\saved_model\outputs_test_test_' + model_type + '_' + img_type + '_epoch_' + str(epoch_done) + '_' + dataset + '_' +
        #                      str(time.gmtime()[2]) + str(time.gmtime()[1]) + '_' + str(time.gmtime()[3]) + str(time.gmtime()[4]) + '.xlsx')
        #
        # except:
        #     print('NU A MERS EXCELU PT TEST')




