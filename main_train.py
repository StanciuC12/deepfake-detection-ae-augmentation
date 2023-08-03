import random
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
from data_loader_pictures import DeepFakeDataset
from capsulenet import CapsuleNet, CapsuleLoss
from cnn import CNN
import torchfunc
import torchvision.models as models

torchfunc.cuda.reset()


# dataset_adr = r'E:\saved_img'
# train_file_path = r'train_test_combined.xlsx'

dataset_adr = r'E:\ff++\saved_images' # r'E:\saved_img'
train_file_path = r'train_test_split.xlsx'
img_type = 'fullface'
dataset = 'FF++'
model_suffix = '_RESNET_attacks_'
normal_data = True
# dataset_adr = r'D:\saved_celefdf_all' # r'E:\saved_img'
# train_file_path = r'train_test_celebdf_corect.xlsx'
# img_type = 'fullface'
# dataset = 'celebDF'


model_type = 'resnet-50' #'capsule_features'
######################
lr = 1e-3
#####################
weight_decay = 0 #0.00002 #0.2
nr_epochs = 10
lr_decay = 0.9
test_data_frequency = 1
train_batch_size = 7
test_batch_size = 8
gradient_clipping_value = None
model_param_adr = None #r'D:\saved_model\Xception_fullface_epoch_77_param_FF++_33_1446.pkl'    # None if new training

transf = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #transforms.ColorJitter(hue=0.05, saturation=0.05, brightness=0.2),
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # Resnet and VGG19 expects to have data normalized this way (because pretrained)
])

if model_type == 'capsule_features':
    from data_loader_pictures import DeepFakeDataset
elif model_type == 'capsule-LSTM':
    from data_loader import DeepFakeDataset


data_train = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf,
                             batch_size=train_batch_size, train=True, image_type=img_type, dataset=dataset)
data_test = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf,
                            batch_size=test_batch_size, train=True, image_type=img_type, dataset=dataset)

# classid = 1
# data_train.label_df = data_train.label_df.loc[data_train.label_df['ClassId_multiple'] != classid, :]
# data_test.label_df = data_test.label_df.loc[data_test.label_df['ClassId_multiple'] == classid, :]
# data_test.label_df = data_test.label_df.reset_index()
# data_train.label_df = data_train.label_df.reset_index()
# model_suffix = model_suffix + str(classid)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

#model = CapsuleNet(architecture=model_type, dataset=dataset)
model = CNN(pretrained=True, finetuning=True, frozen_params=0, architecture='resnet-50')
model.to(device)

# model = models.efficientnet_b4(weights=True)
# for params in model.parameters():
#     params.requires_grad = True
# model.classifier[1] = nn.Sequential(nn.Linear(in_features=1792, out_features=1), nn.Sigmoid())
# model.to(device)

print("# parameters:", sum(param.numel() for param in model.parameters()))


epoch_done = 0
if model_param_adr:
        model.load_state_dict(torch.load(model_param_adr))
        epoch_done = int(model_param_adr.split('_')[-5])  # Number of epoch done is always in that position by convention


params = list(model.parameters())
optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
#optimizer = torch.optim.SGD(params, lr=lr)

# scheduler = lr_scheduler.ReduceLROnPlateau(
# 	optimizer, 'min', patience=opt.lr_patience)

#criterion = CapsuleLoss()
criterion = nn.BCELoss()

# Adding model!!!#############################################
from ae import UNet, AE1, AE2

models = []

model_redo = UNet(n_channels=3, n_classes=3)
model_redo.to(device)
model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_unet_fullface_epoch_4_param_all_222_1939.pkl'))
models.append(model_redo)
#
# model_redo = UNet(n_channels=3, n_classes=3)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_unet_fullface_epoch_8_param_all_222_175.pkl'))
# models.append(model_redo)
#
# model_redo = UNet(n_channels=3, n_classes=3, bilinear=True, model='small')
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_unet_bilinear_big_fullface_epoch_5_param_all_262_2135.pkl'))
# models.append(model_redo)
#
# model_redo = UNet(n_channels=3, n_classes=3, bilinear=True, model='big')
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_unet_bilinear_big_fullface_epoch_2_param_all_262_208.pkl'))
# models.append(model_redo)
#
# model_redo = AE2(n_channels=3, n_classes=3)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE2_fullface_epoch_3_param_all_262_1727.pkl'))
# models.append(model_redo)
#
# model_redo = AE2(n_channels=3, n_classes=3)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE2_fullface_epoch_2_param_all_262_1630.pkl'))
# models.append(model_redo)
#
#
# model_redo = AE1(n_channels=3, n_classes=3)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_unet_fullface_epoch_5_param_all_232_2359.pkl'))
# models.append(model_redo)
#
#
# model_redo = AE1(n_channels=3, n_classes=3)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_unet_fullface_epoch_3_param_all_232_2223.pkl'))
# models.append(model_redo)
#
# from timeCaps.ae_models import U_Net, AttU_Net, NestedUNet, Unet_dict, R2AttU_Net, R2U_Net
#
# model_redo = U_Net()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_U_Net_l1__fullface_epoch_2_param_all_13_1817.pkl'))
# models.append(model_redo)
#
# model_redo = U_Net()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_U_Net_l1__fullface_epoch_2_param_all_23_230.pkl'))
# models.append(model_redo)
#
# model_redo = R2U_Net()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_R2U_Net_l1__fullface_epoch_2_param_all_23_332.pkl'))
# models.append(model_redo)
#
# model_redo = R2U_Net()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_R2U_Net_l1__fullface_epoch_2_param_all_13_1943.pkl'))
# models.append(model_redo)
#
# model_redo = AttU_Net()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_AttU_Net_l1__fullface_epoch_2_param_all_23_49.pkl'))
# models.append(model_redo)
#
# model_redo = AttU_Net()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_AttU_Net_l1__fullface_epoch_2_param_all_13_2319.pkl'))
# models.append(model_redo)
#
# model_redo = R2AttU_Net()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_R2AttU_Net_l1__fullface_epoch_2_param_all_23_026.pkl'))
# models.append(model_redo)
#
# model_redo = R2AttU_Net()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_R2AttU_Net_l1__fullface_epoch_2_param_all_23_514.pkl'))
# models.append(model_redo)
#
# model_redo = NestedUNet()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_NestedUNet_l1__fullface_epoch_2_param_all_23_141.pkl'))
# models.append(model_redo)
#
# model_redo = NestedUNet()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_NestedUNet_l1__fullface_epoch_2_param_all_23_627.pkl'))
# models.append(model_redo)
#
# model_redo = Unet_dict(n_labels=3)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_Unet_dict_l1__fullface_epoch_2_param_all_23_642.pkl'))
# models.append(model_redo)
#
# model_redo = Unet_dict(n_labels=3)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_Unet_dict_l1__fullface_epoch_2_param_all_23_156.pkl'))
# models.append(model_redo)
#
# from timeCaps.double_unet import Double_UNet
#
# model_redo = Double_UNet(output=2)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_unet_double_d2_l1_fullface_epoch_2_param_all_282_172.pkl'))
# models.append(model_redo)
#
# model_redo = Double_UNet(output=1)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_double_d1d2_l1__fullface_epoch_2_param_all_282_219.pkl'))
# models.append(model_redo)
#
# model_redo = Double_UNet(output=2)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_double_d1d2_l1__fullface_epoch_2_param_all_282_219.pkl'))
# models.append(model_redo)
#
# model_redo = Double_UNet(output=1)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_double_d1d2_mse__fullface_epoch_2_param_all_13_519.pkl'))
# models.append(model_redo)
#
# model_redo = Double_UNet(output=2)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_double_d1d2_mse__fullface_epoch_2_param_all_13_519.pkl'))
# models.append(model_redo)
#
# model_redo = Double_UNet(output=1)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_double_d1d2_mse_quarterdata__fullface_epoch_2_param_all_13_852.pkl'))
# models.append(model_redo)
#
# model_redo = Double_UNet(output=2)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_double_d1d2_mse_quarterdata__fullface_epoch_2_param_all_13_852.pkl'))
# models.append(model_redo)
#
# model_redo = Double_UNet(output=1)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_double_d1d2_l1_quarterdata__fullface_epoch_2_param_all_13_1227.pkl'))
# models.append(model_redo)
#
# model_redo = Double_UNet(output=2)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_double_d1d2_l1_quarterdata__fullface_epoch_2_param_all_13_1227.pkl'))
# models.append(model_redo)
#
# #AICI
# ####################################
# model_redo = Double_UNet(output=2)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_double_d1d2_l1_quarterdata__fullface_epoch_1_param_all_13_1158.pkl'))
# models.append(model_redo)
#
# model_redo = Double_UNet(output=1)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_double_d1d2_l1_quarterdata__fullface_epoch_1_param_all_13_1158.pkl'))
# models.append(model_redo)
#
# model_redo = Double_UNet(output=1)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_double_d1d2_mse_quarterdata__fullface_epoch_1_param_all_13_824.pkl'))
# models.append(model_redo)
#
# model_redo = Double_UNet(output=2)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_double_d1d2_mse_quarterdata__fullface_epoch_1_param_all_13_824.pkl'))
# models.append(model_redo)
#
# model_redo = Double_UNet(output=2)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_unet_double_d2_l1_fullface_epoch_1_param_all_282_1454.pkl'))
# models.append(model_redo)
#
# model_redo = Double_UNet(output=1)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_double_d1d2_l1__fullface_epoch_1_param_all_282_1915.pkl'))
# models.append(model_redo)
#
# model_redo = Double_UNet(output=2)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_double_d1d2_l1__fullface_epoch_1_param_all_282_1915.pkl'))
# models.append(model_redo)
#
# model_redo = Double_UNet(output=1)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_double_d1d2_mse__fullface_epoch_1_param_all_13_325.pkl'))
# models.append(model_redo)
#
# model_redo = Double_UNet(output=2)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_double_d1d2_mse__fullface_epoch_1_param_all_13_325.pkl'))
# models.append(model_redo)
#
# model_redo = U_Net()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_U_Net_l1__fullface_epoch_1_param_all_13_180.pkl'))
# models.append(model_redo)
#
# model_redo = U_Net()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_U_Net_l1__fullface_epoch_3_param_all_13_1834.pkl'))
# models.append(model_redo)
#
# model_redo = R2U_Net()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_R2U_Net_l1__fullface_epoch_1_param_all_13_1912.pkl'))
# models.append(model_redo)
#
# model_redo = R2U_Net()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_R2U_Net_l1__fullface_epoch_3_param_all_13_2015.pkl'))
# models.append(model_redo)
#
# model_redo = AttU_Net()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_AttU_Net_l1__fullface_epoch_1_param_all_13_230.pkl'))
# models.append(model_redo)
#
# model_redo = AttU_Net()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_AttU_Net_l1__fullface_epoch_1_param_all_23_350.pkl'))
# models.append(model_redo)
#
# model_redo = R2AttU_Net()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_R2AttU_Net_l1__fullface_epoch_1_param_all_13_2352.pkl'))
# models.append(model_redo)
#
# model_redo = R2AttU_Net()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_R2AttU_Net_l1__fullface_epoch_1_param_all_23_442.pkl'))
# models.append(model_redo)
#
# model_redo = NestedUNet()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_NestedUNet_l1__fullface_epoch_1_param_all_23_14.pkl'))
# models.append(model_redo)
#
# model_redo = NestedUNet()
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_NestedUNet_l1__fullface_epoch_1_param_all_23_551.pkl'))
# models.append(model_redo)
#
# model_redo = Unet_dict(n_labels=3)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_Unet_dict_l1__fullface_epoch_1_param_all_23_148.pkl'))
# models.append(model_redo)
#
# model_redo = Unet_dict(n_labels=3)
# model_redo.to(device)
# model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_Unet_dict_l1__fullface_epoch_1_param_all_23_635.pkl'))
# models.append(model_redo)
#

##########################


mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
denorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1.0 / s for s in std],
        # always_apply=True,
        # max_pixel_value=1.0
)
transf_orig = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

p = 0.25
transf_orig_noise = transforms.Compose([
    transforms.RandomApply([transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(1), saturation=(0.5, 1.5), hue=(-0.1, 0.1))], p=p),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.RandomRotation(15)], p=p),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(10, 25)), transforms.GaussianBlur(kernel_size=(9, 9), sigma=(10, 25))], p=p),
    transforms.RandomApply([transforms.RandomResizedCrop(size=(299, 299), scale=(0.8, 1))], p=p),
    transforms.RandomAdjustSharpness(2, p=p),
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
########################################################

if epoch_done != 0:
        print('Starting from Epoch ', epoch_done+1)
rands = []
for epoch in range(epoch_done + 1, nr_epochs+1):

        print('Epoch: ', epoch)
        train_loss = 0.0
        losses = AverageMeter()
        accuracies = AverageMeter()
        times = AverageMeter()
        test_losses = AverageMeter()
        predictions_vect = []
        targets_vect = []
        prediction_df = pd.DataFrame(columns=['GT', 'prediction'])

        model.train()
        if dataset == 'celebDF':
                data_train.augment_dataset()
                data_train.shuffle()

        print('rands', rands)

        for i in range(int(len(data_train))):

                data_train.shuffle()
                t = time.time()
                data, targets = data_train[i]
                data, targets = data.to(device), targets.to(device)

                if not normal_data:
                        rand = random.randint(0, len(models) - 1)
                        if rand == len(models):
                                data = denorm(data)
                                data_redo = transf_orig_noise(data)
                        else:
                                model_redo = models[rand]
                                ##########################################adding the model
                                with torch.no_grad():
                                        data_redo = model_redo(data)

                                data_redo[data_redo > 1] = 1
                                data_redo[data_redo < -1] = -1

                                data_redo = denorm(data_redo)
                                data_redo = transf_orig_noise(data_redo)

                        # add white noise randomly
                        p = 0.33
                        if p > torch.rand(1):
                                data_redo = data_redo + 0.02 * torch.randn(data_redo.shape).to(device)

                        data = data_redo

                data = transf_orig_noise(data)
                outputs_gpu = model(data)

                outputs = outputs_gpu.to('cpu').flatten()
                targets = targets.to('cpu')
                filter_nan = outputs.isnan()
                outputs = outputs[~filter_nan]
                targets = targets[~filter_nan]
                if outputs.shape[0] < train_batch_size:
                        print('problem with nans????????????')
                if len(targets) == 0:
                        print('BIG problem with nans????????????')
                        continue

                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                if gradient_clipping_value:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_value)
                optimizer.step()

                predictions_vect.append(outputs.detach())
                targets_vect.append(targets)

                outputs_values = copy.copy(outputs.detach())
                outputs[outputs >= 0.5] = 1
                outputs[outputs < 0.5] = 0
                acc = np.count_nonzero(outputs == targets) / len(targets)

                train_loss += loss.item()
                losses.update(loss.item(), data.size(0))
                accuracies.update(acc, data.size(0))

                batch_t = time.time() - t
                times.update(batch_t, 1)
                avg_loss = train_loss

                if len(torch.unique(torch.cat(targets_vect).flatten())) > 1:
                        auc_train = roc_auc_score(torch.cat(targets_vect).flatten(), torch.cat(predictions_vect).flatten())
                else:
                        auc_train = '-'

                print('Minibatch: ' + str(i) + '/' + str(len(data_train))  + ' Loss: ' + str(avg_loss) +
                      ' Acc: ' + str(accuracies.avg) + ' AUC total: ' + str(auc_train) +
                      ' Est time/Epoch: ' + str(int(times.avg * len(data_train) // 3600)) + 'h' +
                      str(int((times.avg * len(data_train) - 3600 * (times.avg * len(data_train) // 3600)) // 60)) + 'm')
                train_loss = 0.0
                # print('Outputs: ', outputs_values)
                # print('Targets: ', targets)

        # Saving model
        model_file_name = model_type + model_suffix + '_' + img_type + '_epoch_' + str(epoch) + '_param_' + dataset + '_' + str(time.gmtime()[2]) + str(time.gmtime()[1]) + '_' + str(time.gmtime()[3]) + str(time.gmtime()[4]) + '.pkl'
        torch.save(model.state_dict(),
                   os.path.join(r'D:\saved_model', model_file_name))

        try:
                prediction_df['GT'] = torch.cat(targets_vect).flatten().numpy()
                prediction_df['prediction'] = torch.cat(predictions_vect).flatten().numpy()
                prediction_df.to_excel(r'D:\saved_model\outputs_train_' + model_type + '_' + img_type + '_epoch_' + str(epoch) + '_' + dataset + '_' +
                                       str(time.gmtime()[2]) + str(time.gmtime()[1]) + '_' + str(time.gmtime()[3]) + str(time.gmtime()[4]) +'.xlsx')
        except:
                pass


        # Testing
        if epoch % test_data_frequency == 0:
                print('Starting TEST')

                test_predictions_vect = []
                test_targets_vect = []
                test_df = copy.copy(data_test.label_df)
                model.eval()

                for i in range(len(data_test)):

                        data, targets = data_test[i]
                        data, targets = data.to(device), targets.to(device)

                        with torch.no_grad():

                                outputs_gpu = model(data)
                                outputs = outputs_gpu.to('cpu').flatten()
                                targets = targets.to('cpu')
                                loss = criterion(outputs, targets)
                                test_losses.update(loss.item(), data.size(0))
                                test_predictions_vect.append(outputs.detach())
                                test_targets_vect.append(targets)

                if len(torch.unique(torch.cat(test_targets_vect).flatten())) > 1:
                        auc_test = roc_auc_score(torch.cat(test_targets_vect).flatten(), torch.cat(test_predictions_vect).flatten())
                else:
                        auc_test = '-'

                print('\n\n\n\n========================================================' +
                      '\n TRAIN Epoch: ' + str(epoch) +'\n TRAIN Loss: ' + str(losses.avg) +
                      '\n TRAIN Accuracy: ' + str(accuracies.avg) + '\n TRAIN AUC total: ' + str(auc_train) +
                      '\n TEST Epoch: ' + str(epoch) + '\n TEST Loss: ' + str(test_losses.avg) +
                      '\n TEST AUC total: ' + str(auc_test) +
                      '\n========================================================')

                f = open(r"D:\saved_model\results_" + model_type + '_' + img_type + '_' + dataset + ".txt", "a")
                f.write('\n\n\n' + model_file_name +
                        '\n========================================================' +
                      '\n' + img_type +
                      '\n TRAIN Epoch: ' + str(epoch) +'\n TRAIN Loss: ' + str(losses.avg) +
                      '\n TRAIN Accuracy: ' + str(accuracies.avg) + '\n TRAIN AUC total: ' + str(auc_train) +
                      '\n TEST Epoch: ' + str(epoch) + '\n TEST Loss: ' + str(test_losses.avg) +
                      '\n TEST AUC total: ' + str(auc_test) +
                      '\n========================================================\n')
                f.close()

                try:
                        test_df['GT'] = torch.cat(test_targets_vect).flatten().numpy()
                        test_df['Pred'] = torch.cat(test_predictions_vect).flatten().numpy()
                        test_df.to_excel(r'D:\saved_model\outputs_test_' + model_type + '_' + img_type + '_epoch_' + str(epoch) + '_' + dataset + '_' +
                                         str(time.gmtime()[2]) + str(time.gmtime()[1]) + '_' + str(time.gmtime()[3]) + str(time.gmtime()[4]) + '.xlsx')
                except:
                        print('NU A MERS EXCELU PT TEST')


        else:

                print('\n\n\n\n========================================================' +
                      '\n TRAIN Epoch: ' + str(epoch) +
                      '\n TRAIN Loss: ' + str(losses.avg) +
                      '\n TRAIN Accuracy: ' + str(accuracies.avg) +
                      '\n TRAIN AUC total: ' + str(auc_train) +
                      '\n========================================================')

                f = open(r"D:\saved_model\results_" + model_type + '_' + img_type + '_' + dataset + ".txt", "a")
                f.write('\n\n\n\n========================================================' +
                                '\n ' + img_type +
                              '\n TRAIN Epoch: ' + str(epoch) +'\n TRAIN Loss: ' + str(losses.avg) +
                              '\n TRAIN Accuracy: ' + str(accuracies.avg) + '\n TRAIN AUC total: ' + str(auc_train) +
                              '\n========================================================\n')
                f.close()

        lr = lr * lr_decay
        for g in optimizer.param_groups:
                g['lr'] = lr





