# from PhysNetModel import PhysNet
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch import optim

from einops import rearrange
from PhysNetModel import PhysNet
from conditional_GAN import ConditionalGenerator
from memoryBank import RangeBasedMemoryBank

from util import *
from dataloader import get_loader
from loss import *
import glob 
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score

from tqdm import tqdm

from tent import *
import time

import pickle


def Find_Optimal_Cutoff(TPR, FPR, threshold):
    # y = TPR - FPR
    y = TPR + (1 - FPR)
    # print(y)
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')


args = get_args()
finetune=False
target_dataset = args.train_dataset
if args.finetune_dataset:
    finetune=True
    

trainName, testName, finetuneName = get_name(args, finetune=finetune, model_name=PhysNet.__name__)
log = get_logger(f"logger/test_tent/{args.test_dataset}/", testName)

print(f"{trainName=}")
print(f"{finetuneName=}")
print(f"{testName=}")

targetName = trainName
if finetune:
    target_dataset = args.finetune_dataset
    targetName = finetuneName
    


seq_len = args.train_T * args.fps
num_batch_per_sample=3
not_preload = args.do_not_preload

test_loader = get_loader(_datasets=args.test_dataset,
                         _seq_length=seq_len,
                         batch_size=args.bs,
                         train=False,
                         if_bg=False,
                         shuffle=False, 
                         real_or_fake="real",
                         num_batch_per_sample=num_batch_per_sample,
                         if_preload=not_preload,
                         if_aug=False,
                         testFold=args.testFold,)

targetName = trainName if not finetune else finetuneName


# TODO: Change the model path
model_path = f"./results/{target_dataset}/{targetName}/weight"
cGAN_model = ConditionalGenerator(device=device).to(device)
cGAN_pth = f"./results/{target_dataset}/{targetName}/weight/cGAN.pt"

print(f"{cGAN_pth=}")
cGAN_model.load_state_dict(torch.load(cGAN_pth, map_location=device))  # load weights to the model
cGAN_model.requires_grad_(False)

MB = RangeBasedMemoryBank(range_width=1, max_length=args.MB_size)
desired_HRs = np.arange(30, 180, 1)

for size in range(1, args.MB_size):
    x, psd, ppg = cGAN_model(bpms=desired_HRs, batch_size=len(desired_HRs))
    for i in range(len(desired_HRs)):
        # MB.add(desired_HRs[i], (x[i].view(-1), psd[i], ppg[i]))
        feature = x[i].detach()
        _psd = psd[i].detach()
        _ppg = ppg[i].detach()
        MB.add(desired_HRs[i], (feature, _psd, _ppg, desired_HRs[i]))


exp_data_dict = {}

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

for epoch in range(3, args.epoch):
    
    epoch_time = 0
    epoch_frame = 0
    
    model = PhysNet(S=args.model_S,
                in_ch=args.in_ch,
                conv_type=args.conv,
                seq_len=seq_len, 
                delta_t=args.delta_T*args.fps, 
                numSample=args.numSample,
                class_num=2).to(device)

    opt_fg = optim.AdamW(model.parameters(), lr=args.lr)

    model_state = torch.load(model_path + f'/fg_epoch{epoch}.pt', map_location=device)
    opt_state = torch.load(model_path + f'/fg_opt_epoch{epoch}.pt', map_location=device)

    load_model_and_optimizer(model, opt_fg, model_state, opt_state)
    model = configure_model(model)
    check_model(model)

    model_tent = Tent(model, opt_fg, MB=MB, cGAN_model=cGAN_model, weight_std=args.weight_std)
    

    # Specify the functions and initialize metric lists
    functions = {
        "HR": (predict_heart_rate_batch, [], [], []),
    }
    
    for step, (face_frames, _, _, ppg_label, subjects) in enumerate(test_loader):

        # (2, 3, 900, 64, 64) to (6, 3, 300, 64, 64)
        face_frames = rearrange(face_frames, 'b c (t1 t2) h w -> (b t1) c t2 h w', t2=seq_len).to(device)
        
        # half_T = imgs.shape[2] // 2
        # face_frames_ori = rearrange(face_frames[:, :, :half_T], 'b c (t1 t2) h w -> (b t1) c t2 h w', t2=seq_len).to(device)
        # face_frames_aug = rearrange(face_frames[:, :, half_T:], 'b c (t1 t2) h w -> (b t1) c t2 h w', t2=seq_len).to(device)

        _label = rearrange(ppg_label, 'b (t1 t2) -> (b t1) t2', t2=seq_len).detach().cpu().numpy()

        starter.record()

        rPPG, loss_dict = model_tent(face_frames, x_aug=None, do_adapt = not args.do_not_adapt)
        
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        
        end_time = time.time()
        epoch_time += curr_time / 1000
        epoch_frame += face_frames.shape[2]
        
        rppg = rPPG.detach().cpu().numpy()[:, -1]
        rppg = butter_bandpass_batch(rppg, lowcut=0.6, highcut=4, fs=args.fps)
        _label = butter_bandpass_batch(_label, lowcut=0.6, highcut=4, fs=args.fps)

        # Processing with the specified functions
        
        try:
            for func_name, (func, mae_list, rmse_list, r_list) in functions.items():
                
                preds = func(rppg.copy(), fs=args.fps)
                lbls = func(_label.copy(), fs=args.fps)

                for i in range(0, len(preds), num_batch_per_sample):
                    pred = preds[i:i+num_batch_per_sample]
                    lbl = lbls[i:i+num_batch_per_sample]

                    mae = np.mean(np.abs(pred - lbl))
                    rmse = np.sqrt(np.mean((pred - lbl) ** 2))
                    pearson_corr = Pearson_np(pred, lbl)

                    mae_list.append(mae)
                    rmse_list.append(rmse)
                    r_list.append(pearson_corr)

                    log.info(f'[epoch {epoch} step {step} mae {mae:>8.5f} rmse {rmse:>8.5f} pearson_corr {pearson_corr:>8.5f}] {subjects[i//num_batch_per_sample]:<12} func {func_name}')                    
                    
        except Exception as e:
            log.info(f'[epoch {epoch} step {step} func {func_name} error {e}]')
            continue
        
        
        exp_data_dict[f"epoch{epoch}_step{step}"] = {
            "mae": mae,
            "rmse": rmse,
            "pearson_corr": pearson_corr,
            **loss_dict
        }
        

    # Logging the average metrics
    for func_name, (_, mae_list, rmse_list, r_list) in functions.items():
        log.info(f'[epoch {epoch} avg all_mae {np.mean(mae_list):>8.5f} all_rmse {np.mean(rmse_list):>8.5f} all_R {np.mean(r_list):>8.5f}] (func {func_name})')

            
    
    log.info(f"Testing time for epoch {epoch}: {epoch_time:.3f} seconds, fps: {epoch_frame / epoch_time:.3f}")
    
    
    pickle.dump(exp_data_dict, open(f"logger/test_tent/{args.test_dataset}/{testName}.pkl", "wb"))