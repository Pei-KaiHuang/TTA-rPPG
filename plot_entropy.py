import torch
import torch.nn as nn
import torchvision.transforms as T

import matplotlib.pyplot as plt
import numpy as np
import os
import glob

from PhysNetModel import PhysNet
from conditional_GAN import ConditionalGenerator
from util import *
from dataloader import get_loader

import torch.nn.functional as F

from loss import CalculateNormPSD, NegPearsonLoss



if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    
    


rPPG_model = PhysNet(S=2,
                in_ch=3,
                conv_type="LDC_M",
                seq_len=300, 
                delta_t=150, 
                numSample=4,
                class_num=2).to(device).train()


cGAN_model = ConditionalGenerator(device=device).to(device).train()


dataset="C"
trainName=f"{dataset}_LDC_M_train_T10_S2_K4_PhysNet"
model_pth_list = sorted(glob.glob(f"./results/{dataset}/{trainName}/weight/fg_epoch*.pt"))

print(f"getting pretrained path in ./results/{dataset}/{trainName}")
print(f"{model_pth_list[-1]=}")
rPPG_model.load_state_dict(torch.load(model_pth_list[-1], map_location=device))  # load weights to the model


calculate_norm_psd = CalculateNormPSD(Fs=30, high_pass=40, low_pass=250)


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def compute_power_spectrum_torch(signal, Fs, zero_pad=None, high_pass=40, low_pass=250):
    # Assuming signal is a PyTorch tensor
    if zero_pad is not None:
        L = signal.size(0)
        padding = int(zero_pad / 2 * L)
        signal = torch.nn.functional.pad(signal, (padding, padding), 'constant', 0)

    # Compute the FFT frequencies
    freqs = torch.fft.fftfreq(signal.size(0), 1 / Fs) * 60  # in bpm

    # Compute the FFT and power spectrum
    ps = torch.abs(torch.fft.fft(signal))**2

    # Only keep the positive frequencies (one-sided spectrum)
    cutoff = len(freqs) // 2
    freqs = freqs[:cutoff]
    ps = ps[:cutoff]
    
    
    valid_freqs = (freqs >= high_pass) & (freqs <= low_pass)
    freqs = freqs[valid_freqs]
    ps = ps[valid_freqs]
    
    return freqs, ps



def PSD_entropy(x: torch.Tensor) -> torch.Tensor:
    
    # (B, spatial_window^2, T) -> (B * spatial_window^2, T)
    x = x.view((x.size(0) * x.size(1), x.size(2)))
    
    psd = []
    for i in range(x.size(0)):
        freqs, ps = compute_power_spectrum_torch(x[i], 30, zero_pad=100)
        psd.append(ps)
    psd = torch.stack(psd)
    entropy = softmax_entropy(psd)
    
    return entropy


# @torch.jit.script



def plot_rPPG_PSD(sig, target, index=0):

    hr = predict_heart_rate(sig, 30)
    # hr2, ps, x_hr = hr_fft(sig, 30)
    # freqs, ps = compute_power_spectrum(sig, 30, zero_pad=100)
    freqs, ps = compute_power_spectrum_torch(torch.from_numpy(sig), 30, zero_pad=100)

    valid_freqs = (freqs >= 40) & (freqs <= 250)
    freqs = freqs[valid_freqs]
    ps = ps[valid_freqs]
    
    print(f"{freqs.shape=}, {ps.shape=}")


    entropy = PSD_entropy(torch.from_numpy(sig).unsqueeze(0).unsqueeze(0)).mean(0)
    # entropy = softmax_entropy(torch.from_numpy(ps).unsqueeze(0)).mean(0)
    print(f"{entropy=}")

    plt.plot(freqs, ps)
        
    print(f"{hr=}")

    plt.figure(figsize=(8, 7))
    plt.subplot(221)
    plt.plot(sig)
    plt.yticks([])
    plt.title("rPPG Prediction")
    plt.subplot(222)
    plt.plot(freqs, ps)
    plt.yticks([])
    plt.title(f"PSD (entropy={entropy:.2f})")
    
    
    hr = predict_heart_rate(target, 30)
    # hr2, ps, x_hr = hr_fft(sig, 30)
    # freqs, ps = compute_power_spectrum(target, 30, zero_pad=100)

    freqs, ps = compute_power_spectrum_torch(torch.from_numpy(target), 30, zero_pad=100)
    valid_freqs = (freqs >= 40) & (freqs <= 250)
    freqs = freqs[valid_freqs]
    ps = ps[valid_freqs]
    
    

    entropy = PSD_entropy(torch.from_numpy(target).unsqueeze(0).unsqueeze(0)).mean(0)
    # entropy = softmax_entropy(torch.from_numpy(ps).unsqueeze(0)).mean(0)
    print(f"{entropy=}")

    plt.subplot(223)
    plt.plot(target)
    plt.yticks([])
    plt.title("rPPG Ground Truth")
    plt.subplot(224)
    plt.plot(freqs, ps)
    plt.yticks([])
    plt.title(f"PSD (entropy={entropy:.2f})")
    
    
    plt.show()
    os.makedirs("entropy", exist_ok=True)
    plt.savefig(f"entropy/rPPG_PSD_{index}.png")
    
    plt.close()


def get_rPPG(path):

    f = open(path, 'r')
    lines = f.readlines()
    PPG = [float(ppg) for ppg in lines[0].split()]
    # hr = [float(ppg) for ppg in lines[1].split()[:100]]
    # no = [float(ppg) for ppg in lines[2].split()[:100]]
    f.close()

    return PPG



# ppg = get_rPPG("../dataset/UBFC/crop_MTCNN/subject15/ground_truth.txt")[300:600]
# plot_rPPG_PSD(ppg)

test_loader = get_loader(_datasets=dataset,
                         _seq_length=300,
                         batch_size=1,
                         train=True,
                         if_bg=False,
                         shuffle=True, 
                         real_or_fake="real",
                         num_batch_per_sample=1)



for step, (face_frames, _, _, ppg_label, subjects) in enumerate(test_loader):

    face_frames = face_frames.to(device)
    rPPG, _ = rPPG_model(face_frames)
    
    rPPG_anc = rPPG[:, -1]
    print(f"{rPPG_anc.shape=}")
    print(f"{ppg_label.shape=}")

    B = rPPG_anc.shape[0]
    for i in range(B):
        plot_rPPG_PSD(rPPG_anc[i].detach().cpu().numpy(), ppg_label[i].detach().cpu().numpy(), step)

    print(f"{step=}")