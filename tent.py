from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F

from loss import CalculateNormPSD, NegPearsonLoss

from einops import rearrange

from memoryBank import RangeBasedMemoryBank, get_HR

from loss import *

import random


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, MB=None, cGAN_model=None, weight_std=5):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        self.MB = MB
        self.cGAN_model = cGAN_model
        self.weight_std = weight_std
        if self.MB is None:
            self.MB = RangeBasedMemoryBank()

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        # self.model_state, self.optimizer_state = \
        #     copy_model_and_optimizer(self.model, self.optimizer)
        
        self.MB_all_item = self.MB.get_all()

    def forward(self, x, x_aug=None, do_adapt=True):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, loss_dict = forward_and_adapt(x, x_aug, self.model, self.optimizer, 
                                                   self.MB, self.MB_all_item, 
                                                   self.cGAN_model, self.weight_std,
                                                   do_adapt)

        return outputs, loss_dict

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
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


def gaussian_weighting(x, mean=0, std_dev=5):
    
    x = torch.tensor(x, dtype=torch.float32)
    
    # normalization_factor = 1 / (std_dev * torch.sqrt(torch.tensor(2 * torch.pi)))
    
    y = torch.exp(-0.5 * ((x - mean) / std_dev) ** 2)

    return y


def MB_cos(x_middle, outputs, MB, MB_all_item, weight_std) -> torch.Tensor:
    
    loss_cos = 0
    
    for b in range(x_middle.size(0)):
        
        sig = outputs[b, -1]
        
        # all_HR, MB_x = MB.get_all_item_and_HR(item="x")
        
        
        
        # random_indices = random.sample(range(MB_all_item), 5)
        # MB_x = [MB_all_item[i][0] for i in random_indices]
        # all_HR = [MB_all_item[i][3] for i in random_indices]
        
        
        MB_x = [item[0] for item in MB_all_item]
        all_HR = [item[3] for item in MB_all_item]

        MB_x = torch.stack(MB_x).detach()#.to(x_middle.device).detach()
        MB_x = MB_x.view((MB_x.size(0), -1))
        

        cur_HR = get_HR(sig, fs=30)
        diff_HR = np.array(all_HR) - cur_HR
        
        # print(f"{cur_HR=}")
        # print(f"{all_HR=}")
        # print(f"{diff_HR=}")
        # Our weighting method
        ########################################################
        weights = gaussian_weighting(diff_HR, std_dev=weight_std).to(x_middle.device)
        
        cos_MB = F.cosine_similarity(x_middle[b].view(1, -1), MB_x, dim=1)
        # cos_MB = (1 + weights) * (1 - cos_MB) + (1 - weights) * cos_MB
        cos_MB = torch.abs(weights - cos_MB)
        loss_cos += cos_MB.mean()
        ########################################################
        
        
        # 2分法 method
        ########################################################
        # threshold = 2
        # same_MB = np.where((diff_HR >= -threshold) & (diff_HR <= threshold))[0]
        # diff_MB = np.where((diff_HR < -threshold) | (diff_HR > threshold))[0]
        
        # anchor = x_middle[b].view(1, -1)
        # cos_MB = 0
        
        # if len(same_MB) > 0:
        #     cos_MB += 1 - F.cosine_similarity(anchor, MB_x[same_MB], dim=1).mean()
        
        # if len(diff_MB) > 0:
        #     cos_MB += F.cosine_similarity(anchor, MB_x[diff_MB], dim=1).mean()    
        
        # loss_cos += cos_MB
        ########################################################
        
        
    loss_cos = loss_cos / x_middle.size(0)
    
    return loss_cos



neg_pearson_loss = NegPearsonLoss()

# @torch.jit.script
def signal_sim(x):
    
    loss = 0
    for signals in x:
        
        target = signals[0:-1]
        anchor = signals[-1]
        anchor = anchor.view((1, anchor.size(0))).repeat(target.size(0), 1)
        
        loss += neg_pearson_loss(target, anchor)        
        
    return loss / x.size(0)


contrastLoss = ContrastLoss(delta_t=150, K=4, Fs=30, 
                            high_pass=40, low_pass=250)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, x_aug, model, optimizer, MB, MB_all_item, cGAN_model, weight_std, do_adapt=True):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    # print(f"{x.shape=}")#, {x_aug.shape=}")
    outputs, x_middle = model(x)

    loss_dict = {}
    # adapt
    if do_adapt:

        # print(f"{x_middle.shape=}")
        x_middle = x_middle.view((x.size(0), -1))
        
        
        ############  L_FE  ############
        loss_entropy = PSD_entropy(outputs)
        ############  L_FE  ############
        
        
        
        ###########  L_F  ############
        loss_cos = MB_cos(x_middle, outputs, MB, MB_all_item, weight_std)
        ###########  L_F  ############
        
        
        # If want to regenerate feature, use this
        # feature, psd_target, ppg_target = cGAN_model(batch_size=3)
        # rPPG = model.forward_cGAN(feature)
        # rPPG_anc = rPPG[:, -1]
        
        
        
        ############  L_E  ############
        random_indices = random.sample(range(len(MB_all_item)), 5)
        feature = [MB_all_item[i][0] for i in random_indices]
        ppg_target = [MB_all_item[i][2] for i in random_indices]
        
        feature = torch.stack(feature).detach()#.to(x.device)
        ppg_target = torch.stack(ppg_target).detach()#.to(x.device)
        
        # print(f"{feature.shape=}")
        rPPG = model.forward_cGAN(feature)
        rPPG_anc = rPPG[:, -1]
        loss_pearson = neg_pearson_loss(rPPG_anc, ppg_target)
        ############  L_E  ############
    
    
        
        print(f"{loss_entropy.mean(0).item()=}, {loss_cos.item()=}, {loss_pearson.item()=}")
        # # print(f"{loss_entropy.mean(0).item()=}, {loss_cos.item()=}")
        
        
        loss = loss_entropy.mean(0) * 0.5 + loss_pearson + loss_cos * 0.5
        # loss = loss_entropy.mean(0) + loss_pearson
        # loss = loss_entropy.mean(0) + loss_cos
        # loss = loss_entropy.mean(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # for b in range(x.size(0)):
        #     ent = loss_entropy[b]
        #     if ent < 2.0:
        #         MB.add_by_sig(x_middle[b], outputs[b, -1])
        
        loss_dict = {"loss_entropy": loss_entropy.mean(0).item(),
                     "loss_cos": loss_cos.item(),
                     "loss_pearson": loss_pearson.item()}
        
        
        # loss_dict = {"a" : 1}

    
    return outputs, loss_dict


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # if isinstance(m, nn.BatchNorm3d):
        #     for np, p in m.named_parameters():
        #         if np in ['weight', 'bias']:  # weight is scale, bias is shift
        #             params.append(p)
        #             names.append(f"{nm}.{np}")
        for np, p in m.named_parameters():
            if np in ['weight', 'bias']:  # weight is scale, bias is shift
                params.append(p)
                names.append(f"{nm}.{np}")
                
    print(params, names)
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
        
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    # assert not has_all_params, "tent should not update all params: " \
    #                            "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm3d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
