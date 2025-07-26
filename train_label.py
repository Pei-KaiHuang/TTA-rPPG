import torch
import torch.nn as nn
import torchvision.transforms as T
from torch import optim

from einops import rearrange
from PhysNetModel import PhysNet

from util import *
from dataloader import get_loader
from loss import *
import glob 


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
    target_dataset = args.finetune_dataset

trainName, _, finetuneName = get_name(args, finetune=finetune, model_name=PhysNet.__name__)

if finetune:
    log = get_logger(f"logger/finetune/{target_dataset}/", finetuneName)
    result_dir = f"./results/{target_dataset}/{finetuneName}/"
else:
    log = get_logger(f"logger/train/{target_dataset}/", trainName)
    result_dir = f"./results/{target_dataset}/{trainName}/"


print(f"{trainName=}, {finetuneName=}")



os.makedirs(f"{result_dir}/weight", exist_ok=True)

seq_len = args.train_T*args.fps
not_preload = args.do_not_preload
if_bg = args.bg
train_loader = get_loader(_datasets=target_dataset,
                          _seq_length=seq_len,
                          batch_size=args.bs,
                          train=True,
                          if_bg=if_bg,
                          shuffle=True, 
                          real_or_fake="real",
                          if_preload=not_preload,
                          if_aug=False,
                          testFold=args.testFold,)


model = PhysNet(S=args.model_S,
                in_ch=args.in_ch,
                conv_type=args.conv,
                seq_len=seq_len, 
                delta_t=args.delta_T*args.fps, 
                numSample=args.numSample,
                class_num=2).to(device).train()

opt_fg = optim.AdamW(model.parameters(), lr=args.lr)

# TODO: modify the path to store the training model
if finetune:
    model_pth_list = sorted(glob.glob(f"./results/{args.train_dataset}/{trainName}/weight/fg_epoch*.pt"))
    opt_pth_list = sorted(glob.glob(f"./results/{args.train_dataset}/{trainName}/weight/fg_opt_epoch*.pt"))
    print(f"getting pretrained path in ./results/{args.train_dataset}/{trainName}")
    print(f"{model_pth_list[-1]=}")
    print(f"{opt_pth_list[-1]=}")
    model.load_state_dict(torch.load(model_pth_list[-1], map_location=device))  # load weights to the model
    opt_fg.load_state_dict(torch.load(opt_pth_list[-1], map_location=device))  # load weights to the model



IPR = IrrelevantPowerRatio(Fs=args.fps, 
                           high_pass=args.high_pass, low_pass=args.low_pass)


SR = SparsityRatio(Fs=args.fps, 
                   high_pass=args.high_pass, low_pass=args.low_pass)


NP_loss = NegPearsonLoss()



for epoch in range(args.epoch):
    
    # for it in range(np.round(10 / (T / fs)).astype('int')):
    print(f"epoch_train: {epoch}/{args.epoch}:")
    for step, (face_frames, bg_frames, ls_label, ppg_label, subjects) in enumerate(train_loader):
        
        
        # face_frames = rearrange(face_frames, 'b d t c h w -> (b d) t c h w').to(device)
        # if batch size < 2, skip this round since we need at least 2 samples for the contrastive loss
        if(face_frames.shape[0] < 2):
            continue
        
        face_frames = face_frames.to(device)
        ppg_label = ppg_label.to(device)
        
        
        bg_output = None
        if bg_frames != []:
            bg_frames = bg_frames.to(device)
            rPPG_output, _ = model(face_frames, bg_frames)
        else:
            rPPG_output = model(face_frames)
        
        rPPG_anc, _ = rPPG_output
        rPPG_anc = rPPG_anc[:, -1]
        # print(f"{rPPG_anc.shape=}")
        
        loss_np = NP_loss(rPPG_anc, ppg_label)
            

        opt_fg.zero_grad()
        total_loss = loss_np
        total_loss.backward()
        opt_fg.step()
        
        
        # evaluate irrelevant power ratio during training
        ipr = torch.mean(IPR(rPPG_anc.clone().detach()))
        sr = torch.mean(SR(rPPG_anc.clone().detach()))
        
        loss_string =  f"[epoch {epoch} step {step}]"
        loss_string += f" loss_np: {loss_np.item():.5f}"
        loss_string += f" IPR: {ipr.item():.5f}"
        loss_string += f" SR: {sr.item():.5f}"
        log.info(loss_string)
        

    # torch.cuda.empty_cache()
    torch.save(model.state_dict(), result_dir + '/weight/fg_epoch%d.pt' % epoch)
    torch.save(opt_fg.state_dict(), result_dir + '/weight/fg_opt_epoch%d.pt' % epoch)

