import torch
import torch.nn as nn
import torchvision.transforms as T
from torch import optim

from einops import rearrange
from PhysNetModel import PhysNet
from conditional_GAN import ConditionalGenerator

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


rPPG_model = PhysNet(S=args.model_S,
                in_ch=args.in_ch,
                conv_type=args.conv,
                seq_len=seq_len, 
                delta_t=args.delta_T*args.fps, 
                numSample=args.numSample,
                class_num=2).to(device).train()

for param in rPPG_model.parameters():
     param.requires_grad = False


cGAN_model = ConditionalGenerator(device=device).to(device).train()



opt_fg = optim.AdamW(rPPG_model.parameters(), lr=args.lr)
opt_cGAN = optim.AdamW(cGAN_model.parameters(), lr=args.lr)

# TODO: Change the model path
model_pth_list = sorted(glob.glob(f"./results/{args.train_dataset}/{trainName}/weight/fg_epoch*.pt"))
opt_pth_list = sorted(glob.glob(f"./results/{args.train_dataset}/{trainName}/weight/fg_opt_epoch*.pt"))
print(f"getting pretrained path in ./results/{args.train_dataset}/{trainName}")
print(f"{model_pth_list[-1]=}")
print(f"{opt_pth_list[-1]=}")
rPPG_model.load_state_dict(torch.load(model_pth_list[-1], map_location=device))  # load weights to the model
opt_fg.load_state_dict(torch.load(opt_pth_list[-1], map_location=device))  # load weights to the model



IPR = IrrelevantPowerRatio(Fs=args.fps, 
                           high_pass=args.high_pass, low_pass=args.low_pass)


SR = SparsityRatio(Fs=args.fps, 
                   high_pass=args.high_pass, low_pass=args.low_pass)


CE_loss= nn.CrossEntropyLoss()
neg_pearson_loss = NegPearsonLoss()


for epoch in range(args.epoch):
    
    # for it in range(np.round(10 / (T / fs)).astype('int')):
    print(f"epoch_train: {epoch}/{args.epoch}:")
        
    
    feature, psd_target, ppg_target = cGAN_model(batch_size=args.bs)
    rPPG = rPPG_model.forward_cGAN(feature)
    rPPG_anc = rPPG[:, -1]
    
    psd = [cGAN_model.norm_psd(rPPG_anc[i]) for i in range(rPPG_anc.shape[0])]
    psd = torch.stack(psd)
    loss_ce = CE_loss(psd, psd_target)
    loss_pearson = neg_pearson_loss(rPPG_anc, ppg_target)
    

    opt_cGAN.zero_grad()
    total_loss = loss_ce + loss_pearson
    # total_loss = loss_ce 
    # total_loss = loss_pearson
    total_loss.backward()
    opt_cGAN.step()
    
    
    # evaluate irrelevant power ratio during training
    ipr = torch.mean(IPR(rPPG_anc.squeeze(1).clone().detach()))
    sr = torch.mean(SR(rPPG_anc.squeeze(1).clone().detach()))
    
    loss_string =  f"[epoch {epoch}]"
    # loss_string += f" loss_ce: {loss_ce.item():.4f}"
    loss_string += f" loss_pearson: {loss_pearson.item():.4f}"

    loss_string += f" IPR: {ipr.item():.4f}"
    loss_string += f" SR: {sr.item():.4f}"
    log.info(loss_string)



    # torch.cuda.empty_cache()
torch.save(cGAN_model.state_dict(), result_dir + '/weight/cGAN.pt')

