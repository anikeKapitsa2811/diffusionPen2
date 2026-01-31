import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split
import torchvision
from tqdm import tqdm
from torch import optim
import copy
import argparse
import uuid
import json
from diffusers import AutoencoderKL, DDIMScheduler
import random
from unet import UNetModel
import wandb
from torchvision import transforms
from feature_extractor import ImageEncoder
from utils.iam_dataset import IAMDataset
from utils.GNHK_dataset import GNHK_Dataset
from utils.auxilary_functions import *
from torchvision.utils import save_image
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer

os.environ.setdefault("MPLCONFIGDIR", "/cluster/datastore/aniketag/.mplcache")
os.environ.setdefault("TORCH_HOME", "/cluster/datastore/aniketag/allData/pytrochWeights")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)

CACHE_ROOT = "/cluster/datastore/aniketag/allData/pytrochWeights"  # <- your folder
os.environ["TORCH_HOME"] = CACHE_ROOT                      # torch hub uses TORCH_HOME/checkpoints
# 2) Avoid ~/.cache write attempts (some libs look at XDG_CACHE_HOME)
os.environ["XDG_CACHE_HOME"] = os.path.join(CACHE_ROOT, ".xdg-cache")

# 3) Silence matplotlib/fontconfig cache errors (optional but nice)
os.environ["MPLCONFIGDIR"] = os.path.join(CACHE_ROOT, ".mpl-cache")


torch.cuda.empty_cache()
OUTPUT_MAX_LEN = 95 #+ 2  # <GO>+groundtruth+<END>
IMG_WIDTH = 256
IMG_HEIGHT = 64

c_classes = '_!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
cdict = {c:i for i,c in enumerate(c_classes)}
icdict = {i:c for i,c in enumerate(c_classes)}

### Borrowed from GANwriting ###
def label_padding(labels, num_tokens):
    new_label_len = []
    ll = [letter2index[i] for i in labels]
    new_label_len.append(len(ll) + 2)
    ll = np.array(ll) + num_tokens
    ll = list(ll)
    #ll = [tokens["GO_TOKEN"]] + ll + [tokens["END_TOKEN"]]
    num = OUTPUT_MAX_LEN - len(ll)
    if not num == 0:
        ll.extend([tokens["PAD_TOKEN"]] * num)  # replace PAD_TOKEN
    return ll


def labelDictionary():
    labels = list(c_classes)
    letter2index = {label: n for n, label in enumerate(labels)}
    # create json object from dictionary if you want to save writer ids
    json_dict_l = json.dumps(letter2index)
    l = open("letter2index.json","w")
    l.write(json_dict_l)
    l.close()
    index2letter = {v: k for k, v in letter2index.items()}
    json_dict_i = json.dumps(index2letter)
    l = open("index2letter.json","w")
    l.write(json_dict_i)
    l.close()
    return len(labels), letter2index, index2letter


char_classes, letter2index, index2letter = labelDictionary()
tok = False
if not tok:
    tokens = {"PAD_TOKEN": 52}
else:
    tokens = {"GO_TOKEN": 52, "END_TOKEN": 53, "PAD_TOKEN": 54}
num_tokens = len(tokens.keys())
print('num_tokens', num_tokens)


print('num of character classes', char_classes)
vocab_size = char_classes + num_tokens



def setup_logging(args):
    #os.makedirs("models", exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)

def save_images(images, path, args, **kwargs):
    #print('image', images.shape)
    grid = torchvision.utils.make_grid(images, padding=0, **kwargs)
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
        if args.color == False:
            im = im.convert('L')
        else:
            im = im.convert('RGB')
    else:
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
    im.save(path)
    return im

def crop_whitespace_width(img):
    #tensor image to PIL
    original_height = img.height
    img_gray = np.array(img)
    ret, thresholded = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresholded)
    x, y, w, h = cv2.boundingRect(coords)
    #rect = img.crop((x, 0, x + w, original_height))
    rect = img.crop((x, y, x + w, y + h))
    return np.array(rect)


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text
    
class EMA:
    '''
    EMA is used to stabilize the training process of diffusion models by 
    computing a moving average of the parameters, which can help to reduce 
    the noise in the gradients and improve the performance of the model.
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())



class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=(64, 256), args=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(args.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = args.device

    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sampling_loader(self, model, test_loader, vae, n, x_text, labels, args, style_extractor, noise_scheduler, mix_rate=None, cfg_scale=3, transform=None, character_classes=None, tokenizer=None, text_encoder=None):
        model.eval()
        tensor_list = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader)
            style_feat = []
            for i, data in enumerate(pbar):
                images = data[0].to(args.device)
                transcr = data[1]
                s_id = data[2].to(args.device)
                style_images = data[3].to(args.device)
                cor_im = data[5].to(args.device)
                img_path = data[4]
                
                
                if args.model_name == 'wordstylist':
                    #print('transcr', transcr)
                    batch_word_embeddings = []
                    for trans in transcr:
                        word_embedding = label_padding(trans) 
                        #print('word_embedding', word_embedding)
                        word_embedding = np.array(word_embedding, dtype="int64")
                        word_embedding = torch.from_numpy(word_embedding).long() 
                        batch_word_embeddings.append(word_embedding)
                    text_features = torch.stack(batch_word_embeddings).to(args.device)
                else:
                    text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=200).to(args.device)
                
                reshaped_images = style_images.reshape(-1, 3, 64, 256)
                
                if style_extractor is not None:
                    style_features = style_extractor(reshaped_images).to(args.device)
                else:
                    style_features = None
            
                if args.latent == True:
                    x = torch.randn((images.size(0), 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
                    
                else:
                    x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
                
                #scheduler
                noise_scheduler.set_timesteps(50)
                for time in noise_scheduler.timesteps:
                    
                    t_item = time.item()
                    t = (torch.ones(images.size(0)) * t_item).long().to(args.device)

                    with torch.no_grad():
                        noisy_residual = model(x, t, text_features, labels, original_images=style_images, mix_rate=mix_rate, style_extractor=style_features)
                        prev_noisy_sample = noise_scheduler.step(noisy_residual, time, x).prev_sample
                        x = prev_noisy_sample
                    
        model.train()
        if args.latent==True:
            latents = 1 / 0.18215 * x
            image = vae.module.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            
            image = torch.from_numpy(image)
            x = image.permute(0, 3, 1, 2)

        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        return x

    def sampling(self, model, vae, n, x_text, labels, args, style_extractor, noise_scheduler, mix_rate=None, cfg_scale=3, transform=None, character_classes=None, tokenizer=None, text_encoder=None, run_idx=None):
        model.eval()
        tensor_list = []
        
        with torch.no_grad():
            style_images = None
            text_features = x_text #[x_text]*n
            #print('text features', text_features.shape)
            text_features = tokenizer(text_features, padding="max_length", truncation=True, return_tensors="pt", max_length=40).to(args.device)
            if args.img_feat == True:
                #pick random image according to specific style
                with open('./writers_dict_train.json', 'r') as f:
                    
                    wr_dict = json.load(f)
                reverse_wr_dict = {v: k for k, v in wr_dict.items()}
                
                #key = reverse_wr_dict[value]
                with open('./utils/splits_words/iam_train_val.txt', 'r') as f:
                #with open('./utils/splits_words/iam_test.txt', 'r') as f:
                    train_data = f.readlines()
                    train_data = [i.strip().split(',') for i in train_data]
                    style_featur = []
                    for label in labels:
                        #print('label', label)
                        label_index = label.item()
    
                        matching_lines = [line for line in train_data if line[1] == reverse_wr_dict[label_index] and len(line[2])>3]

                        #pick the first 5 from matching lines
                        
                        if len(matching_lines) >= 5:
                            #five_styles = matching_lines[:5]
                            #pick first line and repeat
                            #five_styles = [matching_lines[0]]*5
                            five_styles = random.sample(matching_lines, 5)
                            #five_styles = matching_lines_style[:5]
                        else:
                            matching_lines = [line for line in train_data if line[1] == reverse_wr_dict[label_index]]
                            #print('matching lines', matching_lines)
                            five_styles = matching_lines_style[:5]
                            five_styles = [matching_lines[0]]*5
                            #five_styles = random.sample(matching_lines, 5)
                        print('five_styles', five_styles)
                        #five_styles = random.sample(matching_lines, 5)
                        
                        cor_image_random = random.sample(matching_lines, 1)
                        #print('cor_image_random', cor_image_random)
                        #five_styles =[['a05/a05-084/a05-084-04-05.png', '000', 'which'], ['a03/a03-073/a03-073-04-04.png', '000', 'stage'], ['a01/a01-077u/a01-077u-02-02.png', '000', 'cables'], ['a05/a05-089/a05-089-00-05.png', '000', 'debate'], ['a05/a05-048/a05-048-00-00.png', '000', 'Long']] #class id 12
                        #five_styles = [['b06/b06-071/b06-071-08-06.png', '128', 'Labour'], ['b06/b06-019/b06-019-05-04.png', '128', 'West'], ['b06/b06-071/b06-071-05-03.png', '128', 'could'], ['c06/c06-027/c06-027-01-01.png', '128', 'advantage'], ['c06/c06-076/c06-076-01-05.png', '128', 'never']] #class id 1
                        
                        interpol = False
                        if interpol == True:
                            label2 = random.randint(0, 339) #random label
                            matching_lines2 = [line for line in train_data if line[1] == reverse_wr_dict[label2] and len(line[2])>3]
                            #print('matching lines2', matching_lines2)
                            five_styles2 = random.sample(matching_lines2, 5)
                            print('five_styles2', five_styles2)
                            style_images = []
                            style_images2 = []
                            for i in five_styles:
                                #print('i', i)
                                style_im = cv2.imread(os.path.join(args.data_path, i[0]))
                                #print('style_im', style_im.shape)
                                style_images.append(style_im)
                            style_images = np.array(style_images)
                            style_images = transform(torch.from_numpy(style_images).permute(0,3,1,2).float()/255)
                            #print('style_images', style_images.shape)
                            
                            for i in five_styles2:
                                style_im2 = cv2.imread(os.path.join(args.data_path, i[0]))
                                style_images2.append(style_im2)
                            style_images2 = np.array(style_images2)
                            style_images2 = transform(torch.from_numpy(style_images2).permute(0,3,1,2).float()/255)
                            #print('style_images2', style_images2.shape)
                            style_images = torch.cat((style_images, style_images2), dim=0).to(args.device)
                            #print('concat style_images', style_images.shape)
                            
                        else:
                            style_images = []
                            for i in five_styles:
                                #print('i', i)
                                style_im = cv2.imread(os.path.join(args.data_path, i[0]))
                                #print('style_im', style_im.shape)
                                style_images.append(style_im)
                            style_images = np.array(style_images)
                            style_images = transform(torch.from_numpy(style_images).permute(0,3,1,2).float()/255).to(args.device)
                            #print('style_images', style_images.shape)
                        
                        style_features = style_extractor(style_images).to(args.device)
                        #print('style_features', style_features.shape)
                        style_feat.append(style_features)
                
                style_features = torch.stack(style_feat).to(args.device)
                style_features = style_features.reshape(n, -1)
                #print('style features', style_features.shape)
            else:
                style_features = None
            #print('style_features', style_features.shape)
                
            if args.latent == True:
                x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
                
            else:
                x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
            
            #scheduler
            noise_scheduler.set_timesteps(50)
            for time in noise_scheduler.timesteps:
                
                t_item = time.item()
                t = (torch.ones(n) * t_item).long().to(args.device)

                with torch.no_grad():
                    noisy_residual = model(x, t, text_features, labels, original_images=style_images, mix_rate=mix_rate, style_extractor=style_features, character_classes=character_classes, text_encoder=text_encoder)
                    prev_noisy_sample = noise_scheduler.step(noisy_residual, time, x).prev_sample
                    x = prev_noisy_sample
                
        model.train()
        if args.latent==True:
            latents = 1 / 0.18215 * x
            image = vae.module.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            
            image = torch.from_numpy(image)
            x = image.permute(0, 3, 1, 2)

        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        return x



def train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, train_loader, test_loader, style_classes, feature_extractor, vocab_size, noise_scheduler, transform, args, tokenizer=None, text_encoder=None, lr_scheduler=None):
    
    if args.wandb == True:
        wandb.init(project="diffusionpen")
        #watch unet
        wandb.watch(model, log='all')
    
    print('Training started....')
    print('Epoch:', args.epochs)
    model.train()
    
    for epoch in range(args.epochs):
        
        pbar = tqdm(train_loader)
        losses = AvgMeter()
        for i, data in enumerate(pbar):
            
            images = data[0].to(args.device)
            transcr = data[1]
            s_id = data[2].to(args.device)
            style_images = data[3].to(args.device)
            cor_im = data[5].to(args.device)
            img_path = data[4]
            
            if args.model_name == 'wordstylist':
                #print('transcr', transcr)
                batch_word_embeddings = []
                for trans in transcr:
                    word_embedding = label_padding(trans) 
                    #print('word_embedding', word_embedding)
                    word_embedding = np.array(word_embedding, dtype="int64")
                    word_embedding = torch.from_numpy(word_embedding).long() 
                    batch_word_embeddings.append(word_embedding)
                text_features = torch.stack(batch_word_embeddings).to(args.device)
            else:
                text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=200).to(args.device)
            
            reshaped_images = style_images.reshape(-1, 3, 64, 256)
            
            if feature_extractor is not None:
                style_features = feature_extractor(reshaped_images).to(args.device)
            else:
                style_features = None
            
            if args.latent==True:
                with torch.no_grad():
                    latents = vae.module.encode(images).latent_dist.sample()
                    latents = latents * 0.18215
                images = latents
            
            t = diffusion.sample_timesteps(images.shape[0]).to(args.device)
            
            #print('t', t.shape)
            
            x_t, noise = noise_images(images, t, args)
            
            #print('x_t', x_t.shape)
            
            with torch.no_grad():
                context = text_encoder(**text_features.to("cuda:0")).last_hidden_state
            context = context.to(args.device)
            
            predicted_noise = model(x_t, timesteps=t, context=context, y=s_id, style_extractor=style_features)
            
            loss = mse_loss(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)
            losses.update(loss.item(), images.size(0))
            pbar.set_description(f"Epoch: {epoch}, Loss: {losses.avg:.4f}")
            
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            if i % args.save_interval == 0 and i != 0:
                torch.save(model.state_dict(), f'{args.save_path}/models/ckpt.pt')
                torch.save(optimizer.state_dict(), f'{args.save_path}/models/optim.pt')
                torch.save(ema_model.state_dict(), f'{args.save_path}/models/ema_ckpt.pt')
                print('Saved models')
            
            if i % args.test_interval == 0 and i != 0:
                
                with torch.no_grad():
                    #ema model sampling
                    ema_sampled_images = diffusion.sampling_loader(ema_model, test_loader, vae, n=images.size(0), x_text=transcr, labels=s_id, args=args, style_extractor=feature_extractor, noise_scheduler=noise_scheduler, mix_rate=None, cfg_scale=3, transform=transform, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder)
                    
                    #save images
                    save_images(ema_sampled_images, os.path.join(f'{args.save_path}/images', f'ema_sampled_images_{uuid.uuid4()}.png'), args, nrow=8)
                    #save_images(cor_im, os.path.join(f'{args.save_path}/images', f'cor_im_{uuid.uuid4()}.png'), args, nrow=8)
                    save_images(images, os.path.join(f'{args.save_path}/images', f'ground_truth_{uuid.uuid4()}.png'), args, nrow=8)
                    
                    if args.wandb == True:
                        wandb.log({"ema_sampled_images": wandb.Image(os.path.join(f'{args.save_path}/images', f'ema_sampled_images_{uuid.uuid4()}.png'))})
                        #wandb.log({"cor_im": wandb.Image(os.path.join(f'{args.save_path}/images', f'cor_im_{uuid.uuid4()}.png'))})
                        wandb.log({"ground_truth": wandb.Image(os.path.join(f'{args.save_path}/images', f'ground_truth_{uuid.uuid4()}.png'))})
                        wandb.log({"loss": losses.avg})
                        wandb.log({"epoch": epoch})
                        wandb.log({"lr": optimizer.param_groups[0]['lr']})
    
    torch.save(model.state_dict(), f'{args.save_path}/models/ckpt.pt')
    torch.save(optimizer.state_dict(), f'{args.save_path}/models/optim.pt')
    torch.save(ema_model.state_dict(), f'{args.save_path}/models/ema_ckpt.pt')
    print('Saved models')
    
    if args.wandb == True:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='DiffusionPen')
    parser.add_argument('--data_path', type=str, default='/cluster/datastore/aniketag/allData/diffPen/')
    parser.add_argument('--save_path', type=str, default='/cluster/datastore/aniketag/allData/diffPen/saved_GNHK')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--img_size', type=tuple, default=(64, 256))
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='diffusionpen')
    parser.add_argument('--train_mode', type=str, default='train')
    parser.add_argument('--sampling_mode', type=str, default='single_sampling')
    parser.add_argument('--load_check', type=bool, default=False)
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--color', type=bool, default=False)
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--test_interval', type=int, default=1000)
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--stable_dif_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--style_path', type=str, default='/cluster/datastore/aniketag/allData/diffPen/saved_IAM/models/ckpt.pt_161')
    parser.add_argument('--img_feat', type=bool, default=False)
    parser.add_argument('--unet', type=str, default='unet_latent')
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--mix_rate', type=float, default=0.5)
    parser.add_argument('--dataparallel', type=bool, default=True)
    
    args = parser.parse_args()
    
    print('torch version', torch.__version__)
    
    print('loading GNHK')
    
    save_file = os.path.join(args.data_path, 'saved_GNHK_data', 'train_word_GNHK.pt')
    
    transform = transforms.Compose([
                        #transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1, fill=255),
                        transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #transforms.Normalize((0.5,), (0.5,)),  #
                        ])

    
    if not os.path.exists(save_file):
        print('\t creating data from scratch')
        train_data = GNHK_Dataset(args.data_path,'train', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
        torch.save(train_data, save_file)
    else:
        print('\t data already present!!!:', save_file)
    
    print('data loaded from file', save_file)
    train_data = torch.load(save_file)
    
    test_data = GNHK_Dataset(args.data_path,'test','train', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
    
    #train_data = myDataset(iam_folder, 'train', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
    #test_data = myDataset(iam_folder, 'test', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)

    
    
    print('Number of writers', len(train_data.writers))
    
    print('train data', len(train_data))
    print('test data', len(test_data))
    
    #save writers in json
    writers = train_data.writers
    writers_dict = {writers[i]: i for i in range(len(writers))}
    json_dict = json.dumps(writers_dict)
    f = open("writers_dict_train.json","w")
    f.write(json_dict)
    f.close()
    
    style_classes = len(writers)
    print('number of style classes', style_classes)
    
    print('Vocab size: ', vocab_size)
    
    device_ids = [0, 1]
    
    if args.color == False:
        transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print('using dataparallel with device:', device_ids)
    
    if args.model_name == 'diffusionpen':
        
        tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        text_encoder = CanineModel.from_pretrained("google/canine-s").to("cuda:0")
        
    else:
        custom_cache = "./tokenizer/"

        tokenizer = CanineTokenizer.from_pretrained("google/canine-c", cache_dir=custom_cache)
        text_encoder = CanineModel.from_pretrained("google/canine-s").to("cuda:0")
    
    if args.unet=='unet_latent':
        unet = UNetModel(image_size = args.img_size, in_channels=args.channels, model_channels=args.emb_dim, out_channels=args.channels, num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes, context_dim=args.emb_dim, vocab_size=vocab_size, text_encoder=text_encoder, args=args).to("cuda")
    
    if args.dataparallel:
        unet = DataParallel(unet, device_ids=device_ids)
    
    print('unet parameters')
    print('unet', sum(p.numel() for p in unet.parameters() if p.requires_grad))
    
    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)
    lr_scheduler = None 

    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)

    #load from last checkpoint
    
    if args.load_check==True:
        unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt'))
        optimizer.load_state_dict(torch.load(f'{args.save_path}/models/optim.pt'))
        ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt'))
        print('Loaded models and optimizer')
    
    if args.latent==True:
        print('VAE is true')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = DataParallel(vae, device_ids=device_ids)
        vae = vae.to(args.device)
        # Freeze vae and text_encoder
        vae.requires_grad_(False)
    else:
        vae = None

    #add DDIM scheduler from huggingface
    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")
    
    #### STYLE ####
    feature_extractor = ImageEncoder(model_name='mobilenetv2_100', num_classes=0, pretrained=True, trainable=True)
    PATH = args.style_path 
    
    state_dict = torch.load(PATH, map_location=args.device)
    model_dict = feature_extractor.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(state_dict)
    feature_extractor.load_state_dict(model_dict)
    feature_extractor = DataParallel(feature_extractor, device_ids=device_ids)
    feature_extractor = feature_extractor.to(args.device)
    feature_extractor.requires_grad_(False)
    feature_extractor.eval()
    
    if args.train_mode == 'train':
        train(diffusion, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader, test_loader, style_classes, feature_extractor, vocab_size, ddim, transform, args, tokenizer=tokenizer, text_encoder=text_encoder, lr_scheduler=lr_scheduler)
    
    elif args.train_mode == 'sampling':
        
        print('Sampling started....')
        
        unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt', map_location=args.device))
        print('unet loaded')
        unet.eval()
        
        ema = EMA(0.995)
        ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
        ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt'))
        ema_model.eval()
    
        
            
        if args.sampling_mode == 'single_sampling':
            x_text = ['text', 'word']
            for x_text in x_text:
                print('Word:', x_text)
                s = random.randint(0, 339) #index for style class
                
                print('style', s)
                labels = torch.tensor([s]).long().to(args.device)
                ema_sampled_images = diffusion.sampling(ema_model, vae, n=len(labels), x_text=x_text, labels=labels, args=args, style_extractor=feature_extractor, noise_scheduler=ddim, transform=transform, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder, run_idx=None)  
                save_single_images(ema_sampled_images, os.path.join(f'./image_samples/', f'{x_text}_style_{s}.png'), args)

        
        elif args.sampling_mode == 'paragraph':
            print('Sampling paragraph')
            #make the code to generate lines
            lines = 'In this work , we focus on style variation . We present a novel method to control the style of the text . Our method is able to mimic various writing styles .'
            fakes= []
            gap = np.ones((64, 16))
            max_line_width = 900
            total_char_count = 0
            avg_char_width = 0
            current_line_width = 0
            longest_word_length = max(len(word) for word in lines.strip().split(' '))
            #print('longest_word_length', longest_word_length)
            #s = random.randint(0, 339)#.long().to(args.device)
            #s = random.randint(0, 161)#.long().to(args.device)
            s = 12 #25 #129 #201
            for word in lines.strip().split(' '):
                print('Word:', word)
                print('Style:', s)
                labels = torch.tensor([s]).long().to(args.device)
                ema_sampled_images = diffusion.sampling(ema_model, vae, n=len(labels), x_text=word, labels=labels, args=args, style_extractor=feature_extractor, noise_scheduler=ddim, transform=transform, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder, clip_model=None, run_idx=None)  
                #print('ema_sampled_images', ema_sampled_images.shape)
                image = ema_sampled_images.squeeze(0)
                
                im = torchvision.transforms.ToPILImage()(image)
                #reshape to height 32
                im = im.convert("L")
                #save im
                
                #if len(word) < 4:
                    
                im = crop_whitespace_width(im)
                
                im = Image.fromarray(im)
                if len(word) == longest_word_length:
                    max_word_length_width = im.width
                    print('max_word_length_width', max_word_length_width)
                #im.save(f'./_REBUTTAL/{word}.png')
                # Calculate aspect ratio
                aspect_ratio = im.width / im.height
                
                im = np.array(im)
                #im = np.array(resized_img)
                
                fakes.append(im)
            
            # Calculate the scaling factor based on the longest word
            #find the average character width of the max length word
            
            avg_char_width = max_word_length_width / longest_word_length
            print('avg_char_width', avg_char_width)
            #scaling_factor = avg_char_width / (32 * aspect_ratio)  # Aspect ratio of an average character

            # Scale and pad each word
            scaled_padded_words = []
            max_height = 64  # Defined max height for all images
            
            for word, img in zip(lines.strip().split(' '), fakes):
                
                img_pil = Image.fromarray(img)
                as_ratio = img_pil.width / img_pil.height
                #scaled_width = int(scaling_factor * len(word))#) * as_ratio * max_height)
                scaled_width = int(avg_char_width * len(word))
                
                scaled_img = img_pil.resize((scaled_width, int(scaled_width / as_ratio)))
                print(f'Word {word} - scaled_img {scaled_img.size}')
                # Padding
                #if word is in punctuation:
                if word in punctuation:
                    #rescale to height 10
                    w_punc = scaled_img.width
                    h_punc = scaled_img.height
                    as_ratio_punct = w_punc / h_punc
                    if word == '.':
                        scaled_img = scaled_img.resize((int(5 * as_ratio_punct), 5))
                    else:
                        scaled_img = scaled_img.resize((int(13 * as_ratio_punct), 13))
                    #pad on top and leave the image in the bottom
                    padding_bottom = 10
                    padding_top = max_height - scaled_img.height - padding_bottom# All padding goes on top
                      # No padding at the bottom

                    # Apply padding
                    padded_img = np.pad(scaled_img, ((padding_top, padding_bottom), (0, 0)), mode='constant', constant_values=255)
                else:
                    if scaled_img.height < max_height:
                        padding = (max_height - scaled_img.height) // 2
                        #print(f'Word {word} - padding: {padding}')
                        padded_img = np.pad(scaled_img, ((padding, max_height - scaled_img.height - padding), (0, 0)), mode='constant', constant_values=255)
                    else:
                        #resize to max height while maintaining aspect ratio
                        #ar = scaled_img.width / scaled_img.height
                        
                        scaled_img = scaled_img.resize((int(max_height * as_ratio) - 4, max_height - 4))
                        padding = (max_height - scaled_img.height) // 2
                        #print(f'Word {word} - padding: {padding}')
                        padded_img = np.pad(scaled_img, ((padding, max_height - scaled_img.height - padding), (0, 0)), mode='constant', constant_values=255)
                        
                    #padded_img = np.array(scaled_img)
                #print('padded_img', padded_img.shape)
                scaled_padded_words.append(padded_img)

            # Create a gap array (white space)
            height = 64  # Fixed height for all images
            gap = np.ones((height, 16), dtype=np.uint8) * 255  # White gap

            # Concatenate images with gaps
            sentence_img = gap  # Start with a gap
            lines = [] 
            line_img = gap
            # Concatenate images with gaps
            '''
            sentence_img = gap  # Start with a gap
            for img in scaled_padded_words:
                #print('img', img.shape)
                sentence_img = np.concatenate((sentence_img, img, gap), axis=1)
            '''
            
            for img in scaled_padded_words:
                img_width = img.shape[1] + gap.shape[1]

                if current_line_width + img_width < max_line_width:
                    # Add the image to the current line
                    if line_img.shape[0] == 0:
                        line_img = np.ones((height, 0), dtype=np.uint8) * 255  # Start a new line
                    line_img = np.concatenate((line_img, img, gap), axis=1)
                    current_line_width += img_width #+ gap.shape[1]
                    #print('current_line_width if', current_line_width)
                    # Check if adding this image exceeds the max line width
                else:
                    # Pad the current line with white space to max_line_width
                    remaining_width = max_line_width - current_line_width
                    line_img = np.concatenate((line_img, np.ones((height, remaining_width), dtype=np.uint8) * 255), axis=1)
                    lines.append(line_img)

                    # Start a new line with the current word
                    line_img = np.concatenate((gap, img, gap), axis=1)
                    current_line_width = img_width #+ 2 * gap.shape[1]
                    #print('current_line_width else', current_line_width)
            # Add the last line to the lines list
            if current_line_width > 0:
                # Pad the last line to max_line_width
                remaining_width = max_line_width - current_line_width
                line_img = np.concatenate((line_img, np.ones((height, remaining_width), dtype=np.uint8) * 255), axis=1)
                lines.append(line_img)
                
            # # Concatenate all lines to form a paragraph, pad them if necessary
            # max_height = max([line.shape[0] for line in lines])
            # paragraph_img = np.ones((0, max_line_width), dtype=np.uint8) * 255
            # for line in lines:
            #     if line.shape[0] < max_height:
            #         padding = (max_height - line.shape[0]) // 2
            #         line = np.pad(line, ((padding, max_height - line.shape[0] - padding), (0, 0)), mode='constant', constant_values=255)
                
            #     #print the shapes
            #     print('line shape', line.shape)
            #print('paragraph shape', paragraph_img.shape)
            paragraph_img = np.concatenate((lines), axis=0)

                
            paragraph_image = Image.fromarray(paragraph_img)
            paragraph_image = paragraph_image.convert("L")    
            
            paragraph_image.save(f'paragraph_style_{s}.png')

    
if __name__ == "__main__":
    main()