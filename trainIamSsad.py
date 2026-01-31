import os
# set visible GPUs only to 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CACHE_ROOT = "/cluster/datastore/aniketag/allData/pytrochWeights/"

# Create dirs
os.makedirs(CACHE_ROOT, exist_ok=True)
os.makedirs(os.path.join(CACHE_ROOT, ".xdg-cache"), exist_ok=True)
os.makedirs(os.path.join(CACHE_ROOT, ".mpl-cache"), exist_ok=True)

# Critical HF cache envs (these are the ones your log indicates are missing)
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_ROOT   # <- important
os.environ["HF_HUB_CACHE"] = CACHE_ROOT            # <- some versions read this
os.environ["HF_HOME"] = CACHE_ROOT                 # safe to set too

# Torch cache
os.environ["TORCH_HOME"] = CACHE_ROOT

# Other caches
os.environ["XDG_CACHE_HOME"] = os.path.join(CACHE_ROOT, ".xdg-cache")
os.environ["MPLCONFIGDIR"]   = os.path.join(CACHE_ROOT, ".mpl-cache")

# Last-resort hammer if something still expands '~' to /home/aniket
# (use only if the above still fails)
# os.environ["HOME"] = CACHE_ROOT

# Optional: programmatic override (newer huggingface_hub)
try:
    from huggingface_hub import set_cache_home, scan_cache_dir
    set_cache_home(CACHE_ROOT)
    print("HF cache dir:", scan_cache_dir().cache_dir)
except Exception:
    pass

print("Using cache dir:", CACHE_ROOT)

import logging

logging.basicConfig(
    #format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
    #datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('./logs/train.log'),  # Add a FileHandler
        logging.StreamHandler()  # Add a StreamHandler for console output
    ]
)
logger = logging.getLogger('')
#logger = logging.getLogger('wordStylistGenerationLogs2')
logger.info('--- IamDiff training ---')



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
from unetSSAD import UNetModel
import wandb
from torchvision import transforms
from utils.iam_dataset import IAMDataset
from utils.GNHK_dataset import GNHK_Dataset
from utils.auxilary_functions import *
from torchvision.utils import save_image
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer

# ---- ABSOLUTE TOP OF trainGNHK.py ----

# ---- only now import timm / your modules ----
import timm


torch.cuda.empty_cache()
from feature_extractor import ImageEncoder,ImageEncoderOriginal


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
            pbar = tqdm(test_loader,disable=True)
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
                            five_styles = random.sample(matching_lines2, 5)
                        #print('five_styles', five_styles)
                        #cor_image
                        fheight, fwidth = 64, 256
                        root_path = './iam_data/words'
                        cor_im = False
                        if cor_im == True:
                            cor_image = Image.open(os.path.join(root_path, cor_image_random[0][0])).convert('RGB') #['a05/a05-089/a05-089-00-05.png', '000', 'debate']
                            (cor_image_width, cor_image_height) = cor_image.size
                            cor_image = cor_image.resize((int(cor_image_width * 64 / cor_image_height), 64))
                            (cor_image_width, cor_image_height) = cor_image.size
                            
                            if cor_image_width < 256:
                                outImg = ImageOps.pad(cor_image, size=(256, 64), color= "white")#, centering=(0,0)) uncommment to pad right
                                cor_image = outImg
                            
                            else:
                                #reduce image until width is smaller than 256
                                while cor_image_width > 256:
                                    cor_image = image_resize_PIL(cor_image, width=cor_image_width-20)
                                    (cor_image_width, cor_image_height) = cor_image.size
                                cor_image = centered_PIL(cor_image, (64, 256), border_value=255.0)
                                    
                            cor_im_tens = transform(cor_image).to(args.device)
                            #print('cor image', cor_im_tens.shape)
                            cor_im_tens = cor_im_tens.unsqueeze(0)
                            cor_images = vae.module.encode(cor_im_tens.to(torch.float32)).latent_dist.sample()
                            cor_images = cor_images * 0.18215
                            
                        st_imgs = []
                        grid_imgs = []
                        for im_idx, random_f in enumerate(five_styles):
                            file_path = os.path.join(root_path, random_f[0])
                            #print('file_path', file_path)
                            
                            try:
                                img_s = Image.open(file_path).convert('RGB')
                            except ValueError:
                                # Handle the exception (e.g., print an error message)
                                print(f"Error loading image from {file_path}")
                                
                                # Find a replacement image that is not corrupted
                                replacement_idx = (im_idx + 1) % 5
                                replacement_f = five_styles[replacement_idx]
                                name = replacement_f[0] #.split(',')[1]
                                replacement_file_path = os.path.join(root_path, name)
                                img_s = Image.open(replacement_file_path).convert('RGB')
                                
                            (img_width, img_height) = img_s.size
                            img_s = img_s.resize((int(img_width * 64 / img_height), 64))
                            (img_width, img_height) = img_s.size
                            
                            if img_width < 256:
                                outImg = ImageOps.pad(img_s, size=(256, 64), color= "white")#, centering=(0,0)) uncommment to pad right
                                img_s = outImg
                            
                            else:
                                #reduce image until width is smaller than 256
                                while img_width > 256:
                                    img_s = image_resize_PIL(img_s, width=img_width-20)
                                    (img_width, img_height) = img_s.size
                                img_s = centered_PIL(img_s, (64, 256), border_value=255.0)
                            #make grid of all 5 images
                            #img_s = img_s.convert('L')
                            transform_tensor = transforms.ToTensor()
                            grid_im = transform_tensor(img_s)
                            grid_imgs += [grid_im]
                            
                            img_tens = transform(img_s).to(args.device)#.unsqueeze(0)
                            st_imgs += [img_tens]
                            #style_features = style_extractor(style_images).to(args.device)
                            #img_tensor = img_tensor.to(args.device)
                        s_imgs = torch.stack(st_imgs).to(args.device)
                        style_images = torch.cat((style_images, s_imgs)) if style_images is not None else s_imgs
                        
                        grid_imgs = torch.stack(grid_imgs).to(args.device)
                        
                        
                        style_images = style_images.to(args.device)
                        
                    
                    #save style images
                    style_images = style_images.reshape(-1, 3, 64, 256)
                    style_features = style_extractor(style_images).to(args.device)
                    # style_features = torch.stack(style_featur, dim=0) #We get [320, 5, 2048]
                    #print('style features', style_features.shape)
                    #style_features = style_features.reshape(n, -1).to(args.device)
            else:
                style_images = None
                style_features = None            
            if args.latent == True:
                x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
                if cor_im == True:
                    x_noise = torch.randn(cor_images.shape).to(args.device)
                
                    timesteps = torch.full((cor_images.shape[0],), 999, device=args.device, dtype=torch.long)
                    
                    noisy_images = noise_scheduler.add_noise(
                        cor_images, x_noise, timesteps
                    )
                    x = noisy_images
                 
            else:
                x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
            
            #scheduler
            noise_scheduler.set_timesteps(50)
            for time in noise_scheduler.timesteps:
                
                t_item = time.item()
                t = (torch.ones(n) * t_item).long().to(args.device)

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

def readFlags(args):
    
    with open(args.stopFlag,"r") as f:
        stopValue = int(f.readline())
    
    return stopValue


def train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, test_loader, num_classes, style_extractor, vocab_size, noise_scheduler, transforms, args, tokenizer=None, text_encoder=None, lr_scheduler=None):
    model.train()
    loss_meter = AvgMeter()
    print('Training started....')
    
    for epoch in range(args.epochs):
        print('Epoch:', epoch)
        pbar = tqdm(loader,disable=True)
        style_feat = []
        for i, data in enumerate(pbar):
            images = data[0].to(args.device)
            transcr = data[1]
            s_id = data[2].to(args.device)
            style_images = data[3].to(args.device)
            
            # image and style size
            
            #print('images', images.shape, 'style images', style_images.shape)
            #logger.info('Epoch: %s, Iteration: %s, images: %s, style images: %s', epoch, i, images.shape, style_images.shape)
            stopValue = readFlags(args)

            if stopValue == 0:
                logger.info('Stopping Epoch stopValue:%s',stopValue)
                print('Stopping Epoch stopValue:',stopValue)
                exit()

            """
            
            INFO:root:Epoch: 0, Iteration: 0, images: torch.Size([512, 3, 64, 256]), style images: torch.Size([512, 5, 3, 64, 256])
            INFO:root:Epoch: 0, Iteration: 1, images: torch.Size([512, 3, 64, 256]), style images: torch.Size([512, 5, 3, 64, 256])

            
            """
            
            
            if args.model_name == 'wordstylist':
                batch_word_embeddings = []
                for trans in transcr:
                    word_embedding = label_padding(trans, num_tokens) 
                    word_embedding = np.array(word_embedding, dtype="int64")
                    word_embedding = torch.from_numpy(word_embedding).long() 
                    batch_word_embeddings.append(word_embedding)
                text_features = torch.stack(batch_word_embeddings)
            else:
                text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=40).to(args.device)
            
            if style_extractor is not None:
                reshaped_images = style_images.reshape(-1, 3, 64, 256)
                
                #print('0....sssstyle features', reshaped_images.shape)

                style_features = style_extractor(reshaped_images)
                #print('1....sssstyle features', style_features.shape)
                # 1....sssstyle features torch.Size([2560, 1280])
            else:
                style_features = None

            """
            images torch.Size([2, 3, 64, 256]) style images torch.Size([2, 5, 3, 64, 256])
            0....sssstyle features torch.Size([10, 3, 64, 256])
            1....sssstyle features torch.Size([10, 1280])
            """


            if args.latent == True:
                images = vae.module.encode(images.to(torch.float32)).latent_dist.sample()
                images = images * 0.18215
                latents = images
            
            noise = torch.randn(images.shape).to(images.device)
            # Sample a random timestep for each image
            num_train_timesteps = diffusion.noise_steps
            
            timesteps = torch.randint(
                0, num_train_timesteps,
                (images.shape[0],), device=images.device
            ).long()
            
            # Add noise to the clean images according to the noise magnitude
            # at each timestep (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(
                images, noise, timesteps
            )
            x_t = noisy_images
            t = timesteps
            
            if np.random.random() < 0.1:
                labels = None
            
            predicted_noise = model(x_t, timesteps=t, context=text_features, y=s_id, style_extractor=style_features)
            
            loss = mse_loss(noise, predicted_noise)
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            ema.step_ema(ema_model, model)

            count = images.size(0)
            loss_meter.update(loss.item(), count)
            pbar.set_postfix(MSE=loss_meter.avg)
            
            if lr_scheduler is not None:
                lr_scheduler.step()
    
        if epoch % 10 == 0:
            labels = torch.arange(16).long().to(args.device)
            n=len(labels)

            print(" Saving image at location: ", os.path.join(args.save_path, 'images', f"{epoch}_ema.jpg"))
            if args.sampling_word == True:
                #generates the word "text" in 16 different styles
                words = ['text']
                for x_text in words: 
                    ema_sampled_images = diffusion.sample(ema_model, vae, n=n, x_text=x_text, labels=labels, args=args)
                    
                    epoch_n = epoch 
                    sampled_ema = save_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{x_text}_{epoch_n}_ema.jpg"), args)
            else:
                #generates a batch of words
                ema_sampled_images = diffusion.sampling_loader(ema_model, test_loader, vae, n=n, x_text=None, labels=labels, args=args, style_extractor=style_extractor, noise_scheduler=noise_scheduler, transform=transforms, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder)
                epoch_n = epoch 
                sampled_ema = save_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{epoch_n}_ema.jpg"), args)
        
            if args.wandb_log==True:
                wandb_sampled_ema= wandb.Image(sampled_ema, caption=f"{x_text}_{epoch}")
                wandb.log({f"Sampled images": wandb_sampled_ema})
            
            if 1:
                print(" Saving model at location: ", os.path.join(args.save_path,"models", "ckpt.pt"))  
                torch.save(model.state_dict(), os.path.join(args.save_path,"models", "ckpt.pt"))
                torch.save(ema_model.state_dict(), os.path.join(args.save_path,"models", "ema_ckpt.pt"))
                torch.save(optimizer.state_dict(), os.path.join(args.save_path,"models", "optim.pt"))   


def main():
    
    import sys
    print("------ Python Script Execution Info ------")
    print(f"Script Name: {os.path.basename(__file__)}")
    print(f"Full Path: {os.path.abspath(__file__)}")
    print("Arguments passed to script:", sys.argv)
    print("------------------------------------------")
    
    '''Main function'''
    base = "/cluster/datastore/aniketag/allData/diffPen/"
    modelBasePath = "/cluster/datastore/aniketag/allData/diffPen/style_models/"

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--model_name', type=str, default='SsadIAM', help='diffusionpen or wordstylist (previous work)')
    parser.add_argument('--level', type=str, default='word', help='word, line')
    parser.add_argument('--img_size', type=int, default=(64, 256))  
    parser.add_argument('--dataset', type=str, default='iam', help='iam') 
    #UNET parameters
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default=base+'/Ssad_IAM_model_path/') 
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--unet', type=str, default='unet_latent', help='unet_latent')
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--dataparallel', type=bool, default=False)
    parser.add_argument('--load_check', type=bool, default=True)
    parser.add_argument('--sampling_word', type=bool, default=False) 
    parser.add_argument('--mix_rate', type=float, default=None)
    parser.add_argument('--style_path', type=str, default=modelBasePath+'/iam_style_diffusionpen.pth')
    #parser.add_argument('--style_path', type=str, default=modelBasePath+'/gnhk_style_diffusionpen.pth')

    #parser.add_argument('--stable_dif_path', type=str, default='./stable-diffusion-v1-5')
    parser.add_argument('--stable_dif_path', type=str, default="/cluster/datastore/aniketag/allData/supportingSoftwares/stableDiffusion/", help='path to stable diffusion')

    parser.add_argument('--train_mode', type=str, default='train', help='train, sampling')
    parser.add_argument('--sampling_mode', type=str, default='single_sampling', help='single_sampling (generate single image), paragraph (generate paragraph)')
    parser.add_argument('--partialLoad', type=int, default=0)
    parser.add_argument('--stopFlag', type=str, default = "./flags/stopFlagSsad.txt",help ="flag to stop program") # partialLoad

    
    args = parser.parse_args()

    print("\n Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")    

    
    print('torch version', torch.__version__)
    
    if args.wandb_log==True:
        runs = wandb.init(project='DiffusionPen', entity='name_entity', name=args.dataset, config=args)

        wandb.config.update(args)
    
    #create save directories
    setup_logging(args)

    ############################ DATASET ############################
    transform = transforms.Compose([
                        #transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1, fill=255),
                        transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #transforms.Normalize((0.5,), (0.5,)),  #
                        ])
    
    if args.dataset == 'iam':
        print('loading IAM')
        iam_folder = './iam_data/words'
        myDataset = IAMDataset
        style_classes = 339
        if args.level == 'word':
            train_data = myDataset(iam_folder, 'train', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
        else:

            train_data = myDataset(iam_folder, 'train', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
            test_data = myDataset(iam_folder, 'test', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
        print('train data', len(train_data))
        
        test_size = args.batch_size
        rest = len(train_data) - test_size
        test_data, _ = random_split(train_data, [test_size, rest], generator=torch.Generator().manual_seed(42))
        
        print('test data', len(test_data))
        print('number of style classes', style_classes)
        

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    character_classes = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
    
    ######################### MODEL #######################################
    if args.model_name == 'wordstylist':
        vocab_size = len(character_classes) + 2
        print('vocab size', vocab_size)
    else:
        vocab_size = len(character_classes)
    print('Vocab size: ', vocab_size)
    
    if args.dataparallel==True:
        device_ids = [0,1]
        print('using dataparallel with device:', device_ids)
    else:
        idx = int(''.join(filter(str.isdigit, args.device)))
        device_ids = [idx]
    #unet = unet.to(args.device)

    if args.model_name == 'diffusionpen':
        tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        text_encoder = CanineModel.from_pretrained("google/canine-c")
        text_encoder = nn.DataParallel(text_encoder, device_ids=device_ids)
        text_encoder = text_encoder.to(args.device)
        
    else:
        custom_cache = "./tokenizer/"

        tokenizer = CanineTokenizer.from_pretrained("google/canine-c", cache_dir=custom_cache)
        text_encoder = CanineModel.from_pretrained("google/canine-c", cache_dir=custom_cache)
    
    if args.unet=='unet_latent':
        unet = UNetModel(image_size = args.img_size, in_channels=args.channels, model_channels=args.emb_dim, out_channels=args.channels, num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes, context_dim=args.emb_dim, vocab_size=vocab_size, text_encoder=text_encoder, args=args)#.to(args.device)
    
    unet = DataParallel(unet, device_ids=device_ids)
    unet = unet.to(args.device)
    
    #print('unet parameters')
    #print('unet', sum(p.numel() for p in unet.parameters() if p.requires_grad))
    
    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)
    lr_scheduler = None 

    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)

    #load from last checkpoint
    # diffusionpen_GNHK_model_path
    if args.load_check==True:
        # 
        # /cluster/datastore/aniketag/allData/diffPen/diffusionpen_GNHK_model_path/models/ema_ckpt.pt
        
        ckptPath = "/cluster/datastore/aniketag/allData/diffPen//Ssad_IAM_model_path/models/ckpt.pt"
        emaPath = "/cluster/datastore/aniketag/allData/diffPen//Ssad_IAM_model_path/models/ema_ckpt.pt"
        opt = "/cluster/datastore/aniketag/allData/diffPen//Ssad_IAM_model_path/models//optim.pt"
        unet.load_state_dict(torch.load(ckptPath))

        #unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt'))
        #optimizer.load_state_dict(torch.load(f'{args.save_path}/models/optim.pt'))
        
        #optimizer.load_state_dict(torch.load(opt))

        #ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt'))
        ema_model.load_state_dict(torch.load(emaPath))

        print('Loaded models and optimizer')
    else:
        print('Exiting code!!!')
        exit()
    
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
    #feature_extractor = ImageEncoder(model_name='mobilenetv2_100', num_classes=0, pretrained=True, trainable=True)
    
    feature_extractor = ImageEncoderOriginal(model_name='mobilenetv2_100', num_classes=0, pretrained=True, trainable=True)
    
    PATH = args.style_path 
    print('Loading style extractor from:', PATH)
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

            
if __name__ == "__main__":
    main()
  
  
