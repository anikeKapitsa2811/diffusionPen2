
import os

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
        logging.FileHandler('./logs/IamSSADinference.log'),  # Add a FileHandler
        logging.StreamHandler()  # Add a StreamHandler for console output
    ]
)
logger = logging.getLogger('')
#logger = logging.getLogger('wordStylistGenerationLogs2')
logger.info('--- Iam SSAD inference ---')


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
from feature_extractor import ImageEncoder
from utils.iam_dataset import IAMDataset
#from utils.GNHK_dataset import GNHK_Dataset
#from utils.CVL_dataset import CVLDataset_style
from utils.iam_dataset import IAMDataset

from utils.auxilary_functions import *
from torchvision.utils import save_image
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer
#from configCVL import *

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

    def sampling(self, model, vae, n, x_text, labels, args, style_extractor, noise_scheduler,style_images=None, mix_rate=None, cfg_scale=3, transform=None, character_classes=None, tokenizer=None, text_encoder=None, run_idx=None):
        model.eval()
        tensor_list = []
        
        with torch.no_grad():
            #style_images = None
            text_features = x_text #[x_text]*n
            #print('text features', text_features.shape)
            text_features = tokenizer(text_features, padding="max_length", truncation=True, return_tensors="pt", max_length=40).to(args.device)
            #print('text features', text_features.shape)                    
            

            # Process style images if provided
            if style_extractor is not None and style_images is not None:
                reshaped_images = style_images.reshape(-1, 3, 64, 256)
                #print('0.reshaped_images', reshaped_images.shape)
                _,style_features = style_extractor(reshaped_images)
                style_features = style_features.to(args.device)
            else:
                style_features = None

                    
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





    def sampling_1(self, model, vae, n, x_text, labels,realLabels, args, style_extractor, noise_scheduler, style_images=None, mix_rate=None, cfg_scale=3, transform=None, character_classes=None, tokenizer=None, text_encoder=None, run_idx=None,img_path=None):
        model.eval()
        tensor_list = []
        allT = []  # original       
        #print("img_path:===",img_path) 
        baseImageName = [nm.split("/")[-1] for nm in img_path]
        allNoisyImages = dict()

        with torch.no_grad():
            # Tokenize batched text inputs
            text_features = tokenizer(x_text, padding="max_length", truncation=True, return_tensors="pt", max_length=40).to(args.device)
            
            # Process style images if provided
            if style_extractor is not None and style_images is not None:
                reshaped_images = style_images.reshape(-1, 3, 64, 256)
                style_features = style_extractor(reshaped_images).to(args.device)
            else:
                style_features = None
            
            # Prepare batched random noise input
            if args.latent:
                x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
            else:
                x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
            
            # Scheduler setup for batch
            noise_scheduler.set_timesteps(600)
            for time in noise_scheduler.timesteps:
                t_item = time.item()
                t = (torch.ones(n) * t_item).long().to(args.device)

                # Generate predictions for the batch
                noisy_residual = model(x, t, text_features, labels, original_images=style_images, mix_rate=mix_rate, style_extractor=style_features,baseImageName=baseImageName)
                prev_noisy_sample = noise_scheduler.step(noisy_residual, time, x).prev_sample
                x = prev_noisy_sample

                """
                if 0:#time.item() <200:
                    x = prev_noisy_sample

                    noisyImgClass=self.post_process_latents(x,vae)

                    #print("\n\t\t noisyImgClass.shape:",noisyImgClass.shape)
                    #logger.info(f"noisyImgClass.shape: {noisyImgClass.shape}") # torch.Size([8, 64, 256, 3])
                    
                    logits,predictions, confidences=callWriter(self.writer_model, noisyImgClass, args.device)
                    #print("\n\t predictions:",predictions)
                    #print("\n\t realLabels:",realLabels)
                    acc= calculate_writer_accuracy(predictions,realLabels)
                    print("\n\t acc:",acc," timrestep:",time.item())

                """
            
                if 0:#time.item() <200 and (time.item())%20== 0:

                    x = prev_noisy_sample

                    noisyImgClass=self.post_process_latents(x,vae)
                    
                    bs = noisyImgClass.shape[0]
                    #print("\n\t noisyImgClass.shape:",noisyImgClass.shape," bs:",bs," time.item():",time.item())

                    for tempIndx,tempImg in enumerate(noisyImgClass):
                        # generate random no between 0- 1
                        
                        if 1:#random.random() < 0.05:
                        
                            tempImgNm= baseImageName[int(tempIndx)]
                            allNoisyImages[tempImgNm+"_"+str(time.item())] = tempImg                

                            """
                                save tempImg
                            """
                            # pls code saving 
                            tempImg = tempImg.cpu().permute(2,0,1)#
                            tempImg = tempImg.unsqueeze(0)#.numpy()
                            #print("\n\t tempImg.shape:",tempImg.shape," tempIndx:",tempIndx," tempImgNm:",tempImgNm)
                            grid = torchvision.utils.make_grid(tempImg, padding=0)
                            ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
                            
                            # Normalize from [0,1] or other float values to [0,255]
                            ndarr = (ndarr * 255).clip(0, 255).astype('uint8')
                            im = Image.fromarray(ndarr)
                            
                            # save in ./noisyImageDump  using key value as a name 
                            dump_path = './noisyImageDump' #os.path.join(args.save_path, 'noisyImageDump')
                            os.makedirs(dump_path, exist_ok=True)
                            filename = f"{tempImgNm}_{time.item()}.png"
                            filepath = os.path.join(dump_path, filename)
                            #print("\n\t saving noisy image at:",filepath)
                            #logger.info(f"Saving noisy image to {filepath} with label {labels[tempIndx]} and text '{x_text[tempIndx]}'")
                            im.save(filepath)
            
        # Post-processing for batch outputs
        if args.latent:
            latents = 1 / 0.18215 * x
            images = vae.module.decode(latents).sample
            images = (images / 2 + 0.5).clamp(0, 1)
            allT.append(images)

            images = images.cpu().permute(0, 2, 3, 1).numpy()
            images = torch.from_numpy(images).permute(0, 3, 1, 2)
        else:
            images = (x.clamp(-1, 1) + 1) / 2
            images = (images * 255).type(torch.uint8)


        allT = torch.stack(allT)
        allT = allT.squeeze(0)


        return images,allT




def train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, test_loader, num_classes, style_extractor, vocab_size, noise_scheduler, transforms, args, tokenizer=None, text_encoder=None, lr_scheduler=None):
    model.train()
    loss_meter = AvgMeter()
    print('Training started....')
    
    for epoch in range(args.epochs):
        print('Epoch:', epoch)
        pbar = tqdm(loader)
        style_feat = []
        for i, data in enumerate(pbar):
            images = data[0].to(args.device)
            transcr = data[1]
            s_id = data[2].to(args.device)
            style_images = data[3].to(args.device)
            
            
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
                style_features = style_extractor(reshaped_images)
                
            else:
                style_features = None

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
            
            torch.save(model.state_dict(), os.path.join(args.save_path,"models", "ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join(args.save_path,"models", "ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path,"models", "optim.pt"))   



def save_images1(img_path,images, dump_path, args, x_texts, labels,dec_transcr1):
    """
    Save a batch of images with corresponding filenames.

    Args:
        images (torch.Tensor): Batch of images (shape: [batch_size, C, H, W]).
        dump_path (str): Base directory for saving the images.
        args (Namespace): Arguments containing settings (e.g., latent mode).
        x_texts (list): List of text strings corresponding to each image in the batch.
        labels (torch.Tensor): Tensor of style labels corresponding to each image in the batch.
    """
    # Ensure the dump directory exists
    os.makedirs(dump_path, exist_ok=True)

    # Iterate over the batch and save each image
    for idx, (image, text, label) in enumerate(zip(images, x_texts, labels)):
        # Convert the image tensor to PIL Image
        grid = torchvision.utils.make_grid(image.unsqueeze(0), padding=0)
        if args.latent:
            im = torchvision.transforms.ToPILImage()(grid)
            im = im.convert('RGB' if args.color else 'L')
        else:
            ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
            im = Image.fromarray(ndarr)

        # Construct the filename using text and style label
        
        baseName = img_path[idx]
        baseName=baseName.split("/")[-1]
        #print("\n\t 1.baseName=",baseName)
        
        try:
            
            #filename = f"{baseName}_{text}_{label.item()}_{dec_transcr1}.png"

            filename = f"{baseName}_{text}_{label.item()}.png"
            filename = baseName+".png"
        except Exception as e:
            #filename = f"{baseName}_{text}_{label.item()}.png"
            filename = baseName+".png"
        filepath = os.path.join(dump_path, filename)
        
        print("\n\t saving filepath =",filepath)#," label:",label," text:",text)#," dec_transcr1:",dec_transcr1[idx])
        #logger.info(f"Saving image to {filepath} with label {label} and text '{text}' dec_transcr1: {dec_transcr1[idx]}")
        # Save the image
        im.save(filepath)

    #print(f"Saved {len(images)} images to {dump_path}.")


def readFlags(args):
    
    with open(args.stopFlag,"r") as f:
        stopValue = int(f.readline())
    
    return stopValue


def main():
    
    import sys
    print("------ Python Script Execution Info ------")
    print(f"Script Name: {os.path.basename(__file__)}")
    print(f"Full Path: {os.path.abspath(__file__)}")
    print("Arguments passed to script:", sys.argv)
    print("------------------------------------------")

    
    from configCVL import dataset_folder, baseModelDir, styleClssifierModelPath

    '''Main function'''
    base = "/cluster/datastore/aniketag/allData/diffPen/"
    modelBasePath = "/cluster/datastore/aniketag/allData/diffPen/style_models/"

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4) 
    
    parser.add_argument('--model_name', type=str, default='diffusionpenIAM', help='diffusionpen or wordstylist (previous work)')
    parser.add_argument('--level', type=str, default='word', help='word, line')
    parser.add_argument('--img_size', type=int, default=(64, 256))  
    parser.add_argument('--partialLoad', type=int, default=0)  

    parser.add_argument('--dataset', type=str, default='iam', help='iam, gnhk') 
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
    parser.add_argument('--load_check', type=bool, default=False)
    parser.add_argument('--sampling_word', type=bool, default=False) 
    parser.add_argument('--mix_rate', type=float, default=None)
    
    parser.add_argument('--style_path', type=str, default=modelBasePath+'/iam_style_diffusionpen.pth')
    parser.add_argument('--stable_dif_path', type=str, default="/cluster/datastore/aniketag/allData/supportingSoftwares/stableDiffusion/", help='path to stable diffusion')

    parser.add_argument('--train_mode', type=str, default='sampling', help='train, sampling')
    parser.add_argument('--sampling_mode', type=str, default='single_sampling', help='single_sampling (generate single image), paragraph (generate paragraph)')
    parser.add_argument('--lang', type=str, default= "ENG",help = "language") 

    parser.add_argument('--stopFlag', type=str, default = "./flags/stopFlagSsad.txt",help ="flag to stop program") # partialLoad


    parser.add_argument('--dumpPath', type=str, default="/cluster/datastore/aniketag/allData/syntheticData/train/icdar2025/IAM/SSAD/iam1/")

    parser.add_argument('--dataset_folder', type=str, default='/cluster/datastore/aniketag/allData/wordStylist/allCrops_preprocess/', help='path to stable diffusion')


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
        
    elif args.dataset == 'gnhk':
        print('loading GNHK')
        myDataset = GNHK_Dataset
        dataset_folder =  '/cluster/datastore/aniketag/allData/GNHK/allCrops_preprocess/'
        style_classes = 515
        train_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #transforms.Normalize((0.5,), (0.5,)),  #
                            ])
        train_data = myDataset(dataset_folder, 'train', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=train_transform, args=args)
        test_size = args.batch_size
        rest = len(train_data) - test_size
        test_data, _ = random_split(train_data, [test_size, rest], generator=torch.Generator().manual_seed(42))
        
 
    elif args.dataset == 'CVL':
        print('loading CVL')
        myDataset = CVLDataset_style
        dataset_folder = args.dataset_folder

        style_classes = 283
        train_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #transforms.Normalize((0.5,), (0.5,)),  #
                            ])
        
        train_data = myDataset(dataset_folder, 'train', 'word', fixed_size=(1 * 64, 256),  transforms=train_transform, args=args)
        
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
        print('1. vocab size', vocab_size)
    else:
        vocab_size = len(character_classes)
    print('2.Vocab size: ', vocab_size)
    
    if args.dataparallel==True:
        device_ids = [3,4]
        print('using dataparallel with device:', device_ids)
    else:
        idx = int(''.join(filter(str.isdigit, args.device)))
        device_ids = [idx]
    #unet = unet.to(args.device)

    if 0:
        if args.model_name == 'diffusionpen':
            custom_cache = "./tokenizer/"
            
            # Define the custom cache directory
            custom_cache = "./tokenizer/"

            # Create the cache directory if it doesn't exist
            os.makedirs(custom_cache, exist_ok=True)

            # Load the tokenizer and model with the custom cache
            tokenizer = CanineTokenizer.from_pretrained("google/canine-c", cache_dir=custom_cache)
            text_encoder = CanineModel.from_pretrained("google/canine-c", cache_dir=custom_cache)

            # Move the model to the specified device
            text_encoder = text_encoder.to(args.device)        
            
        else:
            custom_cache = "./tokenizer/"

            tokenizer = CanineTokenizer.from_pretrained("google/canine-c", cache_dir=custom_cache)
            text_encoder = CanineModel.from_pretrained("google/canine-c", cache_dir=custom_cache)

    if args.model_name == 'diffusionpen':
        print('1. loading canine tokenizer and text encoder')
            # Define the custom cache directory
        custom_cache = "./tokenizer/"

        # Create the cache directory if it doesn't exist
        os.makedirs(custom_cache, exist_ok=True)

        # Load the tokenizer and model with the custom cache
        tokenizer = CanineTokenizer.from_pretrained("google/canine-c", cache_dir=custom_cache)
        text_encoder = CanineModel.from_pretrained("google/canine-c", cache_dir=custom_cache)
        text_encoder = nn.DataParallel(text_encoder, device_ids=device_ids)
        text_encoder = text_encoder.to(args.device)
        
    else:
        custom_cache = "./tokenizer/"

        tokenizer = CanineTokenizer.from_pretrained("google/canine-c", cache_dir=custom_cache)
        text_encoder = CanineModel.from_pretrained("google/canine-c", cache_dir=custom_cache)


    
    if args.unet=='unet_latent':
        unet = UNetModel(image_size = args.img_size, in_channels=args.channels, model_channels=args.emb_dim, out_channels=args.channels, num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes, context_dim=args.emb_dim, vocab_size=vocab_size, text_encoder=text_encoder, args=args)#.to(args.device)
    
    #unet = DataParallel(unet, device_ids=device_ids)
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

    #modelBasePath = "/cluster/datastore/aniketag/allData/diffPen/diffusionpen_CVL_model_path/"
    #modelBasePath = "/cluster/datastore/aniketag/allData/wordStylist/models/icdar2025/CVL/rebutal/" # rebutal models

    #modelBasePath = "/cluster/datastore/aniketag/allData/diffPen/models/icdar2025/rebutal/CVL//diffusionpen_CVL_model_path/"
    
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
    feature_extractor = ImageEncoder(model_name='mobilenetv2_100', num_classes=0, pretrained=False, trainable=False)
    PATH = args.style_path 
    
    state_dict = torch.load(PATH, map_location=args.device)
    
    model_dict = feature_extractor.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(state_dict)
    feature_extractor.load_state_dict(model_dict)
    print('Loaded style extractor from:', PATH)
    
    feature_extractor = DataParallel(feature_extractor, device_ids=device_ids)
    feature_extractor = feature_extractor.to(args.device)
    feature_extractor.requires_grad_(False)
    feature_extractor.eval()
    
    stopValue = readFlags(args)

    if stopValue == 0:
        #logger.info('Stopping Epoch stopValue:%s',stopValue)
        print('Stopping Epoch stopValue:',stopValue)
        exit()

        
    if args.train_mode == 'train':
        train(diffusion, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader, test_loader, style_classes, feature_extractor, vocab_size, ddim, transform, args, tokenizer=tokenizer, text_encoder=text_encoder, lr_scheduler=lr_scheduler)
    
    elif args.train_mode == 'sampling':
        
        print('Sampling started....')
        
        
        modelBasePath ="/cluster/datastore/aniketag/allData/diffPen//Ssad_IAM_model_path/"
        #modelBasePath = "/cluster/datastore/aniketag/allData/diffPen/models/icdar2025/rebutal/CVL//diffusionpen_CVL_model_path/"

        # check model file present
        print(" checking model file present:",os.path.exists(modelBasePath + '/models/ema_ckpt.pt'))
        #modelBasePath = "/cluster/datastore/aniketag/allData/diffPen/diffusionpen_iam_model_path/"

        state_dict = torch.load(modelBasePath +  '/models/ema_ckpt.pt', map_location=args.device)

        # Fix nested "module.text_encoder.module" structure
        
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        """
        state_dict = {}
        for k, v in checkpoint.items():
            new_key = k.replace("module.text_encoder.module", "module.text_encoder")  # fix nested module
            new_key = new_key.replace("module.", "")  # remove outer DataParallel prefix
            state_dict[new_key] = v
        """
        #state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}

        #unet = DataParallel(unet, device_ids=device_ids)

        unet.load_state_dict(state_dict)
        
        ema = EMA(0.995)
        ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
        checkpoint = torch.load(f'{modelBasePath}/models/ema_ckpt.pt', map_location=args.device)

        # Remove `module.` from parameter names if present
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}

        # Load the state dictionary into the model
        ema_model.load_state_dict(state_dict)
        print('Loaded EMA model from:', modelBasePath + '/models/ema_ckpt.pt')
        ema_model.eval()
        
        print("\n\targs.sampling_mode =",args.sampling_mode)
        
        dumpPath = args.dumpPath 
        os.makedirs(dumpPath, exist_ok=True)
        
        ############################################################################
        
        style_extractor = feature_extractor
        batchNo =0

        """
        with open("./writers_dict_train.json") as f:
            train_data_writer = json.load(f)
            
            # convert value to key and key to value in new dict
            reverse_wr_dict = {v: k for k, v in train_data_writer.items()}
            
            print("\n\t reverse_wr_dict.keys():",len(list(reverse_wr_dict.keys())),len(train_data_writer.keys()))
        """
        while 1:

            stopValue = readFlags(args)

            if stopValue == 0:
                #logger.info('Stopping Epoch stopValue:%s',stopValue)
                print('Stopping Epoch stopValue:',stopValue)
                exit()


            data = next(iter(train_loader))

            #print("\n\n\t batchNo:",batchNo)
            # /cluster/datastore/aniketag/newWordStylist/DiffusionPen2/flags/stopFlagSsad.txt
            with open("./flags/stopFlagSsad.txt","r") as f:
                flag = int(f.read())
            
            if flag == 0:
                exit()
            
            #try:
            

            images = data[0].to(args.device)
            transcr = data[1]
            s_id = data[2].to(args.device)
            #reals_sid = torch.tensor([int(reverse_wr_dict[int(sid)]) for sid in s_id]).to(args.device)

            style_images = data[3].to(args.device)
            
            #print("\n\t style_images.shape: =",style_images.shape)
            
            img_path = list(data[-2])
            #print("\n\t img_path =",img_path)
            
            #print("\n\t images =",images.shape," transcr:",transcr," s_id.shape:",s_id.shape," style_images.shape:",style_images.shape)

        
            if 0:#style_extractor is not None:
                reshaped_images = style_images.reshape(-1, 3, 64, 256)
                print('1.reshaped_images', reshaped_images.shape)

                _,style_features = style_extractor(reshaped_images)
                print('2.style_features', style_features[0].shape)
            else:
                style_features = None

            #ema_sampled_images = diffusion.sampling(ema_model, vae, n=len(labels), x_text=transcr, labels=s_id, args=args, style_extractor=feature_extractor, noise_scheduler=ddim, transform=transform, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder, run_idx=None)  
            
            
            # Example usage
            #style_images = ...  # Load or prepare style images as a tensor
            x_texts = transcr #["text1", "text2", "text3"]
            labels = s_id #torch.tensor([0, 1, 2]).long().to(args.device)

            ema_sampled_images = diffusion.sampling(
                ema_model,
                vae, 
                n=len(labels), 
                x_text=x_texts, 
                labels=labels, 
                args=args, style_extractor=feature_extractor, noise_scheduler=ddim, 
                style_images=style_images,
                transform=transform, 
                character_classes=None, 
                tokenizer=tokenizer, 
                text_encoder=text_encoder)
                #clip_model=None, 
                #run_idx=None)  

            dec_transcr1 = ""
            try:
                save_images1(img_path,ema_sampled_images, dumpPath, args, x_texts, labels,dec_transcr1)
            except Exception as e:
                print("\n\t exception in saving images:",e)
                import sys        
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("\n\t line number:", exc_tb.tb_lineno)


        
        

    
if __name__ == "__main__":
    main()
  
  
