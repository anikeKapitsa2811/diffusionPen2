import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from tqdm import tqdm
from torch import optim
import copy
import argparse
import uuid
import json
from diffusers import AutoencoderKL, DDIMScheduler
import random
from unet2 import UNetModel
import wandb
from torchvision import transforms
from feature_extractor import ImageEncoder
from utils.iam_dataset import IAMDataset
from utils.GNHK_dataset import GNHK_Dataset
from utils.auxilary_functions import *
from torchvision.utils import save_image
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer
import timm

class ImageEncoder(nn.Module):
    """
    Extract style embeddings from the layer *before* classification.
    """
    def __init__(self, model_name='resnet50', num_classes=0, pretrained=True, trainable=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        self.backbone = self.model.forward_features
        self.head = self.model.get_classifier()
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x, return_logits=False):
        feats = self.backbone(x)
        if return_logits:
            logits = self.head(feats)
            return feats, logits
        else:
            return feats

def main():
    '''Main function'''
    modelBasePath = "/cluster/datastore/aniketag/allData/diffPen/style_models/"
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_name', type=str, default='diffusionpen', help='diffusionpen or wordstylist (previous work)')
    parser.add_argument('--level', type=str, default='word', help='word, line')
    parser.add_argument('--img_size', type=int, default=(64, 256))
    parser.add_argument('--dataset', type=str, default='iam', help='iam, gnhk')
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='/cluster/datastore/aniketag/allData/syntheticData/train/icdar2025/IAM/diffusionPen/')
    parser.add_argument('--device', type=str, default='cuda:1')
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
    parser.add_argument('--lang', type=str, default="ENG", help="language")
    parser.add_argument('--backPropogation', type=int, default=1, help="finetune input")
    parser.add_argument('--styleEmbFineTune', type=int, default=0, help="finetune input")
    parser.add_argument('--partialLoad', type=int, default=0, help="finetune input")
    
    args = parser.parse_args()
    
    print('torch version', torch.__version__)
    
    # Initialize feature extractor
    feature_extractor = ImageEncoder(model_name='mobilenetv2_100', num_classes=0, pretrained=True, trainable=False)
    PATH = args.style_path
    state_dict = torch.load(PATH, map_location=args.device)
    model_dict = feature_extractor.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(state_dict)
    feature_extractor.load_state_dict(model_dict)
    feature_extractor = feature_extractor.to(args.device)
    feature_extractor.eval()
    print('feature extractor loaded from', PATH)
    
    writer_embeddings = {}
    
    # Load existing writer embeddings
    """
    embedding_path = "./writerEmbeding/writer_style_refs.pth"
    if os.path.exists(embedding_path):
        writer_embeddings = torch.load(embedding_path, map_location='cpu')
        print(f"Loaded existing writer embeddings with {len(writer_embeddings.keys())} writers")
    """
    if args.wandb_log:
        wandb.init(project='DiffusionPen', entity='name_entity', name=args.dataset, config=args)
        wandb.config.update(args)
    
    # Dataset setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if args.dataset == 'iam':
        print('loading IAM')
        iam_folder = './iam_data/words'
        myDataset = IAMDataset
        style_classes = 339
        if args.level == 'word':
            train_data = myDataset(iam_folder, 'train', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
            test_data = myDataset(iam_folder, 'test', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
            val_data = myDataset(iam_folder, 'val', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
            
        print('train data', len(train_data))
        print('test data', len(test_data))
        print('val data', len(val_data))
        
        merged_dataset = ConcatDataset([train_data])
        merged_loader = DataLoader(merged_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print("Total batches in merged_loader:", len(merged_dataset))
        
        character_classes = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
        
        with open("./writers_dict_train.json") as f:
            train_data_writer = json.load(f)
        with open("./writers_dict_test.json") as f:
            test_data_writer = json.load(f)
        
        combined_writer_dict = {}
        for k, v in train_data_writer.items():
            combined_writer_dict[k] = v
        for k, v in test_data_writer.items():
            if k not in combined_writer_dict.keys():
                combined_writer_dict[k] = v
        
        unqWriterDict = {}
        rowNo = 0
        
            
            
        for data in merged_loader:
            images = data[0].to(args.device)
            transcr = data[1]
            try:
                s_id = data[2].to(args.device)
            except Exception as e:
                s_id = data[2]
            
            """ write data in .txt file"""
            ## pls check and correct the line below
            
            #with open("./writerEmbeding/train.txt", "w") as f:
            
            #f.close()
            
            reals_sid = []
            for sid in s_id:
                writer_id = int(sid.item())
                reals_sid.append(writer_id)
                unqWriterDict[str(sid.item())] = 1
                rowNo += 1

                if rowNo % 10000 == 0:
                    print("\n\t sid:", sid.item())
                
                
            with open("./writerEmbeding/train.txt", "a") as f:  # Append mode
                for i in range(len(s_id)):
                    writer_id1 = str(s_id[i].item())
                    image_path1 = data[-2][i]
                    image_path1 = image_path1.split("/")[-1]
                    text1 = transcr[i]
                    f.write(f"{writer_id1},{image_path1},{text1}\n")

                
            if not reals_sid:
                continue
            reals_sid = torch.tensor(reals_sid).to(args.device)
            
            style_images = data[3].to(args.device)
            style_images = style_images[:len(reals_sid), 0, :, :, :]
            img_path = list(data[-2])

            batch_writer_ids = [int(writer_id.item()) for writer_id in reals_sid]
            new_writer_ids = [wid for wid in batch_writer_ids if wid not in writer_embeddings]
            if not new_writer_ids:
                continue

            mask = torch.tensor([wid in new_writer_ids for wid in batch_writer_ids]).to(args.device)
            if not mask.any():
                continue

            selected_images = style_images[mask]

            with torch.no_grad():
                feats = feature_extractor(selected_images)

            feat_idx = 0
            for i, writer_id in enumerate(batch_writer_ids):
                if writer_id in new_writer_ids:
                    if writer_id not in writer_embeddings:
                        writer_embeddings[writer_id] = []
                    writer_embeddings[writer_id].append(feats[feat_idx].cpu())
                    feat_idx += 1
            print("Total writers processed:", len(writer_embeddings))
            
        
        final_writer_embeds = {}
        for writer_id, emb_list in writer_embeddings.items():
            if isinstance(emb_list, list) and len(emb_list) > 1:
                emb_stack = torch.stack(emb_list)
                avg_embed = emb_stack.mean(dim=0)
            else:
                avg_embed = emb_list[0] if isinstance(emb_list, list) else emb_list
            final_writer_embeds[writer_id] = avg_embed

        print("\n\t 2.unique writers:", sorted(unqWriterDict.keys()))  
        torch.save(final_writer_embeds, "./writerEmbeding/writer_style_refs2.pth")
        print("Saved writer embeddings to writer_style_refs.pth tot writers:", len(final_writer_embeds.keys()))

        loaded_embeddings = torch.load("./writerEmbeding/writer_style_refs2.pth", map_location='cpu')
        print("Total writers in loaded embeddings:", len(loaded_embeddings.keys()))

if __name__ == "__main__":
    main()