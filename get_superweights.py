
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import json
from diffusers import AutoencoderKL, DDIMScheduler
from unetSuperWeights import UNetModel
# from feature_extractor import ImageEncoder
# from utils.CVL_dataset import CVLDataset_style
# from utils.auxilary_functions import *
from transformers import CanineModel, CanineTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/ubuntu/DiffusionPen/models/ema_ckpt_CvlClassification.pt')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=15.0)
    parser.add_argument('--img_size', type=int, default=(64, 256))
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--model_name', type=str, default='diffusionpen')
    args = parser.parse_args()

    # Mock args for UNetModel initialization
    class MockArgs:
        def __init__(self, args):
            self.interpolation = False
            self.mix_rate = 0.5
            self.device = args.device
            self.model_name = args.model_name
            self.latent = args.latent
            self.img_size = args.img_size
            self.channels = args.channels
            self.emb_dim = args.emb_dim
            self.num_heads = args.num_heads
            self.num_res_blocks = args.num_res_blocks
    
    mock_args = MockArgs(args)

    # Initialize components
    print('Loading canine tokenizer and text encoder...')
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    text_encoder = CanineModel.from_pretrained("google/canine-c").to(args.device)
    
    # Initialize UNetModel
    print('Initializing UNetModel...')
    model = UNetModel(
        image_size=args.img_size,
        in_channels=args.channels,
        model_channels=args.emb_dim,
        out_channels=args.channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=args.num_heads,
        num_classes=283, # style_classes for CVL
        context_dim=args.emb_dim,
        vocab_size=69, # character_classes size
        text_encoder=text_encoder,
        args=mock_args
    ).to(args.device)

    # Load weights
    if os.path.exists(args.model_path):
        try:
            checkpoint = torch.load(args.model_path, map_location=args.device)
            model.load_state_dict(checkpoint)
            print(f'Loaded model from: {args.model_path}')
        except Exception as e:
            print(f'Error loading checkpoint: {e}')
    else:
        print(f'Warning: Model path {args.model_path} not found. Using random weights for demonstration.')

    model.eval()

    # Prepare a dummy batch
    print('Preparing dummy batch...')
    # If latent is True, input to UNet is [B, 4, 8, 32] for [64, 256] image
    x = torch.randn(args.batch_size, 4, args.img_size[0] // 8, args.img_size[1] // 8).to(args.device)
    timesteps = torch.tensor([500] * args.batch_size).to(args.device)
    
    # Dummy context (text features)
    text = ["Hello world"] * args.batch_size
    context = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt", max_length=200).to(args.device)
    
    y = torch.zeros(args.batch_size, dtype=torch.long).to(args.device) # Dummy labels

    print("Identifying super weights...")
    super_weights = model.identify_super_weights(x, timesteps, context, y, threshold=args.threshold)

    if not super_weights:
        print("No super weights identified with the current threshold.")
    else:
        print(f"Found {len(super_weights)} super weights:")
        for sw in super_weights:
            print(json.dumps(sw, indent=2))
        
        # Save to file
        with open('super_weights.json', 'w') as f:
            json.dump(super_weights, f, indent=2)
        print("Super weights saved to super_weights.json")

if __name__ == "__main__":
    main()
