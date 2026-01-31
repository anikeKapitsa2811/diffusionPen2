import torch
import numpy as np
from PIL import Image

# Path to your file
#path = "/cluster/datastore/aniketag/allData/diffPen//saved_iam_data/train_word_IAM.pt"
#path = "/cluster/datastore/aniketag/newHTR/icpr/HTR-best-practices/saved_datasets/train_word_IAM.pt"
path = "/cluster/datastore/aniketag/allData/diffPen//saved_GNHK_data//train_word_GNHK.pt"

# Load data
data = torch.load(path, map_location='cpu')

print(f"\nLoaded object type: {type(data)}")
print(f"Total elements: {len(data)}")

if isinstance(data, list) and len(data) > 0:
    first = data[-2]
    print(f"\nType of first element: {type(first)} (length={len(first)})")

    if isinstance(first, (list, tuple)):
        img_part = first[0]
        text_part = first[1]
        thrd_part = first[2] if len(first) > 2 else None
        fourth_part = first[3] if len(first) > 3 else None
        
        print("thrd_part :",thrd_part," fourth_part:",fourth_part)   
        print("\n--- Element 0 (image-related) ---")
        print(f"Type: {img_part.size}")
        print(f"Type: {type(img_part)}")

        if isinstance(img_part, np.ndarray):
            print(f"Shape: {img_part.shape}, dtype: {img_part.dtype}")
            print(f"Min: {img_part.min():.4f}, Max: {img_part.max():.4f}")
        elif isinstance(img_part, torch.Tensor):
            print(f"Shape: {tuple(img_part.shape)}, dtype: {img_part.dtype}")
            print(f"Min: {img_part.min().item():.4f}, Max: {img_part.max().item():.4f}")
        elif isinstance(img_part, str):
            print(f"Looks like a filename: {img_part}")
        else:
            print(f"Unknown image element type: {type(img_part)}")

        print("\n--- Element 1 (text/transcription) ---")
        print(f"Type: {type(text_part)}")
        if isinstance(text_part, str):
            print(f"Text: {text_part}")
        else:
            print("Not a string.")

        if isinstance(img_part, Image.Image):
            img_part = np.array(img_part)
    
            print("After conversion type:", type(img_part))
            print("Image shape:", img_part.shape)
            print("Image dtype:", img_part.dtype)
    
    
else:
    print("\nData format not recognized or list is empty.")
