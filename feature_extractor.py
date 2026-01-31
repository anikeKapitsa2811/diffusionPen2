import torch.nn as nn
import torch
import os

import os

# Your cache root (typo kept as-is; feel free to fix to "pytorchWeights")
CACHE_ROOT = "/cluster/datastore/aniketag/allData/pytrochWeights/"
os.makedirs(CACHE_ROOT, exist_ok=True)

# Matplotlib cache (optional)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(CACHE_ROOT, ".mpl-cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# Torch cache (torch.hub / torchvision)
os.environ["TORCH_HOME"] = CACHE_ROOT
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)

# XDG (some libs fall back here)
os.environ["XDG_CACHE_HOME"] = os.path.join(CACHE_ROOT, ".xdg-cache")
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

# >>> CRITICAL for timm / huggingface_hub <<<
os.environ["HF_HOME"] = CACHE_ROOT
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_ROOT
os.environ["HF_HUB_CACHE"] = CACHE_ROOT
# (optional, if you use transformers anywhere)
os.environ["TRANSFORMERS_CACHE"] = CACHE_ROOT

# As a last resort (if some code still expands '~' to /home/aniket), you can also:
# os.environ["HOME"] = CACHE_ROOT  # uncomment only if needed

# ---- now import libraries that trigger downloads ----
import timm
import torch


class ImageEncoderOriginal(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet50', num_classes=0, pretrained=True, trainable=True
    ):
        super().__init__()
        # checkpoint_path='/home/hankyul/.cache/torch/hub/checkpoints/resnet50_a1_0-14fe96d1.pth'
        
        print("\n\t model name:",model_name)
        
        """
        self.model = timm.create_model(
            model_name, pretrained, num_classes=num_classes, global_pool="max"
        )
        """
        #self.model = timm.create_model(model_name, pretrained, num_classes=num_classes, global_pool="max")

        #self.model = torch.compile(self.model, backend="inductor")
        
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            global_pool="max"
        )
                
        """
        self.model = timm.create_model(
            model_name, pretrained=False, num_classes=num_classes, global_pool="max"
        )
        """
        """
        self.model = timm.create_model(
            model_name, pretrained, num_classes=num_classes, global_pool="max"
        )
        """
        """
        # Define the local checkpoint path
        checkpoint_path = "/cluster/datastore/aniketag/allData/wordStylist/writerStyle/diffPenWeights/iam_style_diffusionpen_triplet.pth"

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda:0"))  # Adjust for GPU if needed

        # Load the weights into the model
        self.model.load_state_dict(checkpoint)
                
        print("Model loaded from checkpoint:",checkpoint_path)
        """
            
        for p in self.model.parameters():
            p.requires_grad = trainable
    def forward(self, x):
        x = self.model(x)
        
        #print("\n\t x.shape:",x.shape)
        return x   
    
    
# feature_extractor.py
import torch
import torch.nn as nn
import timm

class ImageEncoder(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True, trainable=True, emb_dim=256):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)  # no cls head
        if not trainable:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # feature dim from backbone global pooling
        if hasattr(self.backbone, 'num_features'):
            feat_dim = self.backbone.num_features
        else:
            # fallback
            feat_dim = self.backbone.get_classifier().in_features

        # heads
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.embed_head = nn.Identity()  # or nn.Linear(feat_dim, emb_dim) if you want a smaller embedding
        # if you want projection: self.embed_head = nn.Linear(feat_dim, emb_dim)

    def forward(self, x):
        feats_map = self.backbone.forward_features(x)              # [B, C, H, W]
        pooled = self.backbone.global_pool(feats_map)              # [B, C, 1, 1] or [B, C]
        if pooled.dim() == 4:
            pooled = pooled.flatten(1)                             # [B, C]
        features = self.embed_head(pooled)                         # [B, D]
        logits = self.classifier(pooled)                           # [B, num_classes]
        return logits, features


if __name__ == "__main__":
    
    import torch
    from feature_extractor import ImageEncoder   # assuming your file is named feature_extractor.py

    # Dummy config
    model_name = "mobilenetv2_100"
    num_classes = 283       # example number of writer IDs or style classes
    batch_size = 4
    emb_dim = 256           # optional, if you use embedding projection

    # Create the model
    model = ImageEncoder(model_name=model_name,
                        num_classes=num_classes,
                        pretrained=False,   # set to True if you want pretrained weights
                        trainable=True,
                        emb_dim=emb_dim)

    # Print model summary info
    print(f"✅ Model created: {model_name}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Dummy input image batch [B, 3, H, W]
    x = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    logits, features = model(x)

    # Print output shapes
    print("logits.shape:", logits.shape)    # should be [B, num_classes]
    print("features.shape:", features.shape)  # should be [B, feature_dim or emb_dim]
