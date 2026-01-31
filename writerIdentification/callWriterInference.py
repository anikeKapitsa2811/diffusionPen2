

import torch
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
# Add current file's directory to sys.path
sys.path.append(os.path.dirname(__file__))

import GRRNNModifiedWord as net1
import torch.nn.functional as F


class LabelSomCE(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self,x,target,smoothing=0.1):
		confidence = 1.0 - smoothing
		logprobs = F.log_softmax(x,dim=-1)
		nll_loss = - logprobs.gather(dim=-1,index=target.unsqueeze(1))
		nll_loss = nll_loss.squeeze(1)
		smooth_loss = -logprobs.mean(dim=-1)
		loss = confidence * nll_loss + smoothing * smooth_loss

		return loss.mean()

def convert_for_writer_model(tensor):
    if tensor.shape[1] == 3:
        tensor = 0.2989 * tensor[:,0:1,:,:] + 0.5870 * tensor[:,1:2,:,:] + 0.1140 * tensor[:,2:3,:,:]
    return tensor  # No inversion

def callWriter(writer_model, noisy_residual, device, img_size=(64, 128)):
    writer_model.eval().to(device)
    inputs = noisy_residual  # [B, 3, 64, 256]
    #inputs = convert_for_writer_model(inputs)  # [B, 1, 64, 256]
    
    #print(f"1.callWriter: inputs.shape: {inputs.shape}")
    #inputs = F.interpolate(inputs, size=img_size, mode='bilinear', align_corners=False)  # [B, 1, 64, 128]
    #print(f"2.callWriter: inputs.shape: {inputs.shape}")

    logits, _ = writer_model(inputs)
    confidences = torch.max(F.softmax(logits, dim=1), dim=1)[0]
    _, predictions = torch.max(logits, dim=1)
    return logits, predictions, confidences


def callWriterOriginal(writer_model, noisy_residual, device, img_size=(64, 128)):
    """
    Perform writer classification on noisy_residual tensor using a pre-trained model.
    
    Args:
        writer_model: Pre-trained writer classification model (VGGnet11).
        noisy_residual (torch.Tensor): Input tensor from diffusion model, shape [batch_size, channels, height, width].
        device (str): Device for computation (e.g., 'cuda:0' or 'cpu').
        img_size (tuple): Target image size for resizing, default (64, 128).
    
    Returns:
        tuple: (predictions, confidences)
            - predictions (torch.Tensor): Predicted writer IDs, shape [batch_size].
            - confidences (torch.Tensor): Softmax probabilities, shape [batch_size].
    """
    # Ensure model is in eval mode and on correct device
    writer_model.eval()
    writer_model = writer_model.to(device)
    
    # Preprocess noisy_residual
    inputs = noisy_residual#.clone().to(device).float()
    
    """
    # Handle latent mode (channels=4) or pixel mode (channels=3)
    if inputs.shape[1] == 4:  # Latent mode   		 post_process_latents: images.shape: torch.Size([8, 64, 256, 3])

        inputs = F.interpolate(inputs, size=img_size, mode='bilinear', align_corners=False)
        inputs = inputs.mean(dim=1, keepdim=True)  # Reduce to single channel
    elif inputs.shape[1] == 3:  # Pixel mode
        inputs = F.interpolate(inputs, size=img_size, mode='bilinear', align_corners=False)
        inputs = inputs[:, :1, :, :]  # Take first channel to match VGGnet1
    else:
        raise ValueError(f"Unsupported channel dimension: {inputs.shape[1]}")
    """
    #inputs = inputs.squeeze(1)  # Shape: [batch_size, height, width]
    
    inputs = inputs.permute(0,3,1,2)
    
    #print("\n\t callWriterOriginal: inputs.shape:", inputs.shape)
    # Perform classification
    #with torch.no_grad():
    if 1:
        logits, _ = writer_model(inputs)
        confidences = torch.max(F.softmax(logits, dim=1), dim=1)[0]
        _, predictions = torch.max(logits, dim=1)
    
    return logits,predictions, confidences

def calculate_writer_accuracy(predictions, real_labels,baseImageName,imageNameWriterDict,device,batchNo,time):
    """
    Calculate top-1 accuracy for writer predictions.
    
    Args:
        predictions (torch.Tensor): Predicted writer IDs, shape [batch_size].
        real_labels (torch.Tensor): Ground truth writer IDs, shape [batch_size].
    
    Returns:
        float: Top-1 accuracy as a percentage.
    """
    
    # read imageNames get its writer id replace real_label tensor with new tensor
    
    newLabels = []
    
    df = pd.read_csv("/cluster/datastore/aniketag/writerClassification/writer-identification/data/diffTrainTestVal.csv")
    for nm in baseImageName:
        
        try:
            t = int(imageNameWriterDict[nm])
        except Exception as e:
            # fraom dataframe 2nd column search image name and get writer id fraom 1st columns
            #t = df[df['imageName'] == nm].iloc[0, 1]  # Assuming 2nd column is writer ID
            # do it without column name imageName its 2nd columns and writer id is 1st
            t = int(df[df.iloc[:, 1] == nm].iloc[0, 0])
            print("t:",t)
            
        t = torch.tensor(t, device=device)
        if nm in imageNameWriterDict:
            newLabels.append(t)
        else:
            print(f"Warning: {nm} not found in imageNameWriterDict")
            newLabels.append(-1)    
    
    #newLabels = torch.tensor(newLabels, device=real_labels.device)
    
    #newLabels = torch.stack(newLabels)
    #newLabels = torch.stack([l if isinstance(l, torch.Tensor) else torch.tensor(l) for l in newLabels])
    device = predictions.device  # or manually: device = torch.device("cuda:1")
    newLabels = torch.stack([
        l.to(device) if isinstance(l, torch.Tensor) else torch.tensor(l, device=device)
        for l in newLabels
    ])

    #print("\n predictions:", predictions)
    #print("\n newLabels:", newLabels)
    #print("\n real_labels:", real_labels)
    #print("\n baseImageName",baseImageName)
    
    with torch.no_grad():
        #correct = predictions.eq(real_labels).float().sum()
        correct = predictions.eq(newLabels).float().sum()

        total = real_labels.size(0)
        accuracy = (correct / total) * 100
    return accuracy.item()#,correct, total

# Initialize writer model (call once outside the function if needed)
def init_writer_model( device='cuda:0', num_classes=672):
    """
    Initialize and load the writer classification model.
    
    Args:
        model_path (str): Path to preWriter model loaded from-trained model weights.
        device (str): Device for model (e.g., 'cuda:0' or 'cpu').
        num_classes (int): Number of writer classes (default 672).
    
    Returns:
        writer_model: Loaded VGGnet1 model.
    """
    
    #model_path= "/cluster/datastore/aniketag/newWordStylist/DiffusionPen/writerIdentification/styleModel/WordStylistoneDmSplitNoise.pt"
    
    #model_path = "/cluster/datastore/aniketag/writerClassification/writer-identification/styleModel/DiffPenSplitNoise.pt"
    model_path = "/cluster/datastore/aniketag/writerClassification/writer-identification/styleModel/DiffPenTrainTestValSplitNoise.pt"
    writer_model = net1.VGGnet1(1, num_classes=num_classes).to(device)

    try:
        writer_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f" Loading fraom path complete {model_path}")
    except Exception as e:
        print(f"Error loading writer model: {e}")
        raise
        exit()
    return writer_model

def convert_for_writer_model(tensor):
    """
    Convert RGB [0,1] tensor to grayscale-inverted format expected by the writer classifier.
    Assumes tensor shape (B, 3, H, W) and values in [0, 1]
    """
    if tensor.shape[1] == 3:
        tensor = 0.2989 * tensor[:,0:1,:,:] + 0.5870 * tensor[:,1:2,:,:] + 0.1140 * tensor[:,2:3,:,:]
    
    tensor = 1.0 - tensor  # Invert
    return tensor
