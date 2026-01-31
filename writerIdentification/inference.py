"""
    This code is for inference using the model from:
    Sheng He and Lambert Schomaker
    GR-RNN: Global-Context Residual Recurrent Neural Networks for Writer Identification
    Pattern Recognition
    
    @email: heshengxgd@gmail.com
    @author: Sheng He
    @Github: https://github.com/shengfly/writer-identification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataloaderICDAR1 as dset
import GRRNNModifiedWord as net
import os

class Diffusion:
    def __init__(self, noise_steps=512, beta_start=1e-4, beta_end=0.02, img_size=(64, 128), device=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

class DeepWriter_Inference:
    def __init__(self, dataset='CERUG-EN', imgtype='png', mode='horizontal'):
        self.dataset = dataset
        self.folder = dataset
        self.labelfolder = self.folder
        self.test_folder = self.folder + '/test/'
        self.imgtype = imgtype
        self.mode = mode
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.scale_size = (64, 128)


        self.model_dir = 'model'
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.testFile = "./data/styCrossAttention.csv"
        self.labelfolderTest = "/cluster/datastore/aniketag/allData/syntheticData/train/icdar2025/IAM/styleDiff/styCrossAttention_backUp/"

        test_set = dset.DatasetFromFolder2(self.testFile, self.labelfolderTest, 0)
        self.testing_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=2, shuffle=True)

        self.model = net.VGGnet1(1, num_classes=672).to(self.device)
        model_path = "/cluster/datastore/aniketag/newWordStylist/DiffusionPen/writer-identification/weights/WordStylistoneDmSplitNoise.pt"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Loaded model from: {model_path}")

    def accuracy(self, output, target, topk=(1,)):
        with torch.no_grad():
            maxk = 1
            confidences = torch.max(torch.softmax(output, dim=1), dim=1)[0]
            _, pred = torch.softmax(output, dim=1).topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.data.cpu().numpy())
        return res, confidences

    def test(self):
        self.model.eval()
        top1 = 0
        ntotal = 0
        diffusion = Diffusion(img_size=(64, 128), device=self.device)

        for iteration, batch in enumerate(self.testing_data_loader, 1):
            anchor_image, anchor_writer_id, anchor_image_name, positive_image, negative_image = batch
            inputs = anchor_image.to(self.device).float()

            inputs = inputs.squeeze(1)

            print("\n\t 0.inputs = ", inputs.shape)

            target = anchor_writer_id.to(self.device).long()
            t = diffusion.sample_timesteps(2).to(self.device)
            noisy_images, _ = diffusion.noise_images(inputs, t)
            noisy_images = noisy_images.squeeze(1)
            print("\n\t 1.noisy_images = ", noisy_images.shape)
            logits, _ = self.model(noisy_images)
            res, confidences = self.accuracy(logits, target, topk=(1,))
            top1 += res[0]
            ntotal += inputs.size(0)

        top1 /= float(ntotal)
        #print(f'Testing accuracy: top1: {top1*100:.2f}%')
        print(f'Testing accuracy: top1: {top1.item()*100:.2f}%')

if __name__ == '__main__':
    mod = DeepWriter_Inference(dataset='CERUG-EN', mode='horizontal')
    mod.test()