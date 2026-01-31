"""
    This code is for the following paper:
    
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
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import pickle
import dataloaderICDAR1 as dset
import torch.nn.functional as F

#import GRRNN as net
import GRRNNModifiedWord as net 
import numpy as np
import os
import torch
from PIL import Image
import logging
# Specify the log file path
log_file_path = './writerLogs/delMe.txt'

# Configure logging to log only to a file
logging.basicConfig(
    format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_file_path, mode='a')  # Log only to file in append mode
    ]
)

# Get the logger
logger = logging.getLogger('Writer classification Experiment::train')

# Example logging
logger.info("Training started.")
logger.error("An error occurred.")

logger.info('--- Running Writer Training ---')


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


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.mean(F.relu(pos_dist - neg_dist + self.margin))
        return loss


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=(64, 128), device=None):
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

    def noise_images1(self, x, t):
        batch_size = x.shape[0]  # Get batch size (256 here)

        # Treat self.alpha_hat[t] as a scalar and expand to (batch_size, 1, 1, 1)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(1, 1, 1, 1).expand(batch_size, -1, -1, -1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).view(1, 1, 1, 1).expand(batch_size, -1, -1, -1)

        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon



    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

  
class DeepWriter_Train:
    def __init__(self,dataset='CERUG-EN',imgtype='png',mode='vertical'):
    
        self.dataset = dataset
        self.folder = dataset
        #self.labelfolder = 'dataset/'
        
        self.writerEmbeddings = dict()
                
        self.labelfolder = self.folder
        self.train_folder = self.folder+'/train/'
        self.test_folder = self.folder+'/test/'
        
        self.imgtype=imgtype
        self.mode = mode
        self.device = 'cuda:0'
        self.scale_size=(64,128)
        
        if self.device == 'cuda:0':
            torch.backends.cudnn.benchmark = True
        
        if self.dataset == 'CVL':
            self.imgtype = 'tif'
        
        self.model_dir = 'model'
        if not os.path.exists(self.model_dir):
            #raise ValueError('Model directory: %s does not existed'%self.model_dir)
            os.mkdir(self.model_dir)#raise ValueError('Model directory: %s does not existed'%self.model_dir)
        
        basedir = 'GRRNN_WriterIdentification_dataset_'+self.dataset+'_model_'+self.mode+'_aug_16'
        self.logfile=   './allLogs/Mse_text_Phos_condi_FromScratchSameWriter.log'
        self.modelfile = basedir

        """
        1. specify tran data here
        """
        self.labelfolderTrain = "/cluster/datastore/aniketag/allData/wordStylist/allCrops_preprocess/"
        
        
        """
            1st generated synthetic data
            2. specify synthetic data here
        """

        self.labelfolderTest2= "/cluster/datastore/aniketag/allData/syntheticData/train//icdar2025/IAM/styleDiff/styCrossAttention_backUp/"
        
    
        
        """ 
            2. 2nd generated synthetic data
        """

        self.TrainFile =  "./data/wordStylist.csv" #"./data/gan_iam_tr_va_gt.csv"

        self.testFile = "./data/styCrossAttention.csv"

        print("\n\t labelfolder:",self.labelfolder)
        print("\n\t self.TrainFile =",self.TrainFile)
        print("\n\t self.testFile =",self.testFile)
                
        
        if 0:
            test_set = dset.DatasetFromFolder2(dataset=self.dataset,
                            labelfolder = self.labelfolder,
                            foldername=self.test_folder,imgtype=self.imgtype,
                            scale_size=self.scale_size,
                            is_training = False)
        else:
            
            print("\n\t test data from:",self.labelfolderTest2," \t tot examples:",len(os.listdir(self.labelfolderTest2)))
            
            test_set1 = dset.DatasetFromFolder2(self.testFile,self.labelfolderTest2,0)
            
    

        self.testing_data_loader1 = DataLoader(dataset=test_set1, num_workers=4, 
                           batch_size=256, shuffle=True)
        print("\n\t test1:",len(self.testing_data_loader1))#," test2:",len(self.testing_data_loader2))
        

        self.model = net.VGGnet1(1,num_classes=672).to(self.device)

        
        try:
            self.model.load_state_dict(torch.load("./styleModel/WordStylistoneDmSplitNoise.pt")) #MobileNetV2Classifier.pt
            print("\n\t model loaded from styleModel/WordStylistoneDmSplitNoise.pt")
        except Exception as e:
            print("\n\t Exception in loading model:",e)
            print("\n\t model not loaded, check the path")
            exit()
                
                
    def test(self,epoch,dataLoader,during_train=True):
        self.model.eval()
        
        with open("./flagOCR.txt","r") as f:
            flagValue = int(f.read())
        
        #print("\n\t flag:",f)
        
        if flagValue == 0:

            print("exit:",flagValue)
            exit()
        else:
            pass

    
        if not during_train:
            self.load_model(epoch)

        top1 = 0
        top5 = 0
        ntotal=0
        diffusion = Diffusion(img_size=(64, 128), device=self.device)

        for iteration,batch in enumerate(dataLoader,1): # self.testing_data_loader
            

            with open("./flagOCR.txt","r") as f:
                flagValue = int(f.read())
            
            #print("\n\t flag:",f)
            
            if flagValue == 0:

                print("exit:",flagValue)
                exit()
            else:
                pass

            
            anchor_image,anchor_writer_id,anchor_image_name, positive_image, negative_image = batch
            
            #print("\n\t image_name:",image_name)
            
            #print("\n\t len:",len(batch))

            inputs = anchor_image.to(self.device).float() #list(batch[0])
            t = diffusion.sample_timesteps(epoch*10).to(self.device)
            
            noisy_images, noise = diffusion.noise_images(inputs, t)
            
            noisy_images = inputs.squeeze(1)
            
            target = anchor_writer_id.to(self.device).long() #list(batch[1])
            inputs = inputs.squeeze(1)
            
            logits, anchor_output = self.model(noisy_images)
            
            #print("\n\t logits.shape:",logits.shape," target.shape:",target.shape," noisy_images.shape:",noisy_images.shape)
            res,confidences = self.accuracy(logits,target,topk=(1))
            top1 += res[0]
            #top5 += res[1]
            
            ntotal += inputs.size(0)
        

        top1 /= float(ntotal)
        top5 /= float(ntotal)
    
        print('Testing on epoch: %d has accuracy: top1: %.2f top5: %s'%(epoch,top1*100,logits.shape))


    def accuracy(self,output,target,topk=(1,)):
        with torch.no_grad():
            maxk = 1 #max(topk)
            
            confidences = torch.max(torch.softmax(output,dim=1),dim=1)[0]

            _,pred = torch.softmax(output,dim=1).topk(maxk,1,True,True)

            #print("\n\t 1.pred =",pred," target.shape:",target.shape," pred.shape:",pred.shape)            
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            #print("\n\t 1.correct =",correct," pred.shape:",pred.shape)
            res = []
            
            """
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.data.cpu().numpy())
            """
            k=1
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            #print("\n\t 1.correct_k =",correct_k," pred:",pred.shape)

            #input("check!!!")


            res.append(correct_k.data.cpu().numpy())

            
            
        return res,confidences


    def check_exists(self,epoch):
        model_out_path = self.model_dir + '/' + self.modelfile + '-model_epoch_{}.pth'.format(epoch)
        return os.path.exists(model_out_path)
    
    
    
    def load_model(self,epoch):
        #model_out_path = self.model_dir + '/' + self.modelfile + '-model_epoch_{}.pth'.format(epoch)
        
        model_out_path = "/cluster/datastore/aniketag/writerClassification/writer-identification/styleModel/WordStylistoneDmSplitNoise.pt"

        self.model.load_state_dict(torch.load(model_out_path,map_location=self.device))
        
        print('Load model successful from :',model_out_path)
                
    def train_loops(self,start_epoch,num_epoch):
        #if self.check_exists(num_epoch): return
        if start_epoch > 0:
            self.load_model(start_epoch-1)
            print("pretrained model loaded")
            
            #input("model loaded!!!")
                        
        for epoch in range(start_epoch,num_epoch):
            
            self.test(epoch,self.testing_data_loader1)
            
                                                   
                
if __name__ == '__main__':
    
	
    modelist = ['vertical','horzontal']
    mode = modelist[1]
    print("1.")
    mod = DeepWriter_Train(dataset='CERUG-EN',mode=mode)
    mod.train_loops(0,60)




					




