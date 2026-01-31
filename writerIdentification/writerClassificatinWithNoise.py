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
from base import extract_features
import logging
# Specify the log file path
log_file_path = '../logs/test.log'

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
        self.savePath = "/cluster/datastore/aniketag/writerClassification/writer-identification/trainData/"
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
        self.batch_size = 512

        """
        1. specify tran data here
        """
        self.labelfolderTrain = "/cluster/datastore/aniketag/allData/wordStylist/allCrops_preprocess/"
        
        #self.labelfolderTest1 = self.labelfolderTrain #"/cluster/datastore/aniketag/allData/syntheticData/train/ckpt_Word_10head_7res_1/"
        #self.labelfolderTest2 = "/cluster/datastore/aniketag/allData/syntheticData/train/ckpt_Word_10head_7res_1/"
        
        """
            1st generated synthetic data
            2. specify synthetic data here
        """

        #labelfolderTrain = "/cluster/datastore/aniketag/allData/syntheticData/train/styCrossAttention/"
                
        #self.labelfolderTest2= "/cluster/datastore/aniketag/allData/syntheticData/train/ckpt_Word_10head_7res_1/" 
        
        #self.labelfolderTest2= "/cluster/datastore/aniketag/allData/syntheticData/train/icdar2025/IAM/styCrossAttentionTrain10Head/" 
                
        #self.labelfolderTest2="/cluster/datastore/aniketag/allData/syntheticData/train/icdar2025/IAM/variableDataGeneration/oneDmSynthetic//"
        self.labelfolderTest2= "/cluster/datastore/aniketag/allData/syntheticData/train//icdar2025/IAM/styleDiff/styCrossAttention_backUp/"
        
        #"/cluster/datastore/aniketag/allData/syntheticData/train/Mse_text_Phos_condi_FromScratch_FullSampling_sameWriter/"
    
        
        """ 
            2. 2nd generated synthetic data
        """
        #self.labelfolderTest2 = "/cluster/datastore/aniketag/allData/syntheticData/train/icpr//Mse_text_Phos_condi_FromScratchSameWriter/"  #"/cluster/datastore/aniketag/allData/syntheticData/train/wordStylistAuthorModel/"
        #self.testFile = "./data/gan_iam_syntheticTest_va_gt2.csv" #"./data/trainWrtClsForEmbd.csv" #"./data/gan_iam_tr_va_gt.csv" #"/cluster/datastore/aniketag/writerClassification/writer-identification/data1/gt_Mse_text_Phos_condi_FromScratchSameWriter.csv" #  

        # Mse_text_Phos_condi_FromScratch_FullSampling_sameWriter/        
        #self.TrainFile =  "./data/trainWrtClsForEmbd.csv" #"./data/gan_iam_tr_va_gt.csv"

        self.TrainFile =  "./data/wordStylist.csv" #"./data/gan_iam_tr_va_gt.csv"

        #self.testFile = "/cluster/datastore/aniketag/writerClassification/writer-identification/data/oneDmGenData.csv" #"./data/oneDmGenData.csv"
        self.testFile = "./data/styCrossAttention.csv"

        print("\n\t labelfolder:",self.labelfolder)
        print("\n\t self.TrainFile =",self.TrainFile)
        print("\n\t self.testFile =",self.testFile)
                
        if 0:
            train_set = dset.DatasetFromFolder2(dataset=self.dataset,
                            labelfolder = self.labelfolder,
                            foldername=self.train_folder,
                            imgtype=self.imgtype,
                            scale_size=self.scale_size,
                            is_training = True)
            
        else:
            train_set = dset.DatasetFromFolder2(self.TrainFile,self.labelfolderTrain,0)

        
        self.training_data_loader = DataLoader(dataset=train_set, num_workers=4,
                                               batch_size=512, shuffle=True)
        
        print("\n\t train folder length:",len(self.training_data_loader))
        
        
        if 0:
            test_set = dset.DatasetFromFolder2(dataset=self.dataset,
                            labelfolder = self.labelfolder,
                            foldername=self.test_folder,imgtype=self.imgtype,
                            scale_size=self.scale_size,
                            is_training = False)
        else:
            
            print("\n\t test data from:",self.labelfolderTest2," \t tot examples:",len(os.listdir(self.labelfolderTest2)))
            
            test_set1 = dset.DatasetFromFolder2(self.testFile,self.labelfolderTest2,0)
            
            
            
            #test_set2 = dset.DatasetFromFolder2(self.testFile,self.labelfolderTest2,0)



        self.testing_data_loader1 = DataLoader(dataset=test_set1, num_workers=4, 
                           batch_size=1, shuffle=True)
        """
        self.testing_data_loader2 = DataLoader(dataset=test_set2, num_workers=4, 
                           batch_size=self.batch_size, shuffle=False)
        """
        print("\n\t test1:",len(self.testing_data_loader1))#," test2:",len(self.testing_data_loader2))
        
        num_class = 672 #train_set.num_writer
        #self.model = net.GrnnNet1(1,num_classes=num_class,mode=self.mode).to(self.device)
        

        #self.model = net.VGGnet1(1,num_class+1).to(self.device)

        #self.model = net.VGGnet3(1,num_class+1).to(self.device)

        
        #self.model = net.VGGnet3(model_name='resnet50', num_classes = 672,pretrained=True, trainable=True).to(self.device)

        self.model = net.VGGnet1(1,num_classes=672).to(self.device)

        """
        modelName = "./styleModel/resnetNoNoise.pt"

        modelName = "./styleModel/resnetSmallNoise3.pt"

        if 0:
            print("\n\t pred fraom modelName=",modelName)
            
            self.model.load_state_dict(torch.load(modelName))
        """
        #self.model = torch.load("/cluster/datastore/aniketag/writerClassification/writer-identification/model/wordStylist.pt")
        #self.model.load_state_dict(torch.load("/cluster/datastore/aniketag/writerClassification/writer-identification/styleModel/modelWithNoise.pt"))
        #self.model.load_state_dict(torch.load("/cluster/datastore/aniketag/writerClassification/writer-identification/styleModel/modelWithNoise.pt")) MobileNetV2Classifier.pt
        
        if 0:
            modelPath = "./styleModel/modelWithNoise2.pt" # this is mobilenet
            
            #modelPath = "./styleModel/MobileNetV2Classifier.pt"
            #modelPath = "./styleModel/mobilenet_v2Triplet.pt"
            self.model.load_state_dict(torch.load(modelPath)) 
            print("\n\t old model loaded:",modelPath)
        
        if 0:
            modelPath = "/cluster/datastore/aniketag/writerClassification/writer-identification/modelOldBackUp/wordStylist2.pt"
            print("\n\t modelPath =",modelPath)
            self.model.load_state_dict(torch.load(modelPath))
        

        #self.criterion = nn.CrossEntropyLoss()
                
        self.criterion = LabelSomCE()
        
        print("\n\t self.criterion:",self.criterion)
        
        self.optimizer = optim.Adam(self.model.parameters(),lr=0.0001,weight_decay=1e-4) 
        self.scheduler = lr_scheduler.StepLR(self.optimizer,step_size=10,gamma=0.5)
                
    def save_tensor_as_images(self, tensor, folder_path):
        for i in range(tensor.shape[0]):
            img = tensor[i]
            print("\nTensor shape at iteration", i, ":", img.shape)

            # Assuming your tensor is named 'tensor'
            min_value = torch.min(tensor)
            max_value = torch.max(tensor)

            print("Minimum value:", min_value.item())
            print("Maximum value:", max_value.item())

            img = img.permute(1, 2, 0)
            print("After permutation, img shape:", img.shape)

            img = img.cpu().detach().numpy()
            print("After converting to NumPy array, img shape:", img.shape)
            print("Data type of img:", img.dtype)

            img = np.squeeze(img)


            img = (img * 255).astype(np.uint8)
            print("After scaling and converting data type, img shape:", img.shape)
            print("Data type of img:", img.dtype)
            #img = np.dstack((img,img,img)) 

            img = Image.fromarray(img)
            print("After creating PIL image, img mode:", img.mode)

            img_name = f'image_{i}.png'
            img.save(f'{folder_path}/{img_name}')                
     

    import torchvision
    from torchvision.utils import save_image
    import os

    # Create a function to save noisy images
    def save_noisy_images(self,noisy_images, epoch, iteration, output_dir="noisy_images"):
        """
        Save noisy images to disk.

        Args:
        - noisy_images (torch.Tensor): The batch of noisy images.
        - epoch (int): Current epoch number.
        - iteration (int): Current iteration number.
        - output_dir (str): Directory where images will be saved.
        """

        # Ensure the directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("save_image:",save_image)
        # Save noisy images (convert to range [0, 1] if needed)
        save_image(noisy_images, f'{output_dir}/noisy_images_epoch{epoch}_batch{iteration}.png')

        print(f"Saved noisy images to {output_dir}/noisy_images_epoch{epoch}_batch{iteration}.png")

   
                
    def train(self,epoch):
        
        self.model.train()
        losstotal = []
        triplet_loss_fn = TripletLoss(margin=1.0)  # Instantiate Triplet Loss
        diffusion = Diffusion(img_size=(64, 128), device=self.device)

        
        for iteration,batch in enumerate(self.training_data_loader,1):
            
            
            with open("./flagOCR.txt","r") as f:
                flagValue = int(f.read())
            
            #print("\n\t flag:",f)
            
            if flagValue == 0:

                print("exit:",flagValue)
                exit()
                break
            else:
                pass

            """                    
            batch = list(batch)
            inputs = batch[0].to(self.device).float()
            target = batch[1].type(torch.long).to(self.device)
            """
            anchor_image,anchor_writer_id,anchor_image_name, positive_image, negative_image = batch
            
            #print("\n\t image_name:",image_name)
            
            #print("\n\t len:",len(batch))

            inputs = anchor_image.to(self.device).float() #list(batch[0])
            target = anchor_writer_id.to(self.device).long() #list(batch[1])
            
            #t = diffusion.sample_timesteps(10).to(self.device)

            print("\n\t 0.inputs.shape[0]:",inputs.shape[0])
            
            t = diffusion.sample_timesteps(inputs.shape[0]).to(self.device)

            #t = diffusion.sample_timesteps(10).to(self.device)

            inputs = inputs.squeeze(1)

            # Add noise to the input images
            noisy_images, noise = diffusion.noise_images(inputs, t)

            #noisy_images = inputs
            #self.save_noisy_images(noisy_images, epoch, iteration)
            #self.save_tensor_as_images(noisy_images,"./noisy_images/")
            #print("\n\t noisy_images.shape:",noisy_images.shape)
        	#noisy_images.shape: torch.Size([256, 3, 64, 256])

            noisy_images = noisy_images.squeeze(1)
            
            if 0:
                positive_image = positive_image.to(self.device).float() #list(batch[0])
                positive_image = positive_image.squeeze(1)
                
                negative_image = negative_image.to(self.device).float() #list(batch[0])
                negative_image = negative_image.squeeze(1)
            
            #print("\n\t inputs.shape:",inputs.shape," target.shape:",target.shape)
            #print("\n\t target.shape:",target)
            #self.save_tensor_as_images(inputs, self.savePath)
            #inputs = inputs.squeeze()
            #inputs, target = extract_features(inputs, target, "LINES_METHOD")
            #print("\n\t 2.inputs.shape:",inputs.shape," target len:",len(target))
            #print("\n\t 2.dummyTarget:",dummyTarget)            
            
            self.optimizer.zero_grad()
            
            # Use noisy images for training
            
            print("\n\t 1.noisy_images.shape:",noisy_images.shape," target.shape:",target.shape)
            logits, anchor_output = self.model(noisy_images)
            #print("\n\t logits.shape:",logits.shape)
             
            #logits,anchor_output = self.model(inputs)
            #_,positive_output = self.model(positive_image)
            #_,negative_output = self.model(negative_image)
            

            #print("\n\t logits.len:",logits.size,"\t intermediate_output.shape:",intermediate_output.size," target.shape:",target.shape)

            #print("\n\t logits.shape:",logits.shape,"\t intermediate_output.shape:",intermediate_output.shape," target.shape:",target.shape)
            
            train_loss= self.criterion(logits,target) 
            #tripletLoss = triplet_loss_fn(logits, positiveLogits, negativeLogits)
            #tripletLoss = triplet_loss_fn(anchor_output, positive_output, negative_output)
            #train_loss= train_loss+tripletLoss
            
            #print("\n\t epoch:",epoch," train_loss=",train_loss.item()," tripletLoss=",tripletLoss.item())
            losstotal.append(train_loss.item())
            
            #print("\n\t loss:",train_loss.item()," \t iteration:",iteration)
            train_loss.backward()
            self.optimizer.step()
        
        with open(self.logfile,'a') as fp:
            fp.write('Training epoch %d avg loss is: %.6f\n'%(epoch,np.mean(losstotal)))
        print('Traing epoch:',epoch,'  avg loss is:',np.mean(losstotal))

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
            
            #print("\n\t target =",batch[1])

            #batch = list(batch)
            
            #image_name,writer_id,anchor_image, positive_image, negative_image = batch
            
            anchor_image,anchor_writer_id,anchor_image_name, positive_image, negative_image = batch
            
            #print("\n\t image_name:",image_name)
            
            #print("\n\t len:",len(batch))

            inputs = anchor_image.to(self.device).float() #list(batch[0])
            t = diffusion.sample_timesteps(inputs.shape[0]).to(self.device)
            
            if 1:
                noisy_images, noise = diffusion.noise_images(inputs, t)
                
                noisy_images = inputs.squeeze(1)
            
            target = anchor_writer_id.to(self.device).long() #list(batch[1])
            inputs = inputs.squeeze(1)
            #print("\n\t 0.inputs.shape =",inputs.shape)

            #inputs = inputs.repeat(1, 3, 1, 1)  # Repeat the 1 channel to create 3 channels

            #print("\n\t 1.inputs.shape =",inputs.shape)

            #inputs = batch[0].to(self.device).float()
            #target = batch[1].to(self.device).long()
           
           
            #logits,intermediate_output = self.model(inputs)
            #print("\n\t 2.noisy_images.shape:",noisy_images.shape," target.shape:",target.shape)
            logits, anchor_output = self.model(noisy_images)
            
            res,confidences = self.accuracy(logits,target,topk=(1))
            top1 += res[0]
            #top5 += res[1]
            
            ntotal += inputs.size(0)
        

        top1 /= float(ntotal)
        top5 /= float(ntotal)
    
        print('Testing on epoch: %d has accuracy: top1: %.2f top5: %.2f'%(epoch,top1*100,top5*100))
        """
        with open(self.logfile,'a') as fp:
            fp.write('Testing epoch %d accuracy is: top1: %.2f top5: %.2f\n'%(epoch,top1*100,top5*100))
        """


    def accuracy(self,output,target,topk=(1,)):
        with torch.no_grad():
            maxk = 1 #max(topk)
            
            confidences = torch.max(torch.softmax(output,dim=1),dim=1)[0]

            _,pred = torch.softmax(output,dim=1).topk(maxk,1,True,True)

            
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
    
    def checkpoint(self,epoch):
        #model_out_path = self.model_dir + '/' + self.modelfile + '-model_epoch_{}.pth'.format(epoch)

        #model_out_path = self.model_dir + '/' + self.modelfile + '-model_epoch.pth'.format(epoch)
        #model_out_path = self.model_dir + '/' +"largeWriterWriterClassificationTrainHiGanPlus_282.pt"
        #model_out_path = self.model_dir + '/' +"largeWriterWriterClassificationTrainDiffusion_282.pt"

        #model_out_path = self.model_dir + '/' +"largeWriterWriterClassificationTrainOriginalData_pretrainedResnet.pt"
        #model_out_path = self.model_dir + '/' +"wordStylistNewTrainTestSplit.pt"
        
        #model_out_path = self.model_dir + '/' +"Mse_text_Phos_condi_FromScratchSameWriter.pt"
        
        #model_out_path = "/cluster/datastore/aniketag/writerClassification/writer-identification/styleModel/resnetNoNoise.pt"

        #model_out_path = "/cluster/datastore/aniketag/writerClassification/writer-identification/styleModel/WordStylistoneDmSplit.pt"

        model_out_path = "/cluster/datastore/aniketag/writerClassification/writer-identification/styleModel/WordStylistoneDmSplitNoise.pt"

        
        torch.save(self.model.state_dict(),model_out_path)
    
    
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
            
            
        # /cluster/datastore/aniketag/allData/syntheticData/train/ckpt_Word_10head_7res
        for epoch in range(start_epoch,num_epoch):


            
            if epoch%10==0:
                print("\n\t diffusion test Accuracy")
                self.test(epoch,self.testing_data_loader1)
                
            #self.test(epoch,self.testing_data_loader1)
            
             
            self.train(epoch)
            
            if 1:#epoch%5==0 and epoch>0:
                self.checkpoint(epoch)
            
            #print("\n\t HWT test accuracy")
            #print("\n\t HiGan test accuracy")
            # self.test(epoch,self.testing_data_loader2)
            #input("check!!!")

            self.scheduler.step()
                                        
                
if __name__ == '__main__':
    
	
    modelist = ['vertical','horzontal']
    mode = modelist[1]
    print("1.")
    mod = DeepWriter_Train(dataset='CERUG-EN',mode=mode)
    mod.train_loops(0,2000)




					




