
"""

@author: sheng he 
@email: heshengxgd@gmail.com

This code is used for writer identification based on deep learning

"""

import os
import pickle
import numpy as np
#from scipy import misc

import skimage.transform as misc #import resize

import torch.utils.data as data
import torch
        
from torchvision.transforms import Lambda,Compose, ToTensor, RandomHorizontalFlip,RandomRotation
import random
import imageio
import pandas as pd
from PIL import Image
import cv2

import logging
# Specify the log file path
log_file_path = './writerLogs/test.log'

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

logger.info('--- Running writer Training ---')




class DatasetFromFolder(data.Dataset):
        def __init__(self,dataset,foldername,labelfolder,imgtype='png',scale_size=(64,128),
                     is_training=True):
                super(DatasetFromFolder,self).__init__()
                
                self.is_training = is_training
                
                self.imgtype = imgtype
                self.scale_size = scale_size
                self.folder = foldername
                self.dataset = dataset
                
                if self.dataset == 'CERUG-EN':
                    self.cerug = True
                else:
                    self.cerug = False
                
                self.labelidx_name = labelfolder + dataset + 'writer_index_table.pickle'
                print(self.labelidx_name)
                
                self.imglist = self._get_image_list(self.folder)
                
                self.idlist = self._get_all_identity()
                
                self.idx_tab = self._convert_identity2index(self.labelidx_name)
                
                self.num_writer = len(self.idx_tab)
                
                #------------ print info.
                print('-'*10)
                print('loading dataset %s with images: %d'%(dataset,len(self.imglist)))
                print('number of writer is: %d'%len(self.idx_tab))
                print('-*'*10)
                
                #self.trans = True
                
                
        
        # convert to idx for neural network
        def _convert_identity2index(self,savename):
                if os.path.exists(savename):
                        with open(savename,'rb') as fp:
                                identity_idx = pickle.load(fp)
                else:
                        #'''
                        identity_idx = {}
                        for idx,ids in enumerate(self.idlist):
                                identity_idx[ids] = idx
                        
                        with open(savename,'wb') as fp:
                                pickle.dump(identity_idx,fp)
                        #'''
                        
                return identity_idx
                                
        # get all writer identity
        def _get_all_identity(self):
                writer_list = []
                for img in self.imglist:
                        writerId = self._get_identity(img)
                        writer_list.append(writerId)
                writer_list=list(set(writer_list))
                return writer_list
        
        def _get_identity(self,fname):
                if self.cerug:
                        return fname.split('_')[0]
                else: return fname.split('-')[0]
        
        # get all image list 
        def _get_image_list(self,folder):
                flist = os.listdir(folder)
                imglist = []
                for img in flist:
                        if img.endswith(self.imgtype):
                                imglist.append(img)
                return imglist
        
        def transform(self):
                return Compose([ToTensor(),])
        
        def resize(self,image):
                h,w = image.shape[:2]
                ratio_h = float(self.scale_size[0])/float(h)
                ratio_w = float(self.scale_size[1])/float(w)
                
                if ratio_h < ratio_w:
                        ratio = ratio_h
                        hfirst = False
                else:
                        ratio = ratio_w
                        hfirst = True
                        
                nh = int(ratio * h)
                nw = int(ratio * w)
                
                #imre = misc.imresize(image,(nh,nw))
                imre = misc.resize(image,(nh,nw))

                
                imre = 255 - imre
                ch,cw = imre.shape[:2]
                if self.is_training:
                    new_img = np.zeros(self.scale_size)
                    dy = int((self.scale_size[0]-ch))
                    dx = int((self.scale_size[1]-cw))
                    dy = random.randint(0,dy)
                    dx = random.randint(0,dx)
                else:
                    new_img = np.zeros(self.scale_size)
                    dy = int((self.scale_size[0]-ch)/2.0)
                    dx = int((self.scale_size[1]-cw)/2.0)
                
                #new_img = np.zeros(self.scale_size)
                #dy = int((self.scale_size[0]-ch)/2.0)
                #dx = int((self.scale_size[1]-cw)/2.0)

                imre = imre.astype('float')
                
                new_img[dy:dy+ch,dx:dx+cw] = imre
                #new_img /= 256.0
                #print(new_img.shape)
                
                return new_img,hfirst

        
        def __getitem__(self,index):
                
                imgfile = self.imglist[index]
                writer = self.idx_tab[self._get_identity(imgfile)]
                
                image = imageio.imread(self.folder + imgfile,mode='L')
                image,hfirst = self.resize(image)
                image = image / 255.0

                image = self.transform()(image)
                writer = torch.from_numpy(np.array(writer))
                
                return image,writer,imgfile # images, writer_ids, image_names
        
        def __len__(self):
                return len(self.imglist)
        

class DatasetFromFolder1(data.Dataset):
    def __init__(self, csv_path, image_folder, scale_size=(64, 256), is_training=True):
        super(DatasetFromFolder1, self).__init__()

        if  isinstance(csv_path,tuple): 
                self.csv_path = list(csv_path)[0]
        else:
                #print(" path type:",type(csv_path))
                self.csv_path = csv_path
       
        self.is_training = is_training
        self.scale_size = scale_size
        self.image_folder = image_folder
        
        print("\n\t csv:",self.csv_path)
        # Read the CSV file
        self.data = pd.read_csv(self.csv_path)
        
        print("\n\t datashape:",self.data.shape," \t columns:",self.data.columns)
        print("\n\t writer min:",self.data['writer_id'].min()," max:",self.data['writer_id'].max())
        
        # Print some info about the dataset
        print('-' * 10)
        print(f'Loading dataset from {csv_path}')
        print(f'Number of samples: {len(self.data)}')
        print('-' * 10)

        # Define the transformations
        self.transform = Compose([ToTensor()])

    def resize(self, image):
        # Implement the image resizing logic here
        ...

    def __getitem__(self, index):
        xml_file = self.data['xml_file'][index]
        writer_id = self.data['writer_id'][index]
        image_name = self.data['word_id'][index]

        image_path = os.path.join(self.image_folder, image_name)+".png"
        #image = self.load_image(image_path)
        #image = misc.imread(image_path,mode='L')
        
        try:
                image = imageio.imread(image_path, pilmode='L')
        except Exception as e:
                
                image_name1 = image_name.split("__")[0]
                image_path = os.path.join(self.image_folder, image_name1)+".tif"
                image = imageio.imread(image_path, pilmode='L')
                
        
        #print("\n\t image =",image.shape)
        #image, hfirst = self.resize(image)
        
        image = image / 255.0
        image = self.transform(image)

        return image, writer_id, image_name

    def load_image(self, image_path):
        # Implement the image loading logic here
        ...

    def __len__(self):
        return len(self.data)



class DatasetFromFolder3(data.Dataset):
    def __init__(self, csv_path, image_folder,aug=None, scale_size=(64, 256), is_training=True):
        super(DatasetFromFolder3, self).__init__()

        self.aug = aug
        if  isinstance(csv_path,tuple): 
                self.csv_path = list(csv_path)[0]
        else:
                #print(" path type:",type(csv_path))
                self.csv_path = csv_path
       
        self.is_training = is_training
        self.scale_size = scale_size
        self.image_folder = image_folder
        
        print("\n\t self.image_folder =",self.image_folder)
        print("\n\t csv:",self.csv_path)
        # Read the CSV file
        self.data = pd.read_csv(self.csv_path)
        
        print("\n\t datashape:",self.data.shape," \t columns:",self.data.columns)
        #print("\n\t writer min:",self.data['writer_id'].min()," max:",self.data['writer_id'].max())
        
        # Print some info about the dataset
        print('-' * 10)
        print(f'Loading dataset from {csv_path}')
        print(f'Number of samples: {len(self.data)}')
        print('-' * 10)

        # Define the transformations
        self.transform = Compose([ToTensor()])
        
        
        
    def affine_transformation(self,img):
        m=1.0 
        s=0.2 
        border_value=None
        
        h, w = img.shape[0], img.shape[1]
        src_point = np.float32([[w / 2.0, h / 3.0],
                                [2 * w / 3.0, 2 * h / 3.0],
                                [w / 3.0, 2 * h / 3.0]])
        random_shift = m + np.random.uniform(-1.0, 1.0, size=(3,2)) * s
        dst_point = src_point * random_shift.astype(np.float32)
        transform = cv2.getAffineTransform(src_point, dst_point)
        if border_value is None:
                border_value = np.median(img)
        warped_img = cv2.warpAffine(img, transform, dsize=(w, h), borderValue=float(border_value))
        return warped_img

    def __getitem__(self, index):
        #xml_file = self.data['xml_file'][index]
        
        #print("\n\t columns:",self.data.columns)
        
        writer_id = self.data['writer_id'][index]
        image_name = self.data['line_id'][index]
        
        """
        if "__" in image_name:
                image_name = image_name.split("__")[0]
                image_name = image_name
        """

        image_path = os.path.join(self.image_folder, image_name)+".png"
        
        #print("\n\t is file:",os.path.isfile(image_path)," image_path:",image_path)
        
        #image = self.load_image(image_path)
        #image = misc.imread(image_path,mode='L')
        
        try:
                image = imageio.imread(image_path, pilmode='L')
                
        except Exception as e:
                
                image_name1 = image_name.split("__")[0]
                image_path = os.path.join(self.image_folder, image_name1)+".png"
                image = imageio.imread(image_path, pilmode='L')

        
        #image, hfirst = self.resize(image)
        
        image = image / 255.0
        
        if self.aug ==1:
                image = self.affine_transformation(image)        
        
        
        image = self.transform(image)

        #print("\n\t 1.image_name =",image_name.shape)


        return image, writer_id, image_name

    def load_image(self, image_path):
        # Implement the image loading logic here
        ...

    def __len__(self):
        return len(self.data)



import random

class DatasetFromFolder2(data.Dataset):
    def __init__(self, csv_path, image_folder, aug=None, scale_size=(64, 256), is_training=True):
        super(DatasetFromFolder2, self).__init__()

        self.aug = aug
        self.csv_path = csv_path if not isinstance(csv_path, tuple) else list(csv_path)[0]
        self.is_training = is_training
        self.scale_size = scale_size
        self.image_folder = image_folder

        # Read the CSV file containing metadata
        #self.data = pd.read_csv(self.csv_path,nrows=1000)
        
        logger.info("\n\t csv_path:%s",self.csv_path)
        
        self.data = pd.read_csv(
        self.csv_path,
        header=None,
        names=["writer_id", "line_id", "text"],
        delimiter=",",
        quotechar='"',
        encoding='utf-8', engine='python'
        )
        #df = pd.read_csv(file_path, sep=',', header=None, names=["writer_id", "line_id", "text"], encoding='utf-8', engine='python')

        print("\n\t datashape:",self.data.shape," \t columns:",self.data.columns," \t csv_path:",self.csv_path)
        
        print("self.data:\n",self.data.head())
        # Group images by writer_id for positive/negative sampling
        self.writer_groups = self.data.groupby('writer_id').groups

        print("\n\t writer min:",self.data['writer_id'].min()," max:",self.data['writer_id'].max())

        
        def add_gaussian_noise(tensor, mean=0.0, std=0.1):
                noise = torch.randn(tensor.size()) * std + mean
                return tensor + noise

        # Define transformations            
        """
        self.transform = Compose([
            ToTensor(),
            Lambda(lambda x: add_gaussian_noise(x, mean=0.0, std=0.1)),  # First noise addition
        ])
        """
        
        self.transform = Compose([
            ToTensor()
        ])

    def __getitem__(self, index):
        # Anchor image and writer_id
        anchor_info = self.data.iloc[index]
        anchor_image, anchor_writer_id,anchor_image_name = self._load_image(anchor_info)

        # Positive: Choose an image with the same writer_id
        positive_index = random.choice(self.writer_groups[anchor_writer_id])
        positive_info = self.data.iloc[positive_index]
        positive_image, positive_writer_id,positive_image_name = self._load_image(positive_info)

        # Negative: Choose an image from a different writer
        negative_writer_id = random.choice(list(self.writer_groups.keys()))
        while negative_writer_id == anchor_writer_id:
            negative_writer_id = random.choice(list(self.writer_groups.keys()))
        negative_index = random.choice(self.writer_groups[negative_writer_id])
        negative_info = self.data.iloc[negative_index]
        negative_image, negative_writer_id,negative_image_name = self._load_image(negative_info)
        
        
        anchor_image = anchor_image.repeat(1, 3, 1, 1)  # Repeat the 1 channel to create 3 channels
        #anchor_image = anchor_image.squeeze(1)

        positive_image = positive_image.repeat(1, 3, 1, 1)  # Repeat the 1 channel to create 3 channels
        #positive_image = positive_image.squeeze(1)
        
        negative_image = negative_image.repeat(1, 3, 1, 1)  # Repeat the 1 channel to create 3 channels
        #negative_image = negative_image.squeeze(1)
        
        return anchor_image,anchor_writer_id,anchor_image_name, positive_image, negative_image

    def _load_image(self, image_info):
        # Load image based on provided info
        image_name = image_info['line_id']  # Extract using line_id, not word_id
        writer_id = image_info['writer_id']

        image_path = os.path.join(self.image_folder, image_name) + ".png"
        #logger.info("\n\t 1.image_path:%s",image_path)
        #print("\n\t 1.image_path:ÆÆÆ",image_path)
        """
                modification for norwegian dataset
        """
        if os.path.isfile(image_path) == False:
            image_path = os.path.join(self.image_folder, image_name)

        try:
            image = imageio.imread(image_path, pilmode='L')
        except Exception:
            image_name1 = image_name.split("__")[0]
            image_path = os.path.join(self.image_folder, image_name1) + ".png"
            image = imageio.imread(image_path, pilmode='L')

        image = image / 255.0
        image = self.transform(image)
        #image = image.squeeze(1)

        return image, writer_id,image_name

    def __len__(self):
        return len(self.data)


if __name__=="__main__":
        
        """
        image_folder = "/cluster/datastore/aniketag/allData//IAM/allLinesCrop/" # "/cluster/datastore/aniketag/allData/wordStylist//allCrops_preprocess//"

        dataset = DatasetFromFolder1(csv_path="/cluster/datastore/aniketag/allData/IAM/output_split.csv",
                                     image_folder=image_folder)
        """
        
        image_folder = "/cluster/datastore/aniketag/allData/wordStylist//allCrops_preprocess//"
        #csv_path = "/cluster/datastore/aniketag/writerClassification/writer-identification/lineWriter.csv"
        csv_path = "./data/gan_iam_tr_va_gt.csv"
        
        dataset = DatasetFromFolder3(csv_path= csv_path,image_folder=image_folder)

        
        # Create the iterator
        data_loader = data.DataLoader(dataset, batch_size=2, shuffle=True)

        # Iterate over the dataset
        
        indx = 0
        # anchor_image_name,anchor_writer_id,anchor_image, positive_image, negative_image
        for image_name,writer_id,anchor_image, positive_image, negative_image in data_loader:
                # Process the batch of images and writer IDs here
                print("\n\t image_name:",image_name)

                print("\n\t 0.anchor_image:",anchor_image.shape,"\t positive_image:",positive_image.shape," negative_image.shape:",negative_image.shape)
               
                anchor_image = anchor_image.squeeze(1)
                positive_image = positive_image.squeeze(1)
                negative_image = negative_image.squeeze(1)
               
                print("\n\t 1.anchor_image:",anchor_image.shape,"\t positive_image:",positive_image.shape," negative_image.shape:",negative_image.shape)
                
                print("\t writer_ids dtype:",writer_id.shape)
                
                                
                if indx ==10:
                        break
                else:
                        indx+=1
                