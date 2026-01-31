import numpy as np 
from skimage import io as img_io
from utils.word_dataset_cvl import WordLineDataset
from utils.auxilary_functions import image_resize_PIL, centered_PIL
from PIL import Image, ImageOps
import json
import os
import string


class CVLDataset_style(WordLineDataset):
    def __init__(self, basefolder, subset, segmentation_level, fixed_size, transforms, args=None):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, transforms, args)
        self.setname = 'CVL'
        self.trainset_file = '{}/{}/set_split/trainset.txt'.format(self.basefolder, self.setname)
        self.valset_file = '{}/{}/set_split/validationset1.txt'.format(self.basefolder, self.setname)
        self.testset_file = '{}/{}/set_split/testset.txt'.format(self.basefolder, self.setname)
        self.line_file = '{}/ascii/lines.txt'.format(self.basefolder, self.setname)
        self.word_file = './iam_data/ascii/words.txt'.format(self.basefolder, self.setname)
        self.word_path = '{}/words'.format(self.basefolder, self.setname)
        self.line_path = '{}/lines'.format(self.basefolder, self.setname)
        self.forms = './iam_data/ascii/forms.txt'
        #self.stopwords_path = '{}/{}/iam-stopwords'.format(self.basefolder, self.setname)
        super().__finalize__()

    def main_loader(self, subset, segmentation_level) -> list:
        
        #print("1.here")
        
        def readFile(self,filePath):

            dict_path = '/cluster/datastore/aniketag/newWordStylist/DiffusionPen2/writers_dict_train_CVL.json'
            with open(dict_path, 'r') as f:
                wr_dict = json.load(f)

            
            gt = []
            #allFilterImages = os.listdir(self.basefolder)
            
            #print("\n\t allFilterImages =",len(allFilterImages))
            
            print("\n\t readFile filePath =",filePath)
            imageNames = dict()
            imageWriter= dict()
            with open(filePath, 'r') as txt_file:
                for line in txt_file:
                    
                    parts = line.strip().split(',')
                    
                    #print("\n\t parts:",parts) # parts: ['0', 'f04-071-01-07.png_the the']
                    
                    if len(parts) == 2:  # Make sure the line has two parts
                        writer_id = wr_dict[parts[0]]
                        image_info = parts[1].split(' ')
                        image_info = image_info
                        #print("\n\t writer_id =",writer_id ," image_info:",image_info)
                        if len(image_info) == 2:  # Make sure the image info has two parts
                            image_name = image_info[0]
                            
                            if 0:#subset == "train":
                                image_name = image_name[1:] 
                            
                            image_word = image_info[1]
                            
                            
                            if 1:#image_name+".png" in allFilterImages:
                                imageNames[image_name+".tif"] = image_word
                                gt.append((self.basefolder+image_name, image_word,writer_id))
                                
                            #print("\n\t image_name =",image_name,"\t image_word:",image_word," is file:",image_name+".png" in allFilterImages)
                            #print(os.path.isfile(self.basefolder+image_name+".png")," len(allFilterImages):",len(allFilterImages)," \t:",self.basefolder+image_name+".png")
                            #input("check!!")
            
            #gt.append((img_path, transcr))
            return imageNames,gt                   
        
        def gather_iam_info(self, set='test', level='word'):
            
            
            if  subset == 'train':
                
                imageNamesDict,gt = readFile(self,"/cluster/datastore/aniketag/newWordStylist/wordStylist2/WordStylist/gt/cvlTrain_sorted.txt")
                
                #imageNamesDict,_ = readFile(self,"/cluster/datastore/aniketag/newHTR/icpr/HTR-best-practices/data/5_fold_train_data.txt")
            
                print("\n\t train:",len(imageNamesDict.keys())," gt:",len(gt))
            
                #print(imageNamesDict)
            
            
            if subset == 'test':
                
                testFileName = "/cluster/datastore/aniketag/newWordStylist/wordStylist2/WordStylist/gt/cvlTest_sorted.txt"
                
                # test_data_5_fold.txt
                imageNamesDict,gt = readFile(self,testFileName)

                #imageNamesDict,_ = readFile(self,"./data/OovAllWritersTrainSets.txt")

                #imageNamesDict,_ = readFile(self,"/cluster/datastore/aniketag/newHTR/icpr/HTR-best-practices/data/5_fold_test.txt")
                print("\n\t test:",len(imageNamesDict.keys())," gt:",len(gt))

                print("\n\t testFileName =",testFileName)
                
            return gt

        
        info = gather_iam_info(self, subset, segmentation_level)
        
        
        data = []
        widths = []
        for i, (img_path, transcr, writer_name) in enumerate(info):
            if i % 1000 == 0:
                print('imgs: [{}/{} ({:.0f}%)]'.format(i, len(info), 100. * i / len(info)))
            #

            try:
                #print('img_path', img_path + '.png')
                img = Image.open(img_path + '.png').convert('RGB') #.convert('L')
                #print('img shape PIL', img.size)
                #img = image_resize_PIL(img, height=64)
                
                if img.height <= 64 and img.width <= 256:
                    img = img
                else:
                    img = image_resize_PIL(img, height=img.height // 2)
                
                #widths.append(img.size[0])
                
            except Exception as e:
                print('Could not add image file {}.png'.format(img_path), e)
                continue
                
            #except:
            #    print('Could not add image file {}.png'.format(img_path))
            #    continue

            # transform iam transcriptions
            transcr = transcr.replace(" ", "")
            # "We 'll" -> "We'll"
            special_cases  = ["s", "d", "ll", "m", "ve", "t", "re"]
            # lower-case 
            for cc in special_cases:
                transcr = transcr.replace("|\'" + cc, "\'" + cc)
                transcr = transcr.replace("|\'" + cc.upper(), "\'" + cc.upper())

            transcr = transcr.replace("|", " ")
            
            #writer_name = wr_dict[writer_name]
            
            data += [(img, transcr, writer_name, img_path)]
            
        return data




class CVL_Dataset(WordLineDataset):
    def __init__(self, basefolder, subset, segmentation_level, fixed_size,  tokenizer, text_encoder, feat_extractor, transforms, args):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, tokenizer, text_encoder, feat_extractor, transforms, args)
        self.setname = 'CVL'
        #self.trainset_file = f'{self.basefolder}/GNHK_words_train.txt'
        self.trainset_file= "/cluster/datastore/aniketag/newWordStylist/wordStylist2/WordStylist/gt/gnhk_1_ocr.txt"
        self.testset_file = self.trainset_file#f'{self.basefolder}/GNHK_words_test.txt'
        self.word_path = '/cluster/datastore/aniketag/allData/CVL/allCrops_preprocess/' #self.basefolder
        
        #self.stopwords_path = '{}/{}/iam-stopwords'.format(self.basefolder, self.setname)
        super().__finalize__()

    def main_loader(self, subset, segmentation_level) -> list:
        
        
        def gather_iam_info(self, set='train'):
            gtfile = self.trainset_file if subset == 'train' else self.testset_file
            gt = []
            folder = 'train_words' if subset == 'train' else 'test_words'
            for line in open(gtfile):
                
                #print("\n\t line:",line)
                if line.strip():
                    
                    
                    try:
                        style, image_name, transcription  = line.strip().split(' ')
                    except Exception as e:
                        print("\n\t Exception:", e)
                        style, image_name = line.strip().split(' ')
                        transcription  = line.split("_")[-1]
                        print("\n\t style, image_name, transcription:", style," ", image_name," ", transcription)
                    
                    image_name = image_name + '.png'
                    #img_path = os.path.join(self.word_path, folder, image_name)
                    img_path = os.path.join(self.word_path, image_name)

                    
                    gt.append((img_path, transcription, style))
            return gt

        info = gather_iam_info(self, subset)
        data = []
        widths = []
        wr_dict = {}
        character_classes = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
        for i, (img_path, transcr, writer_name) in enumerate(info):
            
            #create writer indexes 
            if writer_name not in wr_dict:
                wr_dict[writer_name] = len(wr_dict)
                
            style = wr_dict[writer_name]
            
            # transform iam transcriptions
            transcr = transcr.replace(" ", "")
            # "We 'll" -> "We'll"
            special_cases  = ["s", "d", "ll", "m", "ve", "t", "re"]
            # lower-case 
            # for cc in special_cases:
            #     transcr = transcr.replace("|\'" + cc, "\'" + cc)
            #     transcr = transcr.replace("|\'" + cc.upper(), "\'" + cc.upper())

            # transcr = transcr.replace("|", " ")
            
            if i % 1000 == 0:
                print('imgs: [{}/{} ({:.0f}%)]'.format(i, len(info), 100. * i / len(info)))
              
            #try:
            if 1:    
                img_original = Image.open(img_path).convert('RGB') #.convert('L')
                
                #if the transcription is in stopwords
                if transcr in string.punctuation:
                    img = centered_PIL(img_original, (64, 256), border_value=255.0)
                
                else:
                    (img_width, img_height) = img_original.size
                    #resize image to height 64 keeping aspect ratio
                    img = img_original.resize((int(img_width * 64 / img_height), 64))
                    (img_width, img_height) = img.size
                    
                    if img_width < 256:
                        outImg = ImageOps.pad(img, size=(256, 64), color= "white")#, centering=(0,0)) uncommment to pad right
                        img = outImg
                    
                    else:
                        #reduce image until width is smaller than 256
                        while img_width > 256:
                            img = image_resize_PIL(img, width=img_width-20)
                            (img_width, img_height) = img.size
                        img = centered_PIL(img, (64, 256), border_value=255.0)
                        #img = image_resize_PIL(img, height=img.height // 2)
                
            #except:
            #   continue
            
            
            
            
            
            data += [(img, transcr, style, img_path)]
            
        print('len data', len(data))
        print("data:", data[0])
        
        #save writer_dict
        with open(f'writer_dict_train_gnhk.json', 'w') as f:
            json.dump(wr_dict, f)
        
        return data