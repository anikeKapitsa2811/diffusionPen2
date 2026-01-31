"""
    1.check iamPath,
    2. gt_train file
    3. language,MAX_CHARS,
    4.loadPrev,baseModelDir,authorBasePath,ckptModelName,emaModelName,save_path
    5. variation encoder saved images
    6.no of writers
    7.global embedding path
"""

import os
import sys
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

lang = ["NOR","ENG","GerEng"][1]

print("\n\t language:",lang)

allInOneIndx = 0
MAX_CHARS = [10,10,"",13][allInOneIndx]

if lang == "NOR":
    MAX_CHARS = 25
elif lang == "GerEng":
    MAX_CHARS = 13

print("\n\t MAX_CHARS=",MAX_CHARS)
# /global/D1/projects/ZeroShot_Word_Recognition/E2E/allData/WordStylist trainSplitIamDAachen.txt
if lang == "ENG":
    gt_train = [""][allInOneIndx]
    
    gnhkPath = "/cluster/datastore/aniketag/allData/diffPen//saved_GNHK_data//train_word_GNHK.pt"
    #iam_path = '/global/D1/projects/ZeroShot_Word_Recognition/E2E/allData/IAM//'
    
    #vaeFromDictPath = "/global/D1/projects/ZeroShot_Word_Recognition/E2E/allData/WordStylist/writerStyle/imageWordLineVae3.pkl"
    #vaeFromDictPath ="/global/D1/projects/ZeroShot_Word_Recognition/E2E/allData/WordStylist/writerStyle/imageWordLineVae3OnlyChar.pkl"

    
elif lang == "NOR":
    gt_train = "/global/D1/projects/ZeroShot_Word_Recognition/E2E/allData/wordStylist/allCrops_preprocess_norwegian_gt/norwegian9000_train_0_All.filter27"
elif lang == "GerEng":
    gt_train ="/cluster/datastore/aniketag/newWordStylist/wordStylist2/WordStylist/gt/cvlTrain.txt"
    cvl_path ="/cluster/datastore/aniketag/allData/CVL/allCrops_preprocess/"
    #vaeFromDictPath = "/global/D1/projects/ZeroShot_Word_Recognition/E2E/allData/WordStylist/writerStyle/imageWordCVLVae.pkl"
    
    # /cluster/datastore/aniketag/allData/CVL
    
    vaeFromDictPath = "/cluster/datastore/aniketag/allData/CVL/imageWordCVLVae.pkl"
    #globalStylePath = "/cluster/datastore/aniketag/allData/CVL/cropWriterStyleEmbMobilenetV2CVL.pkl"
    globalStylePath = "/cluster/datastore/aniketag/allData/CVL/cropWriterStyleEmbMobilenetV2CVL_WriterTrain.pkl"


if lang == "ENG":
    dataIndx = 1
elif lang == "NOR":
    dataIndx = 0
elif lang == "GerEng":
    dataIndx = 3
    
csvRead = ["/global/D1/projects/ZeroShot_Word_Recognition/E2E/allData/IAM/lineData.csv",None][0]

#baseModelDir = "/cluster/datastore/aniketag/allData/wordStylist/models/"

baseModelDir = "/cluster/datastore/aniketag/allData/diffPen//style_models//" 

dataset_folder = '/cluster/datastore/aniketag/allData/GNHK/allCrops_preprocess/'

authorBasePath = [
                  baseModelDir,
                  ][0]
# /global/D1/projects/ZeroShot_Word_Recognition/E2E/allData/WordStylist/models/icdar2025/CVL
ckptModelName =[""][0]

emaModelName  = [""][0]

styleModel = ["GNHK_classification_mobilenetv2_100.pth","triplet_GNHK_mobilenetv2_100.pth"]
styleClssifierModelPath = baseModelDir+"/style_models/"+styleModel[0]

if lang == "ENG":
    save_path = [authorBasePath][0]
elif lang == "NOR":
    save_path = ["/global/D1/projects/ZeroShot_Word_Recognition/E2E/allData/wordStylist/models/Norwegian/Mse_Nor_text_condi_FromScratch/models/",
                 "/global/D1/projects/ZeroShot_Word_Recognition/E2E/allData/wordStylist/models/Norwegian/Mse_Nor_text_Phos_condi_FromScratch/"][allInOneIndx]
elif lang == "GerEng":
    save_path = ["","",authorBasePath][2]

if lang == "ENG":
    saveModelName = [authorBasePath,authorBasePath][allInOneIndx]
elif lang == "NOR":
    saveModelName = "temp.pt"
elif lang == "GerEng":
    saveModelName = [ckptModelName,authorBasePath][0]
    
#optModelName = ["optim_Mse_text_Phos_condi_FromScratch.pt"][0]
device = "cuda"

