import os

basePath = "/cluster/datastore/aniketag/newWordStylist/DiffusionPen2/utils/splits_words/"
filePaths = ["/iam_training.txt","iam_test.txt","iam_val.txt","iam_train_val.txt"]
dict={}

for fileName in filePaths:
    filePath = basePath+fileName #os.path.join(basePath, fileName)

    print("Processing file: ", filePath," with base path: ", basePath)
    with open(filePath, 'r') as f:
        lines = f.readlines()
    
    
    for l in lines:
        imageName = l.split(".png")[0]
        imageName = imageName.split("/")[-1]
        writer = l.split(".png")[1]
        writer = writer.split(",")[1]
        
        dict [imageName] = writer
        
        
# Save the dictionary to a file as a json
import json
outputFilePath = os.path.join(basePath, "imageWriterDict.json")
with open(outputFilePath, 'w') as f:
    json.dump(dict, f, indent=4)
#load the dictionary from the file
def loadImageWriterDict():
    with open(outputFilePath, 'r') as f:
        return json.load(f)

d = loadImageWriterDict()
print("Number of images in the dictionary: ", len(d))