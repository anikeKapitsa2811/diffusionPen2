import os
import pandas as pd

train= pd.read_csv("/cluster/datastore/aniketag/writerClassification/writer-identification/data/diffPen3.csv")
test= pd.read_csv("/cluster/datastore/aniketag/newWordStylist/DiffusionPen2/logs/test.csv")
val = pd.read_csv("/cluster/datastore/aniketag/newWordStylist/DiffusionPen2/logs/val.csv")

# 1st column is the writer id read fraom all three files
# and find the common writers in all three files
def common_writers(train, test, val):
    train_writers = set(train.iloc[:, 0])
    test_writers = set(test.iloc[:, 0])
    val_writers = set(val.iloc[:, 0])
    
    common = train_writers.intersection(test_writers).intersection(val_writers)
    
    return common
common = common_writers(train, test, val)

print(f"Number of common writers: {len(common)}")

# similarly read 2nd column from all three files 
# and find common images in all three files
def common_images(train, test, val):
    train_images = set(train.iloc[:, 1])
    test_images = set(test.iloc[:, 1])
    val_images = set(val.iloc[:, 1])
    
    common = train_images.intersection(test_images).intersection(val_images)
    
    return common
common_images = common_images(train, test, val)
print(f"Number of common images: {len(common_images)}")
