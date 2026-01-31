import os

f1 = "/cluster/datastore/aniketag/newWordStylist/DiffusionPen2/writerEmbeding/train.txt"
f2 = "/cluster/datastore/aniketag/newWordStylist/DiffusionPen2/writerEmbeding/test.txt"

# line by line read

def read_file_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

l1 = read_file_lines(f1)
l2 = read_file_lines(f2)

d1= dict()
d2 = dict()

for line in l1:
    if len(line) > 0:
        imgName= line.split(",")[1]
        d1[imgName] = 1
        
        
for line in l2:
    if len(line) > 0:
        imgName= line.split(",")[1]
        d2[imgName] = 1


# check overlap
overlap = set(d1.keys()).intersection(set(d2.keys()))
print(f"Total images in train: {len(d1)}")
print(f"Total images in test: {len(d2)}")
print(f"Total overlapping images: {len(overlap)}")
# 5 images fraom each dict

print(list(d1.keys())[:5])
print(list(d2.keys())[:5])