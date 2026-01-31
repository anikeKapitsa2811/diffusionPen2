import os

path = "/cluster/datastore/aniketag/newWordStylist/DiffusionPen/noisyImageDump/"

d = dict()

totFiles = os.listdir(path)
print("Total files:", len(totFiles))

for file in totFiles:
    
    file = file.split("_")[0]  # Remove file extension
    if file not in d:
        d[file] = 1
    else:
        d[file] += 1

print("Unique Files", len(d))

