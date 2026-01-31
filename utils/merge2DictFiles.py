import os
import torch

basePath = "/cluster/datastore/aniketag/newWordStylist/DiffusionPen2/"

path1 = "/cluster/datastore/aniketag/newWordStylist/DiffusionPen2/writerEmbeding/writer_style_refs.pth"
path2 = "/cluster/datastore/aniketag/newWordStylist/DiffusionPen2/writerEmbeding/writer_style_refs2.pth"

fstDict = torch.load(path1)
sDict = torch.load(path2)

# merge and create new dict

for key in sDict.keys():
    if key not in fstDict:
        fstDict[key] = sDict[key]
    else:
        print(f"Key {key} already exists in fstDict. Skipping...")

# total keys
print(f"Total keys in merged dictionary: {len(fstDict.keys())}")


