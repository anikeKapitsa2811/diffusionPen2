import json

gt_train = "/cluster/datastore/aniketag/newWordStylist/wordStylist2/WordStylist/gt/cvlTrain_sorted.txt"

lang = "GerEng"

with open(gt_train, 'r') as f:
    train_data = f.readlines()
    train_data = [i.strip().split(' ') for i in train_data]
    
    wr_dict = {}
    full_dict = {}
    image_wr_dict = {}
    img_word_dict = {}
    writerImageDict = {}
    wr_index = 0
    idx = 0
    maxWordLen = 0
    
    for indx,i in enumerate(train_data):
        
        #print("\n\t i:",i)
        
        s_id = i[0].split(',')[0]
        #image = i[0].split(',')[1] + '.png'
        
        if lang == "GerEng":
            image = i[0].split(',')[1] + '.tif'
        else:
            image = i[0].split(',')[1]+ '.png'
        print("\n\t 1.i:",i)
        
        try:
            transcription = i[1]
        except Exception as e:
            #print("\n\t exception:",e)
            transcription = i[0].split('-')[-1]        
        
        #print(s_id)
        full_dict[idx] = {'image': image, 's_id': s_id, 'label':transcription}
        image_wr_dict[image] = s_id
        img_word_dict[image] = transcription
        idx += 1
        if s_id not in wr_dict.keys():
            wr_dict[s_id] = wr_index
            wr_index += 1
            
        maxWordLen = max(maxWordLen,len(transcription))
        
        if s_id in writerImageDict:
            writerImageDict[s_id].append(image)
        else:
            writerImageDict[s_id] = list()
            writerImageDict[s_id].append(image)
    
    """
    logger.info("maxWordLen:%s total record:%s",maxWordLen,i)
    print('number of train writer styles', len(wr_dict))
    """
    style_classes=len(wr_dict)
    
    print("\n\t maxWordLen:",maxWordLen,"\t total record:",idx," len(writerImageDict.keys()):",len(writerImageDict.keys()))
    
    print('\n\t number of train writer styles:', len(wr_dict)," style_classes=",style_classes)
    
    style_classes=len(wr_dict)

# create json object from dictionary if you want to save writer ids
json_dict = json.dumps(wr_dict)
f = open("./writers_dict_train_CVL.json","w")
f.write(json_dict)
f.close()
