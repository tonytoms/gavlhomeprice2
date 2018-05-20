
dataFiles= open('../data/trainUnf.csv','r',encoding="utf8")
dataFiles2= open('../data/train.csv','w',encoding="utf8")


dataFilesT= open('../data/testUnf.csv','r',encoding="utf8")
dataFiles2T= open('../data/test.csv','w',encoding="utf8")

count=-1



datas=[]

#12 beds
#24 landsize
#25 year build
#40 price range min
#44 External Price
for dataStr in dataFiles:
 
    count=count+1

    
    data=dataStr.split(",")
    if data[12]=="0" or data[24]=="0"  or data[25]=="0" or data[40]=="0" or data[44]=="0" or data[47]=="0" :
        continue
    else:
        del data[1]
        dataStrMod = ','.join(data)
        dataFiles2.write(dataStrMod)
        
datas=[]
for dataStr in dataFilesT:
 
    count=count+1

    
    data=dataStr.split(",")
    if data[12]=="0" or data[24]=="0"  or data[25]=="0" or data[40]=="0" or data[44]=="0" or data[47]=="0" :
        continue
    else:
        del data[1]
        dataStrMod = ','.join(data)
        dataFiles2T.write(dataStrMod)