
dataFiles= open('../data/trainUnf.csv','r',encoding="utf8")
dataFiles2= open('../data/train.csv','w',encoding="utf8")


dataFilesT= open('../data/testUnf.csv','r',encoding="utf8")
dataFiles2T= open('../data/test.csv','w',encoding="utf8")

count=-1



datas=[]
for dataStr in dataFiles:
 
    count=count+1

    
    data=dataStr.split(",")
    if data[21]=="NA" or data[22]=="NA"  or data[37]=="NA" or data[39]=="NA" or data[43]=="NA" or data[10]=="NA" or data[42]=="NA":
        continue
    else:
        dataFiles2.write(dataStr)
        
datas=[]
for dataStr in dataFilesT:
 
    count=count+1

    
    data=dataStr.split(",")
    if data[21]=="NA" or data[22]=="NA"  or data[37]=="NA" or data[39]=="NA" or data[43]=="NA" or data[10]=="NA" or data[42]=="NA":
        continue
    else:
        dataFiles2T.write(dataStr)