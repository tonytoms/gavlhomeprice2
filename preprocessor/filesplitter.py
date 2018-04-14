

dataFiles= open('../data/data_domains5.csv','r',encoding="utf8")
dataFilesTr= open('../data/train.csv','w',encoding="utf8")
dataFilesTs= open('../data/test.csv','w',encoding="utf8")

count=-1

splitter={}
splitter2={}

datas=[]
for dataStr in dataFiles:
 
    count=count+1
    if count==0:
        continue
    
    data=dataStr.split(",")
    datas.append(data)
    dateKey=data[6][:-2]
    dateKey=dateKey + data[1].split(" ")[0].lower()
    if dateKey in splitter:
        splitter.update({ dateKey: splitter[dateKey]+":"+data[0] }  )
    else:
        splitter[dateKey] =data[0]
        

testIDs=[]
testCount=int(count*30/100)
print(testCount)
count=-1
for i in splitter:
    
    count=count+1
    IDs=splitter[i].split(":")
    #print(IDs)
    testIDs.append(IDs[0])
    
    if count>=testCount:
        break
    
count=0
for ID in testIDs:
    count=count+1
    print(str(count)+":"+ID)
    
count=-1
dataFiles.close()            

dataFiles= open('../data/data_domains5.csv','r',encoding="utf8")

for dataStr in dataFiles:
    count=count+1
    if count==0:
        dataFilesTr.write(dataStr)
        dataFilesTs.write(dataStr)
    else:
        dataS= dataStr.split(",")
        if dataS[0] in testIDs:
            dataFilesTs.write(dataStr)
        else:
            dataFilesTr.write(dataStr)

dataFilesTs.close()
dataFilesTr.close()
dataFiles.close()            
