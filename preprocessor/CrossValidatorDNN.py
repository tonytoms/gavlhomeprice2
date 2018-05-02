import os
import subprocess
import sys

dataFiles= open('../data/train.csv','r',encoding="utf8")

trainData= open('../data/train1.csv','w',encoding="utf8")

maincount=-1
headerdata=""
for data in dataFiles:
    if maincount==-1:
        headerdata=data
    maincount=maincount+1

KFold=10
dataFiles.close()
dataFiles= open('../data/train.csv','r',encoding="utf8")

splitter=int(maincount/KFold)
testData= open('../data/test1.csv','w',encoding="utf8")
testData.write(headerdata.rsplit(',', 1)[0])
testData.write('\n')

count=-1
fileCounter=1
for data in dataFiles:
    
    count=count+1
    if count==0:
        continue
    if count%splitter==0:
        testData.close()
        fileCounter=fileCounter+1
        testData= open('../data/test'+str(fileCounter)+'.csv','w',encoding="utf8")
        testData.write(headerdata.rsplit(',', 1)[0])
        testData.write('\n')
        testData.write(data.rsplit(',', 1)[0])

    else:
        testData.write(data.rsplit(',', 1)[0])
    testData.write("\n")

testData.close()
dataFiles.close()

dataFiles= open('../data/train.csv','r',encoding="utf8")
for i in range(1,12):
    testData= open('../data/test'+str(i)+'.csv','r',encoding="utf8")
    IDs=[]
    for data in testData:
        IDs.append(data.split(',')[0])
    trainData.close()
    trainData= open('../data/train'+str(i)+'.csv','w',encoding="utf8")
    
    dataFiles.close()
    dataFiles= open('../data/train.csv','r',encoding="utf8")
    trainData.write(headerdata)

    for data in dataFiles:
        if data.split(',')[0] in IDs:
            continue
        else:
            trainData.write(data)



for i in range(1,12):
    inputTrainFile=""
    inputTestFile=""
    DNNFileOut=""
    DNNSHallowOut=""       
    feedFile= open('../files/DNNfeedO.csv','r',encoding="utf8")
    for data in feedFile:
        inputTrainFile=data.split(',')[0]
        print(inputTrainFile)
        print(str(data))
        inputTestFile=data.split(',')[1]
        DNNFileOut=data.split(',')[2]
        DNNSHallowOut=data.split(',')[3]        
    feedFile.close()
    feedFile= open('../files/DNNfeed.csv','w',encoding="utf8")
    feedFile.write(inputTrainFile+str(i)+","+inputTestFile+str(i)+","+DNNFileOut+str(i)+","+DNNSHallowOut+str(i))
    feedFile.close()
    subprocess.call([sys.executable, "../engine/DNNRegAndShallowNReg.py"])
