import pandas as pd

dataFiles= open('../data/data_domains6.csv','r',encoding="utf8")
dataFilesTr= open('../data/train.csv','w',encoding="utf8")
dataFilesTs= open('../data/test.csv','w',encoding="utf8")

count=-1

splitter={}
splitter2={}
UniqueVals = []
datas=[]
for dataStr in dataFiles:
 
    count=count+1
    if count==0:
        continue
    
    data=dataStr.split(",")
    data.append(data[3]+"_"+data[5])

    if data[len(data)-1] not in UniqueVals:
        UniqueVals.append(data[len(data)-1])
        
    datas.append(data)
    


uniqueCount=len(UniqueVals)
totalCount=len(datas)

testSize=(totalCount/100)*30

splitSize=testSize/uniqueCount

splitCheck=[]

for i in range(0,uniqueCount):
    splitCheck.append(0)
    
if splitSize<1:
    for i in range(0,int(testSize)):
        splitCheck[i]=1
else:
    for i in range(0,uniqueCount):
        splitCheck[i]=splitSize    
        
        
test=[]
train=[]  


temp_df = pd.DataFrame(datas)


      
for data in datas:        
        
    uIndex=UniqueVals.index(data[len(data)-1])
    del data[len(data)-1]
    if splitCheck[uIndex]>0:
       
        test.append(data)
        splitCheck[uIndex]=splitCheck[uIndex]-1
    else:
        train.append(data)
 
print(len(test))        
print(len(train))           
if len(test) <testSize:
    for i in range(0,(testSize-len(test))):
        test.append(train[0])
        del train[0]        
print(len(test))        
print(len(train))        
        
test_df = pd.DataFrame(test)
train_df = pd.DataFrame(train)

test_df.to_csv('../data/test.csv', index=False, header=False)
train_df.to_csv('../data/train.csv', index=False, header=False)
dataFilesTs.close()
dataFilesTr.close()
dataFiles.close()            
