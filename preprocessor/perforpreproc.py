dataFiles= open('input/submission_13.1511653333_2018-02-07-17-05.csv','r',encoding="utf8")
dataFiles3= open('input/train.csv','r',encoding="utf8")
dataFiles2= open('input/compare.csv','w',encoding="utf8")

data3List=[]
counter=-1
for dataFile3 in dataFiles3:
    counter=counter+1
    if counter==0:
        continue
    data3=dataFile3.split(",")
    data3List.append(data3)

counter=-1
recordCount=0
for dataFile in dataFiles:
    counter=counter+1
    print(counter)
    data=dataFile.split(",")
    if len(dataFile)==1 or counter<=1:
        continue
    else:
        inpstr=""
        for datum in data:
            datum.replace("\n","")
            inpstr=inpstr+datum.replace('\n', ' ').replace('\r', '')
            inpstr=inpstr+","
        inpstr=inpstr+data3List[recordCount][0]
        inpstr=inpstr+","
        inpstr=inpstr+data3List[recordCount][42].replace('\n', ' ').replace('\r', '')
        inpstr=inpstr+","
        inpstr=inpstr+data3List[recordCount][0]
        inpstr=inpstr+","
        inpstr=inpstr+data3List[recordCount][0]
        inpstr=inpstr+","            
        inpstr=inpstr+"\n"
        recordCount=recordCount+1 
    dataFiles2.write(inpstr)               
    dataFiles2.flush()