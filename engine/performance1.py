import math
filename1='DNNRegOut'
filename2='DNNRegShallowOut'
dataFilesMaster= open('../data/data_domains5.csv','r',encoding="utf8")
dataFile1= open("../data/"+filename1+'.csv','r',encoding="utf8")
dataFile2= open("../data/"+filename2+'.csv','r',encoding="utf8")

outputFile= open("../data/performance.csv","w+")

masterDirectory = {
        "Id" : 0

    }


for data in dataFilesMaster:

    data1=data.split(",")
    if data1[0]=="":
        continue
    if len(data1)<2:
        continue
    masterDirectory[data1[0]]=data1[47].replace("\n","")
    

#outputFile.write("ID,ActualPrice,DNN_PRICE,Difference,Percent_DIFF  \n")


outData=[]
count=0
total1=0
total2=0
perc0T5_1=0
perc5T10_1=0
perc10T20_1=0
perc20T50_1=0
perc50Plus_1=0
mse1=0

perc0T5_2=0
perc5T10_2=0
perc10T20_2=0
perc20T50_2=0
perc50Plus_2=0
mse2=0


dataFileList1=[]
dataFileList2=[]

for data in dataFile1:
    dataFileList1.append(data.replace("\n",""))
for data in dataFile2:
    dataFileList2.append(data.replace("\n",""))
del dataFileList1[0]    
del dataFileList2[0] 
outputFile.write("\n"+",DNN REGRESSION,,,,,DNN  SHALLOW"+"\n")   

stringss="ID,ActualPrice,DNN_PRICE,Difference,Percent_DIFF,,ID,ActualPrice,DNN_Shallow_PRICE,Difference,Percent_DIFF"

outputFile.write("\n"+stringss+"\n")   
while count <len(dataFileList1):

    outData=[]

    data1=dataFileList1[count].split(",")
    data2=dataFileList2[count].split(",")
    
    actualPrice1=masterDirectory[data1[1]]
    difference1=(abs(  float(data1[0])-float(masterDirectory[data1[1]])   ))
    percent1=difference1*100/float(actualPrice1)
    total1=total1+percent1
    mse1=mse1+(difference1*difference1)
    if percent1<5:
        perc0T5_1=perc0T5_1+1
    elif percent1 <10:
        perc5T10_1=perc5T10_1+1
    elif percent1<20:
        perc10T20_1=perc10T20_1+1
    elif percent1<50:
        perc20T50_1=perc20T50_1+1
    else:
        perc50Plus_1=perc50Plus_1+1    
            
    
    actualPrice2=masterDirectory[data2[1]]
    difference2=(abs(  float(data2[0])-float(masterDirectory[data2[1]])   ))
    percent2=difference2*100/float(actualPrice2)
    total2=total2+percent2
    mse2=mse2+(difference2*difference2)
       

    if percent2<5:
        perc0T5_2=perc0T5_2+1
    elif percent2 <10:
        perc5T10_2=perc5T10_2+1
    elif percent2<20:
        perc10T20_2=perc10T20_2+1
    elif percent2<50:
        perc20T50_2=perc20T50_2+1
    else:
        perc50Plus_2=perc50Plus_2+1      
    
    
    outData.append(str(data1[1]))
    outData.append(str(actualPrice1))
    outData.append(str(data1[0]))
    outData.append(str(difference1))
    outData.append(str(percent1))
    
    
    outData.append("")
    
    outData.append(str(data2[1]))
    outData.append(str(actualPrice2))
    outData.append(str(data2[0]))
    outData.append(str(difference2))
    outData.append(str(percent2) ) 
    
    
    stringData=','.join(outData) 

    outputFile.write("\n"+stringData)
    count=count+1


outputFile.write("\n "+  filename1+"\n")
outputFile.write("\nAverage Difference:,- ,"+str(total1/count))
outputFile.write("\nRMSE:,- ,"+str(math.sqrt(mse1/count)))
outputFile.write("\nRecords with Difference, 0%- 5%: ,"+str(perc0T5_1*100/count))
outputFile.write("\nRecords with Difference, 5%-10%: ,"+str(perc5T10_1*100/count))
outputFile.write("\nRecords with Difference, 10% -20%: ,"+str(perc10T20_1*100/count))
outputFile.write("\nRecords with Difference, 20% -50%: ,"+str(perc20T50_1*100/count))
outputFile.write("\nRecords with Difference, 50%+: ,"+str(perc50Plus_1*100/count))

outputFile.write("\n\n  "+ filename2 +"\n")
outputFile.write("\nAverage Difference:,- ,"+str(total2/count))
outputFile.write("\nRMSE:,- ,"+str(math.sqrt(mse2/count)))
outputFile.write("\nRecords with Difference, 0%- 5%: ,"+str(perc0T5_2*100/count))
outputFile.write("\nRecords with Difference, 5%-10%: ,"+str(perc5T10_2*100/count))
outputFile.write("\nRecords with Difference, 10% -20%: ,"+str(perc10T20_2*100/count))
outputFile.write("\nRecords with Difference, 20% -50%: ,"+str(perc20T50_2*100/count))
outputFile.write("\nRecords with Difference, 50%+: ,"+str(perc50Plus_2*100/count))
outputFile.close()
