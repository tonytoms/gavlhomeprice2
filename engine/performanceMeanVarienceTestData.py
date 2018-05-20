import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

filename1='DNNOut'
filename2='lasso_ridge_xgb_v2'

dataFilesMaster= open('../data/data_domains5.csv','r',encoding="utf8")
dataFile1= open("../data/"+filename1+'.csv','r',encoding="utf8")
dataFile2= open("../data/"+filename2+'.csv','r',encoding="utf8")

outputFile= open("../data/performanceTrain.csv","w+")
outputFile2= open("../data/performanceMeanVarianceDataTrain.csv","w+")

outputFile2.write("Head,"+filename1+","+filename2+"\n")
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
diffPercentList1=[]

perc0T5_2=0
perc5T10_2=0
perc10T20_2=0
perc20T50_2=0
perc50Plus_2=0
mse2=0
diffPercentList2=[]


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
    diffPercentList1.append(percent1)

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
    diffPercentList2.append(percent2)
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

RMSE1=math.sqrt(mse1/count)
RMSE2=math.sqrt(mse2/count)
avgDiff1=total1/count
avgDiff2=total2/count
outputFile.write("\n "+  filename1+"\n")
outputFile.write("\n Mean Difference %:,- ,"+str(avgDiff1))
outputFile.write("\nRMSE:,"+filename1 +","+str(RMSE1))
outputFile.write("\nRecords with Difference, 0%- 5%: ,"+str(perc0T5_1*100/count))
outputFile.write("\nRecords with Difference, 5%-10%: ,"+str(perc5T10_1*100/count))
outputFile.write("\nRecords with Difference, 10% -20%: ,"+str(perc10T20_1*100/count))
outputFile.write("\nRecords with Difference, 20% -50%: ,"+str(perc20T50_1*100/count))
outputFile.write("\nRecords with Difference, 50%+: ,"+str(perc50Plus_1*100/count))

outputFile.write("\n\n  "+ filename2 +"\n")
outputFile.write("\n Mean Difference %:,- ,"+str(avgDiff2))
outputFile.write("\nRMSE:,"+filename2 +","+str(RMSE2))
outputFile.write("\nRecords with Difference, 0%- 5%: ,"+str(perc0T5_2*100/count))
outputFile.write("\nRecords with Difference, 5%-10%: ,"+str(perc5T10_2*100/count))
outputFile.write("\nRecords with Difference, 10% -20%: ,"+str(perc10T20_2*100/count))
outputFile.write("\nRecords with Difference, 20% -50%: ,"+str(perc20T50_2*100/count))
outputFile.write("\nRecords with Difference, 50%+: ,"+str(perc50Plus_2*100/count))
outputFile.close()


list_model = [filename1,filename2]
list_score = [RMSE1, RMSE2]


 ###############################Percent Difference########################################################
plot1=plt.figure(1)
objects = (filename1,filename2)
y_pos = np.arange(len(objects))
performance = [avgDiff1,avgDiff2]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Mean Difference %')
plt.title('Difference % Compared')


 ###############################Mean Variance########################################################

varSum1=0 
varMean1=0 
for value in diffPercentList1:
    varSum1=varSum1+ abs(value-avgDiff1)
varMean1= varSum1/count

varSum2=0 
varMean2=0 
for value in diffPercentList2:
    varSum2=varSum2+ abs(value-avgDiff2)
varMean2= varSum2/count

plot2=plt.figure(2)
objects = (filename1,filename2)
y_pos = np.arange(len(objects))
performance = [varMean1,varMean2]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Mean varance %')
plt.title('Variance from Mean : % ')


############################################################################

outputFile2.write("Mean Difference %,"+str(avgDiff1)+","+str(avgDiff2)+"\n")
outputFile2.write("Mean Variance %,"+str(varMean1)+","+str(varMean2)+"\n")
plt.tight_layout()
plt.show()


pp = PdfPages('../data/performancePlotsTrain.pdf')
pp.savefig(plot1, bbox_inches="tight")
pp.savefig(plot2, bbox_inches="tight")
#pp.savefig(plot3, bbox_inches="tight")
#pp.savefig(plot4, bbox_inches="tight")
#pp.savefig(plot5, bbox_inches="tight")
pp.close()