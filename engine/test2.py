'''
Created on May 16, 2018

@author: Default
'''
import matplotlib.pyplot as plt

from datetime import datetime

dataFiles= open('../data/lasso_ridge_xgb_v2.csv','r',encoding="utf8")
dataFiles2= open('../data/Compare1.csv','w',encoding="utf8")


datas=[]
count=0
id="X"
#12 beds
#24 landsize
#25 year build
#40 price range min
totData=[]
for dataStr in dataFiles:
 
    totData.append(dataStr)
    

yaxis=[]
xaxis=[] 

'''
year1=0
year2=0
year3=0
year4=0
year5=0
year6=0
year7=0
year0=0
year1N=0
year2M=0

year1C=0
year2C=0
year3C=0
year4C=0
year5C=0
year6C=0
year7C=0
year0C=0
year1NC=0
year2MC=0

'''

year1Arr=[]
year2Arr=[]
year3Arr=[]
year4Arr=[]
year5Arr=[]
year6Arr=[]
year7Arr=[]
year0Arr=[]
year1NArr=[]
year2MArr=[]

dataMean=[]
while count<len(totData):
    
    outData=[]
    print(count)
    if count==1332:
        temp=2
    data1=totData[count].split(",")
    count=count+1

    data2=totData[count].split(",")

    outData.append(data1[2].replace("\n",""))
    #print(data1[2])

    diff=float(data2[0])-float(data1[0])
    outData.append(str(diff))
    
    date_format = "%Y%m%d"
    a = datetime.strptime(data1[1], date_format)
    b = datetime.strptime(data2[1], date_format)
    daysC =float( (b - a).days/365)
    xaxis.append(diff)
    yaxis.append(daysC)
    outData.append(str(daysC))
    
    if daysC<-2:

        year2MArr.append(diff)
    elif daysC<-1:

        year1NArr.append(diff)

    elif daysC<0:
        year0Arr.append(diff)
    elif daysC<1:
        year1Arr.append(diff)
    elif daysC<2:
        year2Arr.append(diff)
    elif daysC<3:
        year3Arr.append(diff)
    elif daysC<4:
        year4Arr.append(diff)
    elif daysC<5:
        year5Arr.append(diff)
    elif daysC<6:
        year6Arr.append(diff)
    else:
        year7Arr.append(diff)


    dataStrMod = ','.join(outData)
    dataStrMod=dataStrMod+"\n"
    dataFiles2.write(dataStrMod)
    
    count=count+1


year1Avg=0
year2Avg=0
year3Avg=0
year4Avg=0
year5Avg=0
year6Avg=0
year7Avg=0
year0Avg=0
year1NAvg=0
year2MAvg=0


if len(year1Arr)>0:
    year1Avg=sum(year1Arr)/len(year1Arr)
else:
    year1Avg=0

if len(year2Arr)>0:
    year2Avg=sum(year2Arr)/len(year2Arr)
else:
    year2Avg=0
if len(year3Arr)>0:
    year3Avg=sum(year3Arr)/len(year3Arr)
else:
    year3Avg=0
if len(year4Arr)>0:
    year4Avg=sum(year4Arr)/len(year4Arr)
else:
    year4Avg=0
if len(year5Arr)>0:
    year5Avg=sum(year5Arr)/len(year5Arr)
else:
    year5Avg=0
if len(year6Arr)>0:
    year6Avg=sum(year6Arr)/len(year6Arr)
else:
    year6Avg=0
if len(year7Arr)>0:
    year7Avg=sum(year7Arr)/len(year7Arr)
else:
    year7Avg=0
if len(year0Arr)>0:
    year0Avg=sum(year0Arr)/len(year0Arr)
else:
    year0Avg=0  
    
if len(year1NArr)>0:
    year1NAvg=sum(year1NArr)/len(year1NArr)
else:
    year1NAvg=0 
if len(year2MArr)>0:
    year2MAvg=sum(year2MArr)/len(year2MArr)
else:
    year2MAvg=0 
  

year1Mean=year1Arr
year2Mean=year2Arr
year3Mean=year3Arr
year4Mean=year4Arr
year5Mean=year5Arr
year6Mean=year6Arr
year7Mean=year7Arr
year0Mean=year0Arr
year1NMean=year1NArr
year2MMean=year2MArr

year1Mean[:] = [x - year1Avg for x in year1Mean]
year2Mean[:] = [x - year2Avg for x in year2Mean]
year3Mean[:] = [x - year3Avg for x in year3Mean]
year4Mean[:] = [x - year4Avg for x in year4Mean]
year5Mean[:] = [x - year5Avg for x in year5Mean]
year6Mean[:] = [x - year6Avg for x in year6Mean]
year7Mean[:] = [x - year7Avg for x in year7Mean]
year0Mean[:] = [x - year0Avg for x in year0Mean]
year1NMean[:] = [x - year1NAvg for x in year1NMean]
year2MMean[:] = [x - year2MAvg for x in year2MMean]


if len(year1Mean)>0:
    year1MeanAvg=sum(year1Mean)/len(year1Mean)
else:
    year1MeanAvg=0

if len(year2Mean)>0:
    year2MeanAvg=sum(year2Mean)/len(year2Mean)
else:
    year2MeanAvg=0
if len(year3Mean)>0:
    year3MeanAvg=sum(year3Mean)/len(year3Mean)
else:
    year3MeanAvg=0
if len(year4Mean)>0:
    year4MeanAvg=sum(year4Mean)/len(year4Mean)
else:
    year4MeanAvg=0
if len(year5Mean)>0:
    year5MeanAvg=sum(year5Mean)/len(year5Mean)
else:
    year5MeanAvg=0
if len(year6Mean)>0:
    year6MeanAvg=sum(year6Mean)/len(year6Mean)
else:
    year6MeanAvg=0
if len(year7Mean)>0:
    year7MeanAvg=sum(year7Mean)/len(year7Mean)
else:
    year7MeanAvg=0
if len(year0Mean)>0:
    year0MeanAvg=sum(year0Mean)/len(year0Mean)
else:
    year0MeanAvg=0  
    
if len(year1NMean)>0:
    year1NMeanAvg=sum(year1NMean)/len(year1NMean)
else:
    year1NMeanAvg=0 
if len(year2MMean)>0:
    year2MMeanAvg=sum(year2MMean)/len(year2MMean)
else:
    year2MMeanAvg=0 
    

dataFiles2.write("\n"+",Average"+str(year1MeanAvg)+","+str(year2MeanAvg)+","+str(year3MeanAvg)+","+str(year4MeanAvg)+","+str(year5MeanAvg)+","+","+str(year6MeanAvg)+","+","+str(year7MeanAvg)+","+","+str(year0MeanAvg)+","+","+str(year1NMeanAvg)+","+","+str(year2MMeanAvg)+",")
dataFiles2.write(dataStrMod)
dataFiles2.write(dataStrMod)
dataFiles2.write(dataStrMod)
dataFiles2.write(dataStrMod)
dataFiles2.write(dataStrMod)
dataFiles2.write(dataStrMod)

plt.plot(xaxis, yaxis, 'ro')
#plt.axis([0, 6, 0, 20])
plt.show()
