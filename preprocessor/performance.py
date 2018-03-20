from builtins import str
dataFiles= open('input/compare.csv','r',encoding="utf8")
dataFiles2= open('input/compareF.csv','w',encoding="utf8")
total=0
count=0
datalist=[]
slab1=0#0-3 % difference
slab2=0#3-6 % difference
slab3=0#6-9 % difference
slab4=0#9-12 % difference
slab5=0#12+ % difference

Nslab1=0# - 0-3 % difference
Nslab2=0# - 3-6 % difference
Nslab3=0# - 6-9 % difference
Nslab4=0# - 9-12 % difference
Nslab5=0# - 12+ % difference

dataFiles2.write("Estimated ID,Estimated Price,Property ID,Actual Property Price,Difference %, Deviation % \n")

for dataFile in dataFiles:
    diff=0
    count=count+1
    data=dataFile.split(",")
    diff=(float(data[3])-float(data[1]))/float(data[3])
    diff=diff*100
    
    if diff>=0 and diff< 3:
        slab1=slab1+1
    elif diff>=3 and diff< 6:
        slab2=slab2+1
    elif diff>=6 and diff< 9:
        slab3=slab3+1
    elif diff>=9 and diff< 12:
        slab4=slab4+1        
    elif diff>-3 and diff<= 0:
        Nslab1=Nslab1+1
    elif diff>-6 and diff<= -3:
        Nslab2=Nslab2+1
    elif diff>-9 and diff<= -6:
        Nslab3=Nslab3+1
    elif diff>-12 and diff<= -9:
        Nslab4=Nslab4+1 
    elif diff>=12:
        slab5=slab5+1 
    elif diff< -12:
        Nslab5=Nslab5+1
                    
    data[4]=str(diff)
    total=total+diff
    
    datalist.append(data)
    
mean=total/count
totalmean=0
for dataListItem in datalist:    
    
    data2=dataListItem
    data2[5]=float(data2[4])-mean
    totalmean=totalmean+data2[5]
    for datum in dataListItem:
        dataFiles2.write(str(datum))
        dataFiles2.write(",")
    dataFiles2.write("\n")

dataFiles2.write("\n\n MEAN Difference="+str(total/count ))
dataFiles2.write("\n Average DEVIATION="+str(totalmean/count ))
dataFiles2.write("\n ,0  -  3 % difference , records="+str(slab1)+" ,"+str(slab1*100/count)+" %")
dataFiles2.write("\n ,3  -  6 % difference , records="+str(slab2)+" ,"+str(slab2*100/count)+" %")
dataFiles2.write("\n ,6  -  9 % difference , records="+str(slab3)+" ,"+str(slab3*100/count)+" %")
dataFiles2.write("\n ,9  -  12 % difference , records="+str(slab4)+" ,"+str(slab4*100/count)+" %")
dataFiles2.write("\n ,12 % +  difference , records="+str(slab5)+" ,"+str(slab5*100/count)+" %")

dataFiles2.write("\n ,,-0  -  -3 % difference , records="+str(Nslab1)+" ,"+str(Nslab1*100/count)+" %")
dataFiles2.write("\n ,,-3  -  -6 % difference , records="+str(Nslab2)+" ,"+str(Nslab2*100/count)+" %")
dataFiles2.write("\n ,,-6  -  -9 % difference , records="+str(Nslab3)+" ,"+str(Nslab3*100/count)+" %")
dataFiles2.write("\n ,,-9  -  -12 % difference , records="+str(Nslab4)+" ,"+str(Nslab4*100/count)+" %")
dataFiles2.write("\n ,,-12 % +  difference , records="+str(Nslab5)+" ,"+str(Nslab5*100/count)+" %")

