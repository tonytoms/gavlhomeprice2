
dataOut1= open('../data/DNNRegOut.csv','w',encoding="utf8")
dataOut2= open('../data/DNNRegShallowOut.csv','w',encoding="utf8")
filename1="DNNRegOut"
filename2="DNNRegShallowOut"

for i in range(1,12):
    if i==6 or i==11:
        continue
    datafile= open('../data/'+filename1+str(i)+'.csv','r',encoding="utf8")
    check=0
    for data in datafile:
        if check==0:
            check=check+1
            if i==0:
                dataOut1.write(data)
         
            continue
        dataOut1.write(data)  
        
        
for i in range(1,12):
    if i==6 or i==11:
        continue
    datafile= open('../data/'+filename2+str(i)+'.csv','r',encoding="utf8")
    check=0
    for data in datafile:
        if check==0:
            check=check+1
            if i==0:
                dataOut2.write(data)
         
            continue
        dataOut2.write(data)     
