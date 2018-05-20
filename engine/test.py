'''
Created on May 16, 2018

@author: Default
'''


dataFiles= open('../data/test.csv','r',encoding="utf8")
dataFiles2= open('../data/test2.csv','w',encoding="utf8")


datas=[]
count=-1
id="X"
#12 beds
#24 landsize
#25 year build
#40 price range min
for dataStr in dataFiles:
 
    count=count+1

    
    data=dataStr.split(",")

    dataStrMod = ','.join(data)
    data[5]="20180505"
    data[0]=id+data[0]
    dataStrMod2 = ','.join(data)
    dataFiles2.write(dataStrMod)
    dataFiles2.write(dataStrMod2)
        