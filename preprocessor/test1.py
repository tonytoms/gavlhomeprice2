
dataFiles= open('../webscraper/files/data_domains.csv','r',encoding="utf8")

pre=""
mylist=[]
for dataFile in dataFiles:
    data=dataFile.split(",")
    if data[7] not in mylist:
        mylist.append(data[7])
        
for my in mylist:
    print(my)