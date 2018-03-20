dataFiles= open('../webscraper/files/data_domains.csv','r',encoding="utf8")
dataFilesPredicted= open('input/compare.csv','r',encoding="utf8")
dataFiles3= open('input/domain.json','w',encoding="utf8")

directoryVar = {}

for dataFile in dataFilesPredicted:
    dataPredicted=dataFile.split(",")
    directoryVar[dataPredicted[0]] = dataPredicted[1]


counter=-1
strJson="["
for dataFile in dataFiles:
    counter=counter+1
    print("Record Count:"+str(counter))
    data=dataFile.split(",")

    if counter==0:
        continue
 
    try:
        directoryVar[data[0]]
    except:
        continue
        
    if len(strJson)<10:
        strJson=strJson+"\n{\n"
    else:
        strJson=strJson+"\n,{\n"
    
    strJson=strJson+"\"no\":\""+data[0]+"\",\n"
    strJson=strJson+"\"Link\":\""+data[1]+"\",\n"
    strJson=strJson+"\"street\":\""+data[2]+"\",\n"
    strJson=strJson+"\"suburb_add\":\""+data[3]+"\",\n"
    strJson=strJson+"\"post\":\""+data[4]+"\",\n"
    strJson=strJson+"\"sold_year\":\""+data[5]+"\",\n"
    strJson=strJson+"\"status\":\""+data[6]+"\",\n"
    strJson=strJson+"\"property_type\":\""+data[7]+"\",\n"
    strJson=strJson+"\"bed\":\""+data[8]+"\",\n"
    strJson=strJson+"\"bath\":\""+data[9]+"\",\n"
    strJson=strJson+"\"car\":\""+data[10]+"\",\n"
    strJson=strJson+"\"price\":\""+data[11]+"\",\n"
    #strJson=strJson+"\"price_range_low_end\":\""+data[12]+"\",\n"
    #strJson=strJson+"\"price_range_high_end\":\""+data[13]+"\",\n"
    
    try:
        #strJson=strJson+"\"FIELD36\":\""+directoryVar[data[0]]+"\"\n"
        strJson=strJson+"\"price_range_low_end\":\""+str(float(directoryVar[data[0]]) - float(directoryVar[data[0]])/ 12.5 )+"\",\n"
        strJson=strJson+"\"price_range_high_end\":\""+str(float(directoryVar[data[0]]) + float(directoryVar[data[0]])/ 12.5 )+"\",\n"
    except  Exception as e:
        continue
        #print(str(e))
        #strJson=strJson+"\"FIELD36\":\"""\"\n"
        #strJson=strJson+"\"price_range_low_end\":\"\",\n"
        #strJson=strJson+"\"price_range_high_end\":\"\",\n"     
    
    strJson=strJson+"\"External_price_estimator\":\""+data[14]+"\",\n"
    strJson=strJson+"\"Long_term_residence_percent\":\""+data[15]+"\",\n"
    strJson=strJson+"\"Rented_percent\":\""+data[16]+"\",\n"
    strJson=strJson+"\"singles_percent\":\""+data[17]+"\",\n"
    strJson=strJson+"\"garden\":\""+data[18]+"\",\n"
    strJson=strJson+"\"pool\":\""+data[19]+"\",\n"
    strJson=strJson+"\"heating\":\""+data[20]+"\",\n"
    strJson=strJson+"\"cooling\":\""+data[21]+"\",\n"
    strJson=strJson+"\"smlr_pprty_price_1\":\""+data[22]+"\",\n"
    strJson=strJson+"\"smlr_pprty_price_2\":\""+data[23]+"\",\n"
    strJson=strJson+"\"number_of_times_sold\":\""+data[24]+"\",\n"
    strJson=strJson+"\"num_of_times_rented\":\""+data[25]+"\",\n"
    strJson=strJson+"\"last_sold_price\":\""+data[26]+"\",\n"
    strJson=strJson+"\"last_sold_year\":\""+data[27]+"\",\n"
    strJson=strJson+"\"last_rent_price\":\""+data[28]+"\",\n"
    strJson=strJson+"\"last_rent_year\":\""+data[29]+"\",\n"
    strJson=strJson+"\"min_temp\":\""+data[30]+"\",\n"
    strJson=strJson+"\"max_temp\":\""+data[31]+"\",\n"
    strJson=strJson+"\"Rainfall\":\""+data[32]+"\",\n"
    strJson=strJson+"\"distance_cbd\":\""+data[33]+"\",\n"
    strJson=strJson+"\"distance_school\":\""+data[34]+"\",\n"
    strJson=strJson+"\"FIELD36\":\"\"\n"
    
   
    strJson=strJson+"}"
 

strJson=strJson+"]"

dataFiles3.write(strJson)