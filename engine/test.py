dataFiles= open('submission_13.136809166666668_2018-04-12-16-54.csv','r',encoding="utf8")
dataFiles2= open('../data/data_domains5.csv','r',encoding="utf8")
#dataFile.write("no,Link,street,Suburb_add,post,sold_year,status,property_type,bed,bath,car,price,Estimate_low_end,Estimate_high_end,Ext_price_estimate,garden,pool,heating,cooling,neigh_bed_1,neigh_bath_1,neigh_car_1,neigh_price_1,neigh_bed_2,neigh_bath_2,neigh_car_2,neigh_price_2,smlr_pprty_price_1,smlr_pprty_price_2,floor_size,build_size,build_year,num_solds,num_rents,last_sold_price,last_rent_price,last_sold_year,last_rent_year,min_temp,max_temp,Rainfall,near_supermarket,near_school,near_sec_college,near_univ \n")
f= open("guru99.csv","w+")
keyval1 = {
        "iphone" : 2007

    }

keyval2 = {
        "iphone" : 2007

    }
for data in dataFiles:

    data1=data.split(",")
    if data1[0]=="":
        continue
    if len(data1)<2:
        continue
    keyval1[data1[0]]=data1[1]
    

for data2 in dataFiles2:

    data2=data2.split(",")
    if data2[0]=="":
        continue
    if len(data2)<2:
        continue
    keyval2[data2[0]]=data2[48]

f.write("\n")
totaldiff=0
totalperc=0
count=0
for keyval in keyval1:
    if keyval=="Id":
        continue
    diff=0
    percent=0
    try:
        diff= float(keyval2[keyval])-float(keyval1[keyval])
        percent=diff/float(keyval2[keyval])
        percent=percent*100 
        count=count+1
        totaldiff=totaldiff+diff
        totalperc=totalperc+percent
       
    except:
        temp=0
    

    val=str(keyval)+","+str(keyval1[keyval]).replace("\n","")+","+str(keyval2[keyval]).replace("\n","")+","+str(diff)+","+str(percent)

    f.write(val)
    
    f.write("\n")

f.write("Average Difference: ,"+str(totalperc/count))
         
f.close() 
dataFiles.close()
dataFiles2.close()