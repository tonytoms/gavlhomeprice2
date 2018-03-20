import urllib.request
import bs4 as bs 
import datetime
from selenium import webdriver
import webscraper.logger as logger
from datetime import datetime
import re as re
import pandas as pd
import requests
from xml.etree import ElementTree
import traceback
from multiprocessing.pool import ThreadPool
from numpy.core.defchararray import rsplit
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from builtins import int



def yesno(inputVal):
    if inputVal=="" or inputVal is None:
        return ""
    if inputVal.lower()=="no":
        return "0"
    if inputVal.lower()=="yes":
        return "1"
    else:
        return str(inputVal)

def dist(inputVal):
    if inputVal=="" or inputVal is None:
        return ""
    if "m" in inputVal:
        inputVal=inputVal.replace("m","")
        inputret=float(inputVal)/1000  
        return str(inputret)
    else:
        return inputVal



dataFiles= open('../webscraper/files/data_domains.csv','r',encoding="utf8")
dataFiles2= open('input/train.csv','w',encoding="utf8")
dataFiles3= open('input/test.csv','w',encoding="utf8")
#dataFile.write("no,Link,street,Suburb_add,post,sold_year,status,property_type,bed,bath,car,price,Estimate_low_end,Estimate_high_end,Ext_price_estimate,garden,pool,heating,cooling,neigh_bed_1,neigh_bath_1,neigh_car_1,neigh_price_1,neigh_bed_2,neigh_bath_2,neigh_car_2,neigh_price_2,smlr_pprty_price_1,smlr_pprty_price_2,floor_size,build_size,build_year,num_solds,num_rents,last_sold_price,last_rent_price,last_sold_year,last_rent_year,min_temp,max_temp,Rainfall,near_supermarket,near_school,near_sec_college,near_univ \n")





dataHeader=[]
for x in range(0,45):
    dataHeader.append("")

dataHeader[0]="Id"
dataHeader[1]="StreetName"
dataHeader[2]="UnitNumber"
dataHeader[3]="StreetNumber"
dataHeader[4]="SuburbName"
dataHeader[5]="SuburbCode"
dataHeader[6]="SoldDate"
dataHeader[7]="SoldStatus"
dataHeader[8]="PropertyType"
dataHeader[9]="Bed"
dataHeader[10]="Bath"
dataHeader[11]="Garage"
dataHeader[12]="LongTermResidentPercent"
dataHeader[13]="RentedResidentPercent"
dataHeader[14]="SingleResident"
dataHeader[15]="Garden"
dataHeader[16]="Pool"
dataHeader[17]="Heating"
dataHeader[18]="Cooling"
dataHeader[19]="AvgDaysOnMarketSuburb"
dataHeader[20]="PropertySoldInSuburb"
dataHeader[21]="LandSize"
dataHeader[22]="BuildYear"
dataHeader[23]="NumOfTimesSold"
dataHeader[24]="NumOfTimesRented"
dataHeader[25]="LastSoldPrice"
dataHeader[26]="LastRentedPrice"
dataHeader[27]="LastSoldYear"
dataHeader[28]="LastRentedYear"
dataHeader[29]="MinTemp"
dataHeader[30]="MaxTemp"
dataHeader[31]="AvgRainfall"
dataHeader[32]="DistanceCBD"
dataHeader[33]="DistanceRailwayStation"
dataHeader[34]="DistanceBusStation"
dataHeader[35]="DistanceShoppingMall"
dataHeader[36]="DistanceSchool"
dataHeader[37]="PriceLowEnd"
dataHeader[38]="PriceHighEnd"
dataHeader[39]="ExternalPricePrediction"
dataHeader[40]="SimilarPropertyPrice1"
dataHeader[41]="SimilarPropertyPrice2"
dataHeader[42]="Price"


for n in range(0,len(dataHeader)):

    dataFiles2.write(dataHeader[n])
    dataFiles2.write(",")

    dataFiles3.write(dataHeader[n])
    dataFiles3.write(",")

dataFiles2.write("\n")
dataFiles3.write("\n")

data=[]
for i in range(0,43):
    data.append("")

i=-1
for dataFile in dataFiles:
    dataFile=dataFile.replace("\n","")

    i=i+1
     
    if i==0:
        continue 
    data1=dataFile.split(",")
    
#  ID, STREET NAME, UNIT NUMBER,STREET NUMBER    
    data[0]=data1[0] 
    streets=data1[2].split(" ",1) 
    data[1]= streets[1]  
    unitNum=""
    streetNum=""
    if "/" in streets[0]:
        streetsNum=streets[0].split("/") 
        unitNum=re.findall('\d+', streetsNum[0] ) 
        streetNum=re.findall('\d+', streetsNum[1] )
    else:
        unitNum=""
        streetNum=re.findall('\d+', streets[0] )
        
    data[2]=''.join(unitNum)
    data[3]=''.join(streetNum ) 
    
    
    data[4]=data1[3]   # SUBURB NAME
    data[5]=data1[4]   #SUBURB CODE

    #date to int
    
    soldDateList=data1[5].split("/")
    d0 = datetime(2000, 1, 1)
    d1 = datetime(int(soldDateList[2]), int(soldDateList[1]), int(soldDateList[0]))# 2018,6,26
    delta = d1 - d0
    data[6]=( delta.days)    

    soldDateList=data1[5].split("/")
    d0 = datetime(2000, 1, 1)
    d1 = datetime(2018,3,2)# 2018,6,26
    delta = d1 - d0
    nowDate=( delta.days)  
    
    data[7]=data1[6]  # SOLD STATUS
    data[8]=data1[7].replace("Apartment / Unit / Flat","AppartmentFlat")  # Property Type
    data[9]=data1[8]  # Bed
    data[10]=data1[9]  # Bath
    if data1[10]=="" or data1[10] is None:
        data1[10]="0"
    data[11]=data1[10]  # Garage
    
    
    
    

    data[12]=data1[15]  # Long Term Resident Percent
    data[13]=data1[16]  # Rented Resident Percent
    data[14]=data1[17]  # Single Resident Percent

    data[15]=yesno(data1[18])  # Garden
    data[16]=yesno(data1[19])  # Pool
    data[17]=yesno(data1[20])  # Heating
    data[18]=yesno(data1[21])  # Cooling
   
    
    data[19]=yesno(data1[24])  # AvgDaysOnMarketSuburb
    data[20]=yesno(data1[41])  # PropertySoldInSuburb
   
    data[21]=data1[25]  # LandSize

    if data1[26] =="" or data1[26] is None:
        data[22]=""  # BuildYear
    else:

        d0 = datetime(2000, 1, 1)
        d1 = datetime(int(data1[26]), 1,1)# 2018,6,26
        delta = d1 - d0
        data[22]=( delta.days)  #BuildYear
   
    data[23]=data1[27]  # NumOfTimesSold
    data[24]=data1[28]  # NumOfTimesRented
    data[25]=data1[29]  # LastSoldPrice
    data[26]=data1[31]  # LastRentedPrice

    if data1[30] =="" or data1[30] is None:
        data[27]=""  # LastSoldYear
    else:

        d0 = datetime(2000, 1, 1)
        d1 = datetime(int(data1[30]), 1,1)# 2018,6,26
        delta = d1 - d0
        data[27]=( delta.days)  #LastSoldYear
        
        
    if data1[32] =="" or data1[32] is None:
        data[28]=""  # LastRentedYear
    else:

        d0 = datetime(2000, 1, 1)
        d1 = datetime(int(data1[32]), 1,1)# 2018,6,26
        delta = d1 - d0
        data[28]=( delta.days)  #LastRentedYear   

    data[29]=data1[33]  # Min Temp
    data[30]=data1[34]  # Max Temp
    data[31]=data1[35]  # Avg Rainfall

    data[32]=dist(data1[36]) # distance CBD
    data[33]=dist(data1[37]) # distance train station
    data[34]=dist(data1[38] ) # distance bus station
    data[35]=dist(data1[39]) # distance shopping mall
    data[36]=dist(data1[40]) # distance school
  

    
    if data1[12]=="0":
        data1[12]=""
    if data1[13]=="0":
        data1[13]=""
    if data1[14]=="0":
        data1[14]=""
    if data1[22]=="0":
        data1[22]=""
    if data1[23]=="0":
        data1[23]=""
    if data1[11]=="0":
        data1[11]=""
    data[37]=data1[12]  # LowerPrice
    data[38]=data1[13]  # HigherPrice
    data[39]=data1[14]  # ExternalPrice
    data[40]=data1[22]  # Similar Pprty Price 1
    data[41]=data1[23]  # Sim ilar pprty Price 2
    data[42]=data1[11]  # Price
    
    nullcounter=0
    if data[42]=="" or data[42] is None:
        continue
    try:
        float(data[42])
    except:
        continue
    '''
    for n in range(0,len(data)):

        if data[n]=="" or data[n] is None:
            nullcounter=nullcounter+1
    if nullcounter>3:
        continue
    '''        

    for n in range(0,len(data)):
        if data[n] is None or data[n]=="":
            data[n]="NA"
        dataFiles2.write(str(data[n]))
        dataFiles2.write(",")
    dataFiles2.write("\n")
   
    dataFiles2.flush()
    data[6]=nowDate
    for n in range(0,len(data)):
        if data[n] is None or data[n]=="":
            data[n]="NA"
        dataFiles3.write(str(data[n]))
        dataFiles3.write(",")
    dataFiles3.write("\n")
   
    dataFiles3.flush()