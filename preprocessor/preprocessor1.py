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
from datetime import datetime
from datetime import date


import holidays
import calendar



def shortDay(inputVal):
    if inputVal=="" or inputVal is None:
        return ""
    if inputVal.lower()=="sunday":
        return "SDay"
    if inputVal.lower()=="monday":
        return "MDay"
    if inputVal.lower()=="tuesday":
        return "TuDay"
    if inputVal.lower()=="wednesday":
        return "WDay"
    if inputVal.lower()=="thursday":
        return "ThDay"
    if inputVal.lower()=="friday":
        return "FDay"
    if inputVal.lower()=="saturday":
        return "SDay"
    
    
def yesno(inputVal):
    if inputVal=="" or inputVal is None:
        return ""
    if inputVal.lower()=="no":
        return "N"
    if inputVal.lower()=="yes":
        return "Y"
    else:
        return str(inputVal)
    
    slop
def slop(inputVal):
    retval=""
    if inputVal=="Nearly Level":
        retval="NrlyLvl"
    elif inputVal=="Gently Inclined":
        retval="GntlyInclnd"
    elif inputVal=="Moderately Inclined":
        retval="ModInclnd"
    elif inputVal=="Steep":
        retval="Steep"
    else:
        retval=inputVal  
    return retval    
def propertyTypeFn(inputVal):
    retval=""
    if inputVal=="Apartment / Unit / Flat":
        retval="AptFlat"
    elif inputVal=="House":
        retval="Hse"
    elif inputVal=="Vacant land":
        retval="VacLnd"
    elif inputVal=="Apartment / Unit / Flat|Townhouse":
        retval="AptTwHse"
    elif inputVal=="Townhouse":
        retval="TwnHse"
    elif inputVal=="Block of Units":
        retval="BlkUnt"
    elif inputVal=="Apartment / Unit / Flat|Villa":
        retval="AptVilla"
    elif inputVal=="Apartment / Unit / Flat|House":
        retval="AptHse"
    elif inputVal=="Villa":
        retval="Villa"
    elif inputVal=="Development Site":
        retval="DevSte"
    elif inputVal=="Semi-Detached":
        retval="SmiDtchd"
    elif inputVal=="House|Townhouse":
        retval="HseTwnHse"
    elif inputVal=="House|Vacant land":
        retval="HseVacLnd"
    elif inputVal=="New House & Land":
        retval="NewHseLnd"
    elif inputVal=="House|Vacant land":
        retval="HseVacLnd"
    elif inputVal=="Apartment / Unit / Flat|House|Townhouse":
        retval="AptHseTwnHse"  
    elif inputVal=="Villa|House":
        retval="VllaHse" 
    elif inputVal=="Townhouse|House":
        retval="TwnHseHse" 
    elif inputVal=="House|Townhouse|Villa":
        retval="HseTwnHseVilla" 
    elif inputVal=="Apartment / Unit / Flat|House|Townhouse|Villa":
        retval="AptFlatTwnHseVilla" 
    elif inputVal=="Apartment / Unit / Flat|Townhouse|Villa":
        retval="AptFlatTwnHseVilla" 
    elif inputVal=="Block of Units|House":
        retval="BlkUnitHse" 
    elif inputVal=="House|Townhouse|Vacant land":
        retval="HseTwnHseVacLnd" 
    elif inputVal=="Block of Units|Townhouse":
        retval="BlkUnitTwnHse"           
    elif inputVal=="Apartment / Unit / Flat|Studio":
        retval="AptFlatStdio" 
    elif inputVal=="Apartment / Unit / Flat|House|Semi-Detached|Townhouse":
        retval="AptFltSmDetTwnHse" 
    elif inputVal=="New Apartments / Off the Plan|House":
        retval="NewAptOffPlanHse" 
    elif inputVal=="House|Villa":
        retval="HseVilla" 
    elif inputVal=="House|Vacant land|Villa":
        retval="HseVacLndVilla" 
    elif inputVal=="Apartment / Unit / Flat|House|Villa":
        retval="AptFltHseVilla" 
    elif inputVal=="New Apartments / Off the Plan":
        retval="NewAptOffpln" 
    elif inputVal=="New Land|Development Site":
        retval="NwLndDevSite" 
    elif inputVal=="Studio":
        retval="Stdo" 
    elif inputVal=="Terrace":
        retval="Trce"   
    else:
        retval=inputVal  
    return retval    

def dist(inputVal):
    if inputVal=="" or inputVal is None:
        return ""
    if "m" in inputVal:
        inputVal=inputVal.replace("m","")
        inputret=float(inputVal)/1000  
        return str(inputret)
    else:
        return inputVal



dataFiles= open('../data/data_domains4.csv','r',encoding="utf8")
dataFiles2= open('../data/data_domains5.csv','w',encoding="utf8")
#dataFile.write("no,Link,street,Suburb_add,post,sold_year,status,property_type,bed,bath,car,price,Estimate_low_end,Estimate_high_end,Ext_price_estimate,garden,pool,heating,cooling,neigh_bed_1,neigh_bath_1,neigh_car_1,neigh_price_1,neigh_bed_2,neigh_bath_2,neigh_car_2,neigh_price_2,smlr_pprty_price_1,smlr_pprty_price_2,floor_size,build_size,build_year,num_solds,num_rents,last_sold_price,last_rent_price,last_sold_year,last_rent_year,min_temp,max_temp,Rainfall,near_supermarket,near_school,near_sec_college,near_univ \n")





dataHeader=[]
for x in range(0,49):
    dataHeader.append("")

dataHeader[0]="Id"
dataHeader[1]="StreetName"
dataHeader[2]="UnitNumber"
dataHeader[3]="StreetNumber"
dataHeader[4]="SuburbName"
dataHeader[5]="SuburbCode"
dataHeader[6]="SoldDate"
dataHeader[7]="DayOfWeek"
dataHeader[8]="DayType"
dataHeader[9]="HolidayStatus"
dataHeader[10]="SoldStatus"
dataHeader[11]="PropertyType"
dataHeader[12]="Bed"
dataHeader[13]="Bath"
dataHeader[14]="Garage"
dataHeader[15]="LongTermResidentPercent"
dataHeader[16]="RentedResidentPercent"
dataHeader[17]="SingleResident"
dataHeader[18]="Garden"
dataHeader[19]="Pool"
dataHeader[20]="Heating"
dataHeader[21]="Cooling"
dataHeader[22]="AvgDaysOnMarketSuburb"
dataHeader[23]="PropertySoldInSuburb"
dataHeader[24]="LandSize"
dataHeader[25]="BuildYear"
dataHeader[26]="NumOfTimesSold"
dataHeader[27]="NumOfTimesRented"
dataHeader[28]="LastSoldPrice"
dataHeader[29]="LastRentedPrice"
dataHeader[30]="LastSoldYear"
dataHeader[31]="LastRentedYear"
dataHeader[32]="MinTemp"
dataHeader[33]="MaxTemp"
dataHeader[34]="AvgRainfall"
dataHeader[35]="DistanceCBD"
dataHeader[36]="DistanceRailwayStation"
dataHeader[37]="DistanceBusStation"
dataHeader[38]="DistanceShoppingMall"
dataHeader[39]="DistanceSchool"
dataHeader[40]="PriceLowEnd"
dataHeader[41]="PriceHighEnd"
dataHeader[42]="LandArea"
dataHeader[43]="Frontage"
dataHeader[44]="Slop"
dataHeader[45]="ExternalPricePrediction"
dataHeader[46]="SimilarPropertyPrice1"
dataHeader[47]="SimilarPropertyPrice2"
dataHeader[48]="Price"


for n in range(0,len(dataHeader)):

    dataFiles2.write(dataHeader[n])
    dataFiles2.write(",")


dataFiles2.write("\n")

data=[]
for i in range(0,49):
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
        
    data[2]=''.join( unitNum)
    if data[2]=="" :
            placeholder=""
    else:
        data[2]=int(data[2]) % 10
    data[3]=''.join(streetNum ) 
    if data[3]=="" :
        placeholder=""
    else:
        data[3]=int(data[3]) % 10
    
    
    data[4]=data1[3]   # SUBURB NAME
    data[5]=data1[4]   #SUBURB CODE

    #dates
    
    
    soldDateList=data1[5].split("/")    
    datetime_object = datetime.strptime(data1[5] , '%d/%m/%Y')
    
    
    day=(calendar.day_name[datetime_object.weekday()])
    timeofWeek=''
    if(day=="Sunday" or day=="Saturday"):
        timeofWeek="WkEnd"
    else:
        timeofWeek="WkDay" 
    
    
    aus_holidays = holidays.AU(prov = 'VIC')  # or holidays.US(), or holidays.CountryHoliday('US')
    dayType=""
    if date(int(soldDateList[2]), int(soldDateList[1]), int(soldDateList[0])) in aus_holidays:
        dayType="HliDy"
    else:
        dayType="WrkngDy"

    if len(soldDateList[1])<2:
        soldDateList[1]="0"+soldDateList[1]
    if len(soldDateList[0])<2:
        soldDateList[0]="0"+soldDateList[0]
    data[6]=soldDateList[2]+soldDateList[1]+soldDateList[0]
    data[7]=timeofWeek
    data[8]=shortDay(day)   
    data[9]=dayType   
    

    data[10]=data1[6]  # SOLD STATUS
    data[11]=propertyTypeFn(data1[7])


  
    data[12]=data1[8]  # Bed
    data[13]=data1[9]  # Bath
    if data1[10]=="" or data1[10] is None:
        data1[10]="0"
    data[14]=data1[10]  # Garage
    
    
    
    

    data[15]=data1[15]  # Long Term Resident Percent
    data[16]=data1[16]  # Rented Resident Percent
    data[17]=data1[17]  # Single Resident Percent

    data[18]=yesno(data1[18])  # Garden
    data[19]=yesno(data1[19])  # Pool
    data[20]=yesno(data1[20])  # Heating
    data[21]=yesno(data1[21])  # Cooling
   
    
    data[22]=(data1[24])  # AvgDaysOnMarketSuburb
    data[23]=(data1[41])  # PropertySoldInSuburb
   
    data[24]=data1[25]  # LandSize

    if data1[26] =="" or data1[26] is None:
        data[25]=""  # BuildYear
    else:
        
        data[25]=data1[26]  

   
    data[26]=data1[27]  # NumOfTimesSold
    data[27]=data1[28]  # NumOfTimesRented
    data[28]=data1[29]  # LastSoldPrice
    data[29]=data1[31]  # LastRentedPrice

    if data1[30] =="" or data1[30] is None:
        data[30]=""  # LastSoldYear
    else:

        data[30]=data1[30] 

        
    if data1[32] =="" or data1[32] is None:
        data[31]=""  # LastRentedYear
    else:

        data[31]=data1[32]   


    data[32]=data1[33]  # Min Temp
    data[33]=data1[34]  # Max Temp
    data[34]=data1[35]  # Avg Rainfall

    data[35]=dist(data1[36]) # distance CBD
    data[36]=dist(data1[37]) # distance train station
    data[37]=dist(data1[38] ) # distance bus station
    data[38]=dist(data1[39]) # distance shopping mall
    data[39]=dist(data1[40]) # distance school
  

    
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
    data[40]=data1[12]  # LowerPrice
    data[41]=data1[13]  # HigherPrice
    data[45]=data1[14]  # ExternalPrice
    data[46]=data1[22]  # Similar Pprty Price 1
    data[47]=data1[23]  # Sim ilar pprty Price 2
    data[48]=data1[11]  # Price
 
 
 
    landarea=data1[42].split(" ")
    data[42]=landarea[0] # LandArea
    frontage=data1[43].split(" ")
    data[43]=frontage[0]  # Frontage
    data[44]=slop(data1[44])  # Slop

   
    nullcounter=0
    if data[48]=="" or data[48] is None:
        continue
    try:
        float(data[48])
    except:
        continue
       

    for n in range(0,len(data)):
        if data[n] is None or data[n]=="":
            data[n]="NA"
        dataFiles2.write(str(data[n]))
        dataFiles2.write(",")
    dataFiles2.write("\n")
   
    dataFiles2.flush()
   