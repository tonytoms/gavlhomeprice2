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
'''
Created on Dec 25, 2017

@author: Tony Toms
'''

def data_writter(dataFile,data):
    for dataCount in range(0,43):
        if dataCount==0 :
            continue;
        dataFile.write(str(data[dataCount])+",")
    dataFile.write("\n")
    dataFile.flush()
    return True
def driver_loader(urlHist,drivr):
    drivr.get(urlHist)
    return True

#get Weather data
def getWeatherData(filenamePointer,year,month,day):
    
    retVal=""
    df1 = filenamePointer
    
    keywordsYear=[year]
    keywordsMonth=[month]
    keywordsDay=[day]
    df2 = df1[df1["Year"].isin(keywordsYear) & df1["Month"].isin(keywordsMonth) & df1["Day"].isin(keywordsDay)] 
    # write the data back to a csv file 
    
    df2str=str(df2.values)
    df2str=df2str.replace("[", "")
    df2str=df2str.replace("]", "")
    df2List=df2str.split(" ")
    
    if df2List is not None and len(df2List)>6:
        retVal=df2List[5]

    return retVal

#format the price , remove commas a
def price_formatter(inp):
    if inp is None:
        return "0"
    if inp== "":
        return "0"
    '''
    inp=inp.replace("\n", "").replace('\r', '')  
    inp=inp.replace("k","000")
    inp=inp.replace("K","000")
    inp=inp.replace("m","0000")
    inp=inp.replace("M","0000")
    '''
    #
    inp=inp.replace(",", "")   
    inp=inp.replace("$", "")
    inp=inp.replace(" ", "")   

    inp=inp.replace("\n", " ").replace('\r', ' ')  
    inp=inp.strip()
    #inpArr=re.split('^[0-9mMkK] ',inp)
    inpArr=re.split(' ',inp)
    inp=inpArr[0]
    try:
        if "m" in inp or "M" in inp:
            inp=inp.replace("m","")
            inp=inp.replace("M","")
            inp=str( float(inp  ) * 1000000)
        if "k" in inp or "K" in inp:
            inp=inp.replace("k","")
            inp=inp.replace("K","")
            inp=str( float(inp  ) * 1000)
    except :
        return ""

    

    
    outp= inp
    return outp
  
def dateFormatter(dateAuc):
    datestr=""
    try:
        dateAuc=dateAuc.strip()
        dateAuc=dateAuc.replace("rd "," ")
        dateAuc=dateAuc.replace("th "," ")
        dateAuc=dateAuc.replace("st "," ")
        dateAuc=datetime.strptime(dateAuc, '%d %B %Y')
        datestr=dateAuc.strftime('%d/%m/%Y')
        
        
        
    except ValueError as e:
        #print(" DATE ERRORR:::"+e)
        datestr=""
    return datestr

#READ links_realestate.csv' and extract the links and returns a list of links
def getLinks(arlist):
    links=[]
    myfile= open('../data/links_domain.csv','r')
    lines=myfile.readlines()
    count=0
    actual_count=0
    for line in lines:
        actual_count=actual_count+1
        link=line.split(',')
        if len(link)>2 and actual_count>=int(arlist[1]):
            links.append( link[1])
            count=count+1
    arlist[0]=count
    return links



########## PROGRAM STARTS                      ###################################################################################################
data=[]

#create column
for i in range(0,43):
    data.append("")
        
data[1]="1-no"
data[2]="2-Link"
data[3]="3-street"
data[4]="4-Suburb_add"
data[5]="5-post"
data[6]="6-sold_year"
data[7]="7-status"
data[8]="8-property_type"
data[9]="9-bed"
data[10]="10-bath"
data[11]="11-car"
data[12]="12-price"
data[13]="13-price_range_low_end"
data[14]="14-price_range_high_end"
data[15]="15-External_price_estimator"
data[16]="16-Long_term_residence_percent"
data[17]="17-Rented_percent"
data[18]="18-singles_percent"
data[19]="19-garden"
data[20]="20-pool"
data[21]="21-heating"
data[22]="22-cooling"
data[23]="23-smlr_pprty_price_1"
data[24]="24-smlr_pprty_price_2"
data[25]="25-average_days_in_market_Suburb"
data[26]="26-land_size"
data[27]="27-build_year"
data[28]="28-number_of_times_sold"
data[29]="29-num_of_times_rented"
data[30]="30-last_sold_price"
data[31]="31-last_sold_year"
data[32]="32-last_rent_price"
data[33]="33-last_rent_year"
data[34]="34-min_temp"
data[35]="35-max_temp"
data[36]="36-Rainfall"
data[37]="37-distance_cbd"
data[38]="38-distance_train_station"
data[39]="39-distance_bus_school"
data[40]="40-distance_shopping_mall"
data[41]="41-distance_school"
data[42]="42-properties_sold_suburb"

dataFile= open('../data/data_domains.csv','a',encoding="utf8")
LinksFile= open('../data/data_links.csv','a',encoding="utf8")
#dataFile.write("no,Link,street,Suburb_add,post,sold_year,status,property_type,bed,bath,car,price,Estimate_low_end,Estimate_high_end,Ext_price_estimate,garden,pool,heating,cooling,neigh_bed_1,neigh_bath_1,neigh_car_1,neigh_price_1,neigh_bed_2,neigh_bath_2,neigh_car_2,neigh_price_2,smlr_pprty_price_1,smlr_pprty_price_2,floor_size,build_size,build_year,num_solds,num_rents,last_sold_price,last_rent_price,last_sold_year,last_rent_year,min_temp,max_temp,Rainfall,near_supermarket,near_school,near_sec_college,near_univ \n")
data_writter(dataFile, data)
uclient=urllib.request.urlopen('https://www.google.com.au/')
#drivr=webdriver.Firefox(executable_path=r'D:\workspace\GAVL\geckodriver-v0.19.1-win32\geckodriver.exe')
#drivr=webdriver.Chrome("D:\workspace\GAVL\chromedriver_win32\chromedriver.exe")


chromeOptions = webdriver.ChromeOptions()
prefs = {"profile.managed_default_content_settings.images":2}
chromeOptions.add_experimental_option("prefs",prefs)
#chromeOptions.add_argument("--headless")  
drivr = webdriver.Chrome("D:\workspace\GAVL\chromedriver_win32\chromedriver.exe",chrome_options=chromeOptions)
                          

loggerFile=logger.init()
logger.fileWriteln("----------------------------------------------------------------------",loggerFile)
logger.fileWriteln("STEP 3 of 5 : Data Collection DOMAIN.COM.AU----  "+str(datetime.now()),loggerFile)
print("STEP 4 of 5 : Data Collection DOMAIN.COM.AU----------")
logger.fileWriteln("----------------------------------------------------------------------",loggerFile)

start = input("ENter the STarting Listing Number(Enter 1 to begin from start):")
logger.fileWriteln("   Starting at Listing No:"+str(start),loggerFile)


logger.fileWrite("  Retriving Links...",loggerFile)
arlist=[0,start]
links=getLinks(arlist)

logger.fileWriteln("  : "+str(arlist[0])+" Links Found",loggerFile)

main_count=int(start)

file_temp_max=pd.read_csv("../files/mel_temp_max_data.csv", sep=",")
file_temp_min=pd.read_csv("../files/mel_temp_min_data.csv", sep=",")
file_rainfall=pd.read_csv("../files/mel_rainfall_data.csv", sep=",")




        
print(":"+str(datetime.now()))
for link in links:
    
    LinksFileData=[]
    LinksFileData.append(link)
    #create column
    for i in range(0,40):
        data[i]=""

    
    try:
        uclient=urllib.request.urlopen(link)
    except:
        logger.fileWriteln("\n    ***  WARNING : While browsing.Skipping count:"+str(main_count)+" URL:"+link,loggerFile)            
        data[0]=main_count
        data[1]=link
        main_count=main_count+1
        #data_writter(dataFile,data)

        continue
    #drivr.get(link)
    
    #page= drivr.page_source;
    page=uclient.read()
    soup= bs.BeautifulSoup(page,'html.parser');
    #HISTORY LOADING ####################################################################################
    
    
    
    historyList=soup.select('section.content-wrap.property-profile')
    #print(soup.prettify())
    if historyList is not None and len(historyList)>0:
        history=historyList[0]
        historyList2=history.findAll('a')
        if historyList2 is not None and len(historyList2)>0:
            history2=historyList2[0]['href']
            urlHist=history2
        else:
            urlHist="https://www.domain.com.au/property-profile/"
            addressHistList=link.split("https://www.domain.com.au/")
            addressHist=addressHistList[1]
            count=-1
            y=0
            addressHist2List=addressHist.rsplit("-",1)
        
            history2=addressHist2List[0]
          
            urlHist=urlHist+history2
    else:    
        urlHist="https://www.domain.com.au/property-profile/"
        addressHistList=link.split("https://www.domain.com.au/")
        addressHist=addressHistList[1]
        count=-1
        y=0
        addressHist2List=addressHist.rsplit("-",1)
    
        history2=addressHist2List[0]
      
        urlHist=urlHist+history2
     
    print("1:"+str(datetime.now()))

    
    #pool2 = ThreadPool(processes=2)
    #async_result2 = pool2.apply_async(driver_loader, (urlHist, drivr)) # tuple of args for foo
    driver_loader(urlHist, drivr)
    LinksFileData.append(urlHist)

 
    logger.fileWriteln("----------------------------------------------------------------------------------------",loggerFile)
    logger.fileWriteln(str(main_count)+" :  Processing :"+link,loggerFile)
    data[1]=str(main_count)
    data[2]=link





    
    dateAuc= soup.select('span[class*="listing-details__summary-tag"]')
    if len(dateAuc)<1:
        main_count=main_count+1
        continue
    


    

    #STREET ADDRESS  ##########################################################################################################
    streetAddress= soup.select('h1')[0].text.strip()
    streetAddressList=streetAddress.split(",")
    
    if len(streetAddressList)>0:
        streetAddress=streetAddressList[0]
        data[3]=streetAddress
        #dataFile.write(streetAddress+",")# 3  STREET 
        streetA=streetAddress


    #SUBURB NAME,POST CODE   ####################################################################################################################
    suburb=""
    if len(streetAddressList)>1:
        streetAddress1=streetAddressList[1].lower()
        suburbList=streetAddress1.strip().split('vic')
        if len(suburbList)>1:
            suburb=suburbList[0].strip()
            #dataFile.write(suburbList[0]+",") #  4  SUBURB ADDRESS
            data[4]=(suburbList[0].strip()) #  4  SUBURB ADDRESS
            suburbA=suburbList[0].strip()
            suburbPost=suburbList[1].strip()
            #dataFile.write(suburbPost+",") # 5 POST CODE
            data[5]=(suburbPost) # 5 POST CODE
            postA=suburbPost
            suburbA=suburbA+" vic "+postA
            suburbA=suburbA.replace(" ","-")


     
    distance_matrix_list=[0,0,0,0,0,0,0]
    #pool = ThreadPool(processes=1)
    # async_result = pool.apply_async(distance_google_api_wrapper, (distance_matrix_list,addressGoogle+","+data[4]+","+data[5]+"vic",keys[0], loggerFile)) # tuple of args for foo
    
    
    #AUCTION DATE  , STATUS   ####################################################################################################################
    dateAuc= soup.select('span[class*="listing-details__summary-tag"]')
    if len(dateAuc)>0:
        #dateAucUnformatted=dateAuc[0].text.split('advertiser')
        dateAucUnformatted=re.split(r'(^[^\d]+)', dateAuc[0].text)[1:]
        if len(dateAucUnformatted)>1:
            
            dateformatted=dateFormatter(dateAucUnformatted[1])
            
            yearlist=dateformatted.split("/")
            if len(yearlist)>2:
                year=int(yearlist[2])
            else:
                year=0
            if year<2014:
                main_count=main_count+1
                logger.fileWriteln("\n   *** WARNING: in property sold year below 2013 Skipping count:"+str(main_count)+"URL:"+link,loggerFile)
                #data_writter(dataFile,data)
                continue 
            
            #dataFile.write(dateformatted+",")# 6 SOLD YEAR
            data[6]=(dateformatted)# 6 SOLD YEAR

    
        #print( dateAuc[0].text)    
        if dateAuc[0].text.find("sold") != -1 or dateAuc[0].text.find("Sold") != -1:
            #dataFile.write("sold,") # 7 Status
            data[7]=("sold") # 7 Status



    details= soup.select('div[class*="listing-details__key-features--value"]')
    landsize=""
    internalsize=""
    propertyType=""
    if len(details)>0:
        data[8]=details[0].text# 8 property Type
    
    ''' 
    if len(details)>1:
        internalsize=details[1].text
    if len(details)>2:
        propertyType=details[2].text        
        propertyType2=propertyType.strip()
        propertyType2=propertyType2.replace(",","|")
        #dataFile.write(propertyType2+",") # 8 property Type
        data[8]=(propertyType2) # 8 property Type
    
    else:
        pprty= soup.select('ul[class*="list-vertical"]')
        if len(pprty)>0:
            matching = pprty[len(pprty)-1]
            propertyType2List=matching.text.split(':')
            if len(propertyType2List)>1:
                propertyType2=propertyType2List[1].strip()
                propertyType2=propertyType2.replace(",","|")            
                #dataFile.write(propertyType2+",") # 8 property Type
                data[8]=(propertyType2) # 8 property Type
            
    '''    
     
    # BED, Bath, GARRAGE   ####################################################################################################################
    features= soup.findAll("span", {"class":"f-icon with-text"})
    if len(features)>0:
        if len(re.findall('\d+', features[0].text ))>0:
            #dataFile.write(re.findall('\d+', features[0].text )[0]+",")# 9 Beds\
            data[9]=(re.findall('\d+', features[0].text )[0])# 9 Beds\



    if len(features)>1:
        if len(re.findall('\d+', features[1].text ))>0:
            #dataFile.write(re.findall('\d+', features[1].text )[0]+",")#10 baths
            data[10]=(re.findall('\d+', features[1].text )[0])#10 baths




    if len(features)>2:
        if len(re.findall('\d+', features[2].text ))>0:
            #dataFile.write(re.findall('\d+', features[2].text )[0]+",")#11 Garage
            data[11]=(re.findall('\d+', features[2].text )[0])#11 Garage




    #    ####################################################################################################################           
    url3="https://www.realestateview.com.au/property-360/property/"
    
    streetA1=streetA.replace(" ","-")
    streetA2=streetA1.replace("/","-")
    
    url3=url3+streetA2+"-"+suburbA+"/"
    extPriceFlag=False
    #uclient3=urllib.request.urlopen(url3)
    try:
        uclient3=urllib.request.urlopen(url3)               #  LINK 2 : EXTERAL PRICE
        extPriceFlag=True
    except:
        logger.fileWriteln("\n   *** WARNING: in PRice Estmator  URL :"+url3,loggerFile)
    if extPriceFlag:
        page3= uclient3.read();
        soup3= bs.BeautifulSoup(page3,'html.parser');
        uclient3.close()
        extprice= soup3.find("span", {"class":"total-price"})
    else:
        extprice= None
    #EXTERNAL PRICE ESTIMATOR ENDS    ####################################################################################################################
    
    
    #return_val2 = async_result2.get()  # get the return value from your function.

    
    
    page2= drivr.page_source;
    soup2= bs.BeautifulSoup(page2,'html.parser');    
    
    #PRICE, Low End , High End    ####################################################################################################################
    
    price=soup.select('span.h4.pricedisplay.truncate-single')
    if price is not None and len(price)>0:
        #dataFile.write(price_formatter(price[0].text)+",")#12 price
        data[12]=(price_formatter(price[0].text))#12 price

        
    price0= soup2.find("span", {"class":"short-price-display low-end"})
    price1= soup2.find("span", {"class":"short-price-display high-end"})

    if price0 is not None and len(price0)>0:
        #dataFile.write(price_formatter(price0.text)+",")#13 price Low End
        data[13]=(price_formatter(price0.text))#13 price Low End

        
    if price1 is not None and len(price1)>0:
        #dataFile.write(price_formatter(price1.text)+",")#14 price High End
        data[14]=(price_formatter(price1.text))#14 price High End

        

    #EXTERNAL PRICE WRITING    ####################################################################################################################
    
    if extprice is not None and len(extprice)>0:
        data[15]=price_formatter(extprice.text)#15 External Price estimator
  
    # Garden,pool,heating,cooling
    # Long term resident, rented, singles
    #print(soup)
    #neighbourhoodInsight=soup.find('section[class*="neighbourhood-insights"]')
    neighbourhoodInsight=soup.findAll("section", {"class": "neighbourhood-insights"})
    if neighbourhoodInsight is not None and len(neighbourhoodInsight)>0:
        #longTermResident=neighbourhoodInsight[0].find('text[class*="single-value-doughnut-graph__text"]')
        longTermResident=neighbourhoodInsight[0].findAll("text", {"class": "single-value-doughnut-graph__text"})
        if longTermResident is not None and len(longTermResident)>0:
            data[16]=(longTermResident[0].text.replace("%",""))#16 Long Term Resident
        
        #rentedList=neighbourhoodInsight[0].select('div[class*="composite-bar-graph__bar-right.has-right-colour"]')
        rentedList=neighbourhoodInsight[0].select('div.composite-bar-graph__bar-right,div.has-right-colour')
        #rentedList=neighbourhoodInsight[0].findAll("div", {"class": "composite-bar-graph__bar-right.has-right-colour"})
        
        rentcont=0
        for rented in rentedList:
            rentcont=rentcont+1
            if rentcont==1:
                data[17]=rented.text.replace("%","") #17 % rented
            else:
                data[18]=rented.text.replace("%","") #18 % singles
    # NEAREST SCHOOL
    
    nearestschool=soup.select('span[class*="school-catchment__school-distance"]')
    if nearestschool is not None and len(nearestschool)>0:
        data[41]=nearestschool[0].text.replace("km","")
    
    # Garden,pool,heating,cooling
    try:
        
        tellmemorelist=soup.select('div[class*="listing-details__description"]')
        
        
        if re.search('Garden', tellmemorelist[0].text, re.IGNORECASE):
            data[19]=("yes")#19 garden
        else:
            data[19]=("no")#19 garden
        if re.search('pool', tellmemorelist[0].text, re.IGNORECASE):
            data[20]=("yes")#20 pool
        else:
            data[20]=("no")# 20 pool
        if re.search('heating', tellmemorelist[0].text, re.IGNORECASE):
            data[21]=("yes")#21 heating
        else:
            data[21]=("no")#21 heating
        if re.search('cooling', tellmemorelist[0].text, re.IGNORECASE):
            data[22]=("yes")#22 cooling
        else:
            data[22]=("no")#22 cooling
    except:
        #dataFile.write(",,,,")#14,15,16,17
        logger.fileWriteln("\n    ***  ERROR: in Garden ,pool heating,cooling:"+traceback.format_exc())            
        
        

 

    #SIMILAR PROPERTY SALES  
    similar=0
    for simi in drivr.find_elements_by_xpath('.//div[@class = "properties-like-this__heading"]'):
        similar=similar+1
        if similar> 2:
            break

        data[22+similar]=(price_formatter(simi.text))#23,24 smlr_pprty_price



    


    #print("6:"+str(datetime.now()))

    #dataFile.write(",,,")#28,29,30 floor_size,build_size,build_year
    sold=0
    soldCheck=0
    rentCheck=0
    rent=0
    rentYear=""
    soldYear=""
    soldPrice=""
    rentPrice=""
    #drivr.get("https://www.domain.com.au/property-profile/1513-1-queens-road-melbourne-vic-3000")
    try:
        drivr.find_element_by_css_selector('.button.button__muted').click()
    except:
        temp=0#Place Holder
    for timeLineItem in drivr.find_elements_by_xpath('.//li[@class = "property-timeline-item"]'):
        
            
        try:
            status=timeLineItem.find_element_by_xpath('.//span[@class = "property-timeline__card-category-wrap"]')
        except:
            continue
        if status is not None:
            if status.text.lower()=="sold":
                sold=sold+1
                if(soldCheck==2):
                    continue
                if(soldCheck==0):
                    soldCheck=1
                    continue
                soldYr=timeLineItem.find_element_by_xpath('.//div[@class = "property-timeline__card-date-year"]')
                if soldYr is not None:
                    soldYear=soldYr.text
                soldCheck=2
                soldPr=timeLineItem.find_element_by_xpath('.//span[@class = "property-timeline__card-heading"]')
                if soldPr is not None:
                    soldPrice=price_formatter(soldPr.text)

            else:
                rent=rent+1
                if(rentCheck==0):
                    rentCheck=rentCheck+1
                    rentYr=timeLineItem.find_element_by_xpath('.//div[@class = "property-timeline__card-date-year"]')
                    if rentYr is not None:
                        rentYear=rentYr.text
                    rentPr=timeLineItem.find_element_by_xpath('.//span[@class = "property-timeline__card-heading"]')
                    if rentPr is not None:
                        rentPrice=price_formatter(rentPr.text)               
    #dataFile.write(str(sold)+","+str(rent)+","+soldPrice+","+rentPrice+","+soldYear+","+rentYear+",")# 31 sold num,32 rent num,33 last sold price,34 last rent price,35 sold ye,36 rent yr
    data[28]=str(sold)
    data[29]=str(rent)
    data[30]=soldPrice
    data[31]=soldYear
    data[32]=rentPrice
    data[33]=rentYear
    
    #Weather 
    dateformattedList=dateformatted.split("/")
    if dateformattedList is not None and len(dateformattedList)>2:
        tempmax=getWeatherData(file_temp_max, dateformattedList[2], dateformattedList[1], dateformattedList[0])
        tempmin=getWeatherData(file_temp_min, dateformattedList[2], dateformattedList[1], dateformattedList[0])
        temprain=getWeatherData(file_rainfall, dateformattedList[2], dateformattedList[1], dateformattedList[0])
        #dataFile.write(tempmin+",")#37 Min Temp
        #dataFile.write(tempmax+",")#38 Max Temp
        #dataFile.write(temprain+",")#37 rainfall
        data[34]=tempmin
        data[35]=tempmax
        data[36]=temprain       
    
 

    url4="http://www.onthehouse.com.au/real_estate/vic/"
    #url4=url4+
    #url4=url4+data[3]+",vic "+data[5]
    postal=data[4]+"_"+data[5]
    postal=postal.replace(" ","_")
    url4=url4+postal+"/"
    streetOTHList=data[3].split(" ", 1)
    streetOTH=streetOTHList[1].replace(" ","_")
    url4=url4+streetOTH+"?streetNumber="
    streetOTH0List=streetOTHList[0].split("/")
    if len(streetOTH0List)>1:
        url4=url4+streetOTH0List[1]+"&unitNumber="+streetOTH0List[0]
    else:
        url4=url4+streetOTH0List[0]
    data[25]=url4
    print(url4)
    #http://www.onthehouse.com.au/real_estate/vic/burwood_3125/Delany_Avenue?unitNumber=201&streetNumber=1
    driver_loader(url4, drivr)
    try:
        
        #pricestObj=drivr.find_element_by_css_selector('span.price.ng-binding')
        #pricest=pricestObj.text
        
        drivr.find_element_by_css_selector('div.property-list').click()
        
        driver_loader(drivr.current_url, drivr)
       
        delay = 10 # seconds
        try:
            myElem = WebDriverWait(drivr, delay).until(EC.presence_of_element_located((By.CLASS_NAME, 'legal-attribute-row')))
        except :
            print ("Loading took too much time!")
        legalObjs=drivr.find_elements_by_css_selector('tr.legal-attribute-row')
        for legalObj in legalObjs:
            print(legalObj.text)
            if "year" in legalObj.text.lower():
                    #data[26]= re.findall('\d+', legalObj.text)
                    data[27] = ''.join(x for x in legalObj.text if x.isdigit())
            elif "land" in legalObj.text.lower() and "size" in legalObj.text.lower():
                    data[26] = ''.join(x for x in legalObj.text if x.isdigit())
                    data[26]=data[27][:-1]
            
        print(data[26])
        print(data[27])
        
    except Exception as e:
        print(str(e))
    url4="https://homesales.com.au/location/"+data[4].replace(" ","-")+"-vic/"
    
    
    print(url4)
    
    uclient4=urllib.request.urlopen(url4)
    
    page4=uclient4.read()
    soup4= bs.BeautifulSoup(page4,'html.parser')
    avgdaysonMrkts=soup4.select('table.buy-suburb-stats')
    annualSales=0
    avgDaysOnMarket=0
    if avgdaysonMrkts is not None and len(avgdaysonMrkts)>0:
        avgdaysonMrktstd=avgdaysonMrkts[0].select('td')

        for avgdaysonMrkt in avgdaysonMrktstd:
            if "days" in avgdaysonMrkt.text:
                avgDaysOnMarket=avgDaysOnMarket+int(avgdaysonMrkt.text.replace("days",""))
            try:
                annualSales=annualSales+int(avgdaysonMrkt.text)
            except:
                placeHolder=""
    data[25]=str(avgDaysOnMarket/2)
    data[42]=str(annualSales/2)
    page4=uclient4.read()
    soup4= bs.BeautifulSoup(page4,'html.parser');
    
    #NEAREST SUPERMARKET,RESTAURENT,SCHOOL,TRAIN_STOP,BUS_STOP,TRAM_STOP
    #return_val = async_result.get()  # get the return value from your function.
    data[37]=distance_matrix_list[1]
    data[38]=distance_matrix_list[2]
    data[39]=distance_matrix_list[3]
    data[40]=distance_matrix_list[4]
    #dataFile.write(str(distance_matrix_list[0])+","+str(distance_matrix_list[1])+","+str(distance_matrix_list[2])+","+str(distance_matrix_list[3])+",")
    #dataFile.write("\n")
    print("5:"+str(datetime.now()))
    
    data_writter(dataFile, data)
    #dataFile.flush()

    print("      "+str((main_count/arlist[0])*100)+" % Completed at : "+str(datetime.now()))
    main_count=main_count+1