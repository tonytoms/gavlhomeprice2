import bs4 as bs 
import urllib.request
import webscraper.logger as logger
from datetime import datetime
import copy
import sys


'''
Created on Dec 23, 2017
 The URL scraper will find all the URLs from the domain.com.au 
@author: Tony Toms
'''

#This function will find all total number of pages for the particular post code 
def totPageNumber(pageUrl):
        uclient=urllib.request.urlopen(pageUrl)
        page= uclient.read();
        uclient.close();
        soup= bs.BeautifulSoup(page,'html.parser');
        paginator= soup.findAll("a", {"class":"paginator__page-button"})
        if len(paginator)>0:
            lastPageUrl=paginator[len(paginator) - 1]
            lastPageUrl=lastPageUrl.string
        else:
            lastPageUrl=0
            
        uclient.close()
        return int(lastPageUrl)
 

def writeListingLinks(link, ListingCount):   
    Linkfile = open("../data/links_domain.csv","a") 
    k=1

    uclient2=urllib.request.urlopen(link)
    page= uclient2.read();
    uclient2.close();
    soup= bs.BeautifulSoup(page,'html.parser');
    count=0
    for EachPart in soup.select('a[class*="listing-result"]'):
        Linkfile.write(str(ListingCount)+","+ EachPart["href"]+",D\n")
        ListingCount=ListingCount+1
        

    

    Linkfile.close()
    return ListingCount

    
#Basic URL    
domainUrl="https://www.domain.com.au/sold-listings/?ssubs=1&postcode="
postCode=3000
ListingCount=1
 
'''
FOR     MOUNT WAVERLY 3149
nearby suburbs are:
3125,3147,3148,3150,3151,3166,3168,

'''
loggerFile=logger.init()

logger.fileWriteln("----------------------------------------------------------------------",loggerFile)
logger.fileWriteln("STEP 1 of 4 : URL SCRAPPING DOMAIN.COM.AU----  "+str(datetime.now()),loggerFile)
print("STEP 1 of 4 : URL SCRAPPING DOMAIN.COM.AU----------")
logger.fileWriteln("----------------------------------------------------------------------",loggerFile)

start = input("ENter the STarting Post CODE:")
logger.fileWriteln(" Starting from POST CODE :"+str(start),loggerFile)

#postcodes=[3149,3125,3147,3148,3150,3151,3166,3168]


#PRICE PARAMETER IN URL
priceControl=50000
incrementer=50000
pricerange=[]
while priceControl<13000000:
    if priceControl<1000000:
        incrementer=50000
    elif priceControl<2000000:
        incrementer=100000
    elif priceControl<3000000:
        incrementer=250000
    elif priceControl<3000000:
        incrementer=500000
    else:
        incrementer=1000000
    pricerange.append("&price="+str(priceControl)+"-"+str(priceControl+incrementer))
    priceControl=priceControl+incrementer

#Post codes range from 3000 to 3999
for x in range(int(start),4000):
    
    '''
    # ONLY CONSIDER THE POSTCODES In 'postcodes'
    if x in postcodes:
        placeholder=1
    else:
        continue

    '''
    
    pageUrl=domainUrl+str(x);

    # LOOP THROUGH PRICE RANGE
    for pricerangeItem in pricerange:
        pageUrl1=pageUrl+pricerangeItem
        totalPages=totPageNumber(pageUrl1)
        logger.fileWriteln("Post Code:"+str(x)+" : price Range :"+pricerangeItem+" : Number of Pages: " +str(totalPages) ,loggerFile)
        #If total page > 9 then the all pages for that Post code is saved to the file
        print(pageUrl1)
        if totalPages>0:
            for y in range(1,totalPages+1):
     
                ListingCountBef=ListingCount
                
                # TRY 5 times in case of page not found error
                for t in range(1,5):
                    try:
                        ListingCount=writeListingLinks(pageUrl1+"&page="+str(y),ListingCount)
                    except:
                        continue
                    break

                #print("\n       Listing SCRAPPING DOMAIN.COM.AU: "+str(y*100/totalPages)+"% Completed")
                logger.fileWriteln("      - URL :"+pageUrl1+"&page="+str(y)  +" : Listings Found :" +str(ListingCount-ListingCountBef),loggerFile)

       

            
    #print("\n STEP 1 of 4 : URL SCRAPPING DOMAIN.COM.AU:"+str( ((x-3000)/(3999-3000) ) *100 )+"% Completed")
    print("\n STEP 1 of 4 : URL SCRAPPING DOMAIN.COM.AU:Suburb - "+str(x)+" Completed")

logger.fileWriteln("----------------------------------------------------------------------",loggerFile)

logger.fileWriteln("STEP 1 of 4: URL SCRAPPING DOMAIN.COM.AU COMPLETED :"+str(datetime.now())+ " TOTAL :"+str(ListingCount)+"  URLs Found",loggerFile)
logger.fileWriteln("----------------------------------------------------------------------",loggerFile)

logger.destry(loggerFile)
