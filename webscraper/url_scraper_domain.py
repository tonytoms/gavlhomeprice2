import bs4 as bs 
import urllib.request
import webscraper.logger as logger
from datetime import datetime
import copy
import sys


'''
Created on Dec 23, 2017

     Execution Order : 1
     Input files: None
     Output Files: links_domain.csv - contains all listing links
     Input : 
             1- Starting post code(usually 3000)
     
 The URL scraper will find all the URLs of listings from the domain.com.au 
 
@author: Tony Toms
'''


##################################################################################################################################################
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


##################################################################################################################################################
 
# From each search result 'link' found, all listing urls are written into a file
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




##################################################################################################################################################
    
#Basic URL    
domainUrl="https://www.domain.com.au/sold-listings/?ssubs=1&postcode="
postCode=3000  # Starting Post code
ListingCount=1
 
'''
FOR     MOUNT WAVERLY 3149
nearby suburbs are:
3125,3147,3148,3150,3151,3166,3168,

'''
loggerFile=logger.init()

logger.fileWriteln("----------------------------------------------------------------------",loggerFile)
logger.fileWriteln("SCRAPPING : URL SCRAPPING DOMAIN.COM.AU----  "+str(datetime.now()),loggerFile)
print("SCRAPPING : URL SCRAPPING DOMAIN.COM.AU----------")
logger.fileWriteln("----------------------------------------------------------------------",loggerFile)

start = input("ENter the STarting Post CODE:")
logger.fileWriteln(" Starting from POST CODE :"+str(start),loggerFile)

#postcodes=[3149,3125,3147,3148,3150,3151,3166,3168]


#PRICE PARAMETER IN URL
priceControl=50000
incrementer=50000
pricerange=[]


# For each post code 
#     each search result is again filtered by price slab starting from 50 000 to 1 300 000
#    Search results after page number 50 can't be obtained. In order to reduce the total page numbers , we are splitting the search
#    with price slabs. 
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

                logger.fileWriteln("      - URL :"+pageUrl1+"&page="+str(y)  +" : Listings Found :" +str(ListingCount-ListingCountBef),loggerFile)
    print("\n SCRAPPING : URL SCRAPPING DOMAIN.COM.AU:Suburb - "+str(x)+" Completed")



logger.fileWriteln("----------------------------------------------------------------------",loggerFile)
logger.fileWriteln("SCRAPPING: URL SCRAPPING DOMAIN.COM.AU COMPLETED :"+str(datetime.now())+ " TOTAL :"+str(ListingCount)+"  URLs Found",loggerFile)
logger.fileWriteln("----------------------------------------------------------------------",loggerFile)

logger.destry(loggerFile)
