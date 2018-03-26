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
import csv
import requests


start = input("Enter starting index: ")
dataFile= open('../data/data_domains3.csv','a',encoding="utf8")

keys=[]
with open('../files/googleAPIKEYS.txt', 'r') as ins:
    for line in ins:
        keys.append(line)

count=-1
with open('../data/data_domains2.csv', newline='') as myFile:  
   
        reader = csv.reader(myFile)
        for data in reader:
            count=count+1
            if count<int(start) or count==0:
                tempPlaceHolder=0
            else: 
                
                link="http://www.showneighbour.com/land.php?sta=vic&region="
                link=link+data[3]
                link=link+"&addr="
                
                addressList=data[2].split("/")
                if len(addressList)>1:
                    address=addressList[1]
                else:
                    address=addressList[0]
 
                 
                addressList=address.split("-")
                if len(addressList)>1:
                    address=addressList[1]
                else:
                    address=addressList[0]                   
                    
                link=link+address
                link=link.strip()
                link=link.replace(" ","+")
                print(link)
                uclient=urllib.request.urlopen(link)
                
                #page= drivr.page_source;
                page=uclient.read()
                soup= bs.BeautifulSoup(page,'html.parser')
                
                tds= soup.select('td')
                
                details=["","","","","",""]
                
                count=0
                check=0
                while count<len(tds):
                    
                    #print(tds[count].text)
                    if(tds[count].text=="Land size:"):
                        details[0]=tds[count+1].text
                    if(tds[count].text=="Frontage:"):
                        details[1]=tds[count+1].text
                    if(tds[count].text=="Slope:"):
                        details[2]=tds[count+1].text
                    #if("Degree of slope" in tds[count].text and "The back of the land is" in tds[count].text and check==0):
                     #   details[3]=tds[count].text
                      #  check=check+1
                    #if("Degree of slope" in tds[count].text and "The left of the land" in tds[count].text and check==1):
                     #   details[4]=tds[count].text                                                                                            
                    
                    count=count+1

                for detail in details:
                    detail=detail.replace(",","")
                    data.append(detail)
                    #print(detail+"\n")            
                #start = input("Enter starting index: ")
                myString = ",".join(data )
                dataFile.write(myString)
                dataFile.write("\n")
                dataFile.flush()      
        
