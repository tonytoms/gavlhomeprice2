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


def distance_google_api_wrapper(distance_matrix_list,addresss,keys):
    url4="https://maps.googleapis.com/maps/api/distancematrix/xml?units=metric&origins="+addresss+"&destinations=melbourne,CBD&key="
    url5="https://maps.googleapis.com/maps/api/place/textsearch/xml?query=train stop+in+"+addresss+"&key="
    url6="https://maps.googleapis.com/maps/api/place/textsearch/xml?query=bus stop+in+"+addresss+"&key="
    url7="https://maps.googleapis.com/maps/api/place/textsearch/xml?query=shopping mall+in+"+addresss+"&key="
    err_keys=[""]
    #for strkey in keys:
    #    try:
    
    
    distance_google_api(1,url4,distance_matrix_list,keys,addresss)#40 Distance to nearest shopping mall
    distance_google_api(2,url5,distance_matrix_list,keys,addresss)#41 Distance to nearest restaurant
    distance_google_api(3,url6,distance_matrix_list,keys,addresss)#42 Distance to nearest school
    distance_google_api(4,url7,distance_matrix_list,keys,addresss)#43 Distance to nearest secondary college
    
    
    
    #        break
    #    except Exception as e:
    #        logger.fileWriteln("WARNING IN GOOGLE API:"+str(e),loggerFile)
    #        err_keys.append(strkey)
    #del err_keys[0]
    #for err_key in err_keys:
    #    index = keys.index(err_key)
        #keys.remove(index)  
    #    del keys[index] 
     
    return True

  

# distance Calcualtor
def distance_google_api(no,url,distance_matrix_list,strkey,address):

       
    if no!=1:
        response = requests.get(url+strkey.strip())
        
        tree = ElementTree.fromstring(response.content)
        #tree9 = ElementTree.fromstring(page4)
        tree1=tree.find("result")
        tree2=tree1.find("geometry")
        tree3=tree2.find("location")
        tree4=tree3.find("lat").text
        tree5=tree3.find("lng").text
             
        urlDist="https://maps.googleapis.com/maps/api/distancematrix/xml?units=metric&origins="+tree4+","+tree5+"&destinations="+address+"&key="+strkey.strip()
        print(urlDist)
        response = requests.get(urlDist)

        tree = ElementTree.fromstring(response.content)
        #tree9 = ElementTree.fromstring(page4)
        tree10=tree.find("row")
        tree11=tree10.find("element")
        tree12=tree11.find("distance")
        tree13=tree12.find("text")
        distance=tree13.text.replace("km","")
        distance=distance.strip()
        distance_matrix_list[no]=distance
    else:
        #urlDist=url+"AIzaSyCE2i1XF7hq3plRbHznMNpOsM0kuGvJeg8"
        urlDist=url+""+strkey.strip()

        print(urlDist)
        #response = requests.get("https://maps.googleapis.com/maps/api/distancematrix/xml?units=metric&origins=G36/382-386 Burwood Highway, burwood,3125vic&destinations=melbourne,CBD&key=AIzaSyCE2i1XF7hq3plRbHznMNpOsM0kuGvJeg8")
        response = requests.get(urlDist)

        tree = ElementTree.fromstring(response.content)
        #tree9 = ElementTree.fromstring(page4)
        tree10=tree.find("row")
        tree11=tree10.find("element")
        tree12=tree11.find("distance")
        tree13=tree12.find("text")
        distance=tree13.text.replace("km","")
        distance=distance.strip()
        distance_matrix_list[no]=distance             
    
    return True
 

import csv

start = input("Enter starting index: ")
dataFile= open('../data/data_domains2.csv','a',encoding="utf8")

keys=[]
with open('../files/googleAPIKEYS.txt', 'r') as ins:
    for line in ins:
        keys.append(line)

count=-1
with open('../data/data_domains.csv', newline='') as myFile:  
   
        reader = csv.reader(myFile)
        for row in reader:
            count=count+1
            if count<int(start) or count==0:
                tempPlaceHolder=0
            else: 
            
                addressGoogles=row[2].split("/")
                if len(addressGoogles)>1:
                    addressGoogle=addressGoogles[1]
                else:
                    addressGoogle=addressGoogles[0]
                regex = re.compile('[^a-zA-Z ]')
                addressGoogle=regex.sub('', addressGoogle)
                distance_matrix_list=[0,0,0,0,0,0,0]
                distance_google_api_wrapper(distance_matrix_list,addressGoogle+","+row[3]+","+row[4]+"vic",keys[0])        
                row[36]=distance_matrix_list[1]
                row[37]=distance_matrix_list[2]
                row[38]=distance_matrix_list[3]
                row[39]=distance_matrix_list[4]       
            myString = ",".join(row )
            dataFile.write(myString)
            dataFile.write("\n")
       
        
        dataFile.flush()
        
        
        
        
        
         