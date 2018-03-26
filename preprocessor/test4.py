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


#dataFile= open('../data/data_domains4.csv','a',encoding="utf8")

count=-1
items=['','']
with open('../data/data_domains4.csv', newline='') as myFile:  
   
        reader = csv.reader(myFile)
        for row in reader:
            count=count+1
            #print(row[7])
            if row[44] in items:
                temp=0
            else:
                items.append(row[44]) 
                
for item in items:
    print(item)

            
