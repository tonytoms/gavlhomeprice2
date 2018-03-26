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
import csv



dataFiles2= open('../data/data_domains4.csv','w',encoding="utf8")





dataHeader=[]
for x in range(0,46):
    dataHeader.append("")

dataHeader[1]="1-no"
dataHeader[2]="2-Link"
dataHeader[3]="3-street"
dataHeader[4]="4-Suburb_add"
dataHeader[5]="5-post"
dataHeader[6]="6-sold_year"
dataHeader[7]="7-status"
dataHeader[8]="8-property_type"
dataHeader[9]="9-bed"
dataHeader[10]="10-bath"
dataHeader[11]="11-car"
dataHeader[12]="12-price"
dataHeader[13]="13-price_range_low_end"
dataHeader[14]="14-price_range_high_end"
dataHeader[15]="15-External_price_estimator"
dataHeader[16]="16-Long_term_residence_percent"
dataHeader[17]="17-Rented_percent"
dataHeader[18]="18-singles_percent"
dataHeader[19]="19-garden"
dataHeader[20]="20-pool"
dataHeader[21]="21-heating"
dataHeader[22]="22-cooling"
dataHeader[23]="23-smlr_pprty_price_1"
dataHeader[24]="24-smlr_pprty_price_2"
dataHeader[25]="25-average_days_in_market_Suburb"
dataHeader[26]="26-land_size"
dataHeader[27]="27-build_year"
dataHeader[28]="28-number_of_times_sold"
dataHeader[29]="29-num_of_times_rented"
dataHeader[30]="30-last_sold_price"
dataHeader[31]="31-last_sold_year"
dataHeader[32]="32-last_rent_price"
dataHeader[33]="33-last_rent_year"
dataHeader[34]="34-min_temp"
dataHeader[35]="35-max_temp"
dataHeader[36]="36-Rainfall"
dataHeader[37]="37-distance_cbd"
dataHeader[38]="38-distance_train_station"
dataHeader[39]="39-distance_bus_school"
dataHeader[40]="40-distance_shopping_mall"
dataHeader[41]="41-distance_school"
dataHeader[42]="42-properties_sold_suburb"
dataHeader[43]="43-Land-Area"
dataHeader[44]="44-Frontage"
dataHeader[45]="45-slop"


for n in range(1,len(dataHeader)):

    dataFiles2.write(dataHeader[n])
    dataFiles2.write(",")

dataFiles2.write("\n")

count=-1
with open('../data/data_domains3.csv', newline='') as myFile:  
   
        reader = csv.reader(myFile)
        for row in reader:
            count=count+1
            myString = ",".join(row )
            dataFiles2.write(myString)
            dataFiles2.write("\n")
       
        
        dataFiles2.flush()