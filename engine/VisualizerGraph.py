import matplotlib.pyplot as plt
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

count=-1
x_axis=[]
y_axis=[]
with open('../data/data_domains5.csv', newline='') as myFile:  
   
        reader = csv.reader(myFile)
        for row in reader:
            count=count+1
            if count==0:
                continue
            if row[2]=="NA":
                continue
            if row[12]!="2":
                continue
            x_axis.append(int(row[48]))
            y_axis.append(float(row[2]))

plt.plot(x_axis, y_axis, 'ro')

#plt.plot([50003,50006,555555], [20150000,20150102,20170001], 'ro')
plt.axis([50000, 1200000, 0,10000 ])
plt.show()