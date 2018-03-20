'''
Created on Dec 23, 2017

This module is for logging activities
@author: Tony Toms
'''

def init():
    file = open("../files/log.txt","a")
    return file
def destry(file):
    file.close()
def fileWriteln(strng,file):
    file.write(strng+"\n")
    file.flush()
def fileWrite(strng,file):
    file.write(strng) 
    file.flush()   
