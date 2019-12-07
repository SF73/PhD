# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:18:37 2019

@author: sylvain.finot
"""
import os.path
import time
import numpy as np
from expfit import process_fromFile
from FullReport import make_linescan


def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
#    dirName = r'\\srv-echange\echange\Sylvain\2019-04-16 - T2594Al - 300K\4\HYP1-T2594Al-300K-Vacc5kV-spot7-zoom6000x-gr600-slit0-2-t5ms-cw440nm'
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            print(fullPath)
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles
def process_all(path):
    files=getListOfFiles(path)
    Carto = [x for x in files if (("Hyp" in x) & (x.endswith(".dat")))]
    TRCL = [x for x in files if (("TRCL" in x) & (x.endswith(".dat")))]
    for p in TRCL:
        try:
            process_fromFile(p,save=True,autoclose=True)
        except:
            pass
    for p in Carto:
        try:
            make_linescan(p,save=True,autoclose=False,Linescan=True,deadPixeltol=200)
        except:
            pass          
def process_network():
#    dirName = r"\\srv-echange\echange\Sylvain"
    dirName = r'\\srv-echange.grenoble.cnrs.fr\echange\Sylvain'
    while(True):
        print('Processing')
        dirs = os.listdir(dirName)
        selection = [x for x in dirs if (time.strftime("%Y-%m-%d") in x)]
        for entry in selection:
            fullPath = os.path.join(dirName, entry)
            toProcess = os.path.exists(os.path.join(fullPath,'process.dat')) & (not os.path.exists(os.path.join(fullPath,'done.dat')))
            if toProcess==True:
                print(fullPath)
                process_all(fullPath)
                os.path.exists(os.path.join(fullPath,'process.dat'))
                np.savetxt(os.path.join(fullPath,'done.dat'),[])
        time.sleep(5)