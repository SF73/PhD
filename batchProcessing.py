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
from FileHelper import getListOfFiles

def process_all(path):
    files=getListOfFiles(path)
    Carto = [x for x in files if (("Hyp" in x) & (x.endswith(".dat")))]
    TRCL = [x for x in files if (("TRCL" in x) & (x.endswith(".dat")))]
    for p in TRCL:
        try:
            process_fromFile(p,save=True,autoclose=True)
        except Exception as e:
            print(e)
            pass
    for p in Carto:
        try:
            make_linescan(p,save=True,autoclose=False,Linescan=True,deadPixeltol=200)
        except Exception as e:
            print(e)
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