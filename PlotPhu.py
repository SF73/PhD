# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:20:33 2019

@author: Sylvain
"""

# Read_PHU.py    Read PicoQuant Unified TTTR Files
# This is demo code. Use at your own risk. No warranties.
# Keno Goertz, PicoQUant GmbH, February 2018

# Note that marker events have a lower time resolution and may therefore appear 
# in the file slightly out of order with respect to regular (photon) event records.
# This is by design. Markers are designed only for relatively coarse 
# synchronization requirements such as image scanning. 

# T Mode data are written to an output file [filename]
# We do not keep it in memory because of the huge amout of memory
# this would take in case of large files. Of course you can change this, 
# e.g. if your files are not too big. 
# Otherwise it is best process the data on the fly and keep only the results.

import time
import sys
import struct
import matplotlib.pyplot as plt
import numpy as np
from ReadPhu import readphu
plt.style.use("Rapport")

if __name__ == '__main__':
    if len(sys.argv)>1:
        fig, ax =plt.subplots()
        paths = sys.argv[1:]
        for p in paths:
            #inputfile = sys.argv[1]
            t,counts,_ = readphu(p)
            ax.plot(t*1e9,counts,'.')
            ax.set_xlabel("Delay (ns)")
            ax.set_ylabel("Counts")
        plt.show(block=True)