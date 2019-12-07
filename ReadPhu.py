# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:57:27 2019

@author: Sylvain.FINOT
"""

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
import struct
import numpy as np
import matplotlib.pyplot as plt

def normalize(x,min,max):
    return (x-min)/(max-min)

def readphu(path):
    # Tag Types
    tyEmpty8      = struct.unpack(">i", bytes.fromhex("FFFF0008"))[0]
    tyBool8       = struct.unpack(">i", bytes.fromhex("00000008"))[0]
    tyInt8        = struct.unpack(">i", bytes.fromhex("10000008"))[0]
    tyBitSet64    = struct.unpack(">i", bytes.fromhex("11000008"))[0]
    tyColor8      = struct.unpack(">i", bytes.fromhex("12000008"))[0]
    tyFloat8      = struct.unpack(">i", bytes.fromhex("20000008"))[0]
    tyTDateTime   = struct.unpack(">i", bytes.fromhex("21000008"))[0]
    tyFloat8Array = struct.unpack(">i", bytes.fromhex("2001FFFF"))[0]
    tyAnsiString  = struct.unpack(">i", bytes.fromhex("4001FFFF"))[0]
    tyWideString  = struct.unpack(">i", bytes.fromhex("4002FFFF"))[0]
    tyBinaryBlob  = struct.unpack(">i", bytes.fromhex("FFFFFFFF"))[0]
    
    
    inputfile = open(path, "rb")
    
    # Check if inputfile is a valid PHU file
    # Python strings don't have terminating NULL characters, so they're stripped
    magic = inputfile.read(8).decode("ascii").strip('\0')
    if magic != "PQHISTO":
        print("ERROR: Magic invalid, this is not a PHU file.")
        exit(0)
    
    version = inputfile.read(8).decode("ascii").strip('\0')
    
    # Write the header data to outputfile and also save it in memory.
    # There's no do ... while in Python, so an if statement inside the while loop
    # breaks out of it
    tagDataList = []    # Contains tuples of (tagName, tagValue)
    while True:
        tagIdent = inputfile.read(32).decode("ascii").strip('\0')
        tagIdx = struct.unpack("<i", inputfile.read(4))[0]
        tagTyp = struct.unpack("<i", inputfile.read(4))[0]
        if tagIdx > -1:
            evalName = tagIdent + '(' + str(tagIdx) + ')'
        else:
            evalName = tagIdent
        if tagTyp == tyEmpty8:
            inputfile.read(8)
            tagDataList.append((evalName, "<empty Tag>"))
        elif tagTyp == tyBool8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            if tagInt == 0:
                tagDataList.append((evalName, "False"))
            else:
                tagDataList.append((evalName, "True"))
        elif tagTyp == tyInt8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyBitSet64:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyColor8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyFloat8:
            tagFloat = struct.unpack("<d", inputfile.read(8))[0]
            tagDataList.append((evalName, tagFloat))
        elif tagTyp == tyFloat8Array:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyTDateTime:
            tagFloat = struct.unpack("<d", inputfile.read(8))[0]
            tagTime = int((tagFloat - 25569) * 86400)
            tagTime = time.gmtime(tagTime)
            tagDataList.append((evalName, tagTime))
        elif tagTyp == tyAnsiString:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagString = inputfile.read(tagInt).decode("ascii").strip("\0")
            tagDataList.append((evalName, tagString))
        elif tagTyp == tyWideString:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagString = inputfile.read(tagInt).decode("ascii").strip("\0")
            tagDataList.append((evalName, tagString))
        elif tagTyp == tyBinaryBlob:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        else:
            print("ERROR: Unknown tag type")
            exit(0)
        if tagIdent == "Header_End":
            break
    
    # Reformat the saved data for easier access
    tagNames = [tagDataList[i][0] for i in range(0, len(tagDataList))]
    tagValues = [tagDataList[i][1] for i in range(0, len(tagDataList))]
    
    # Write histogram data to file
    curveIndices = [tagValues[i] for i in range(0, len(tagNames))\
                    if tagNames[i][0:-3] == "HistResDscr_CurveIndex"]
    
    for i in curveIndices:
        histogramBins = tagValues[tagNames.index("HistResDscr_HistogramBins(%d)" % i)]
        resolution = tagValues[tagNames.index("HistResDscr_MDescResolution(%d)" % i)]
    #    outputfile.write("\nCurve#  %d" % i)
    #    outputfile.write("\nnBins:  %d" % histogramBins)
    #    outputfile.write("\nResol:  %3E" % resolution)
    #    outputfile.write("\nCounts:")
        counts = -np.ones(histogramBins)
        for j in range(0, histogramBins):
            try:
                histogramData = struct.unpack("<i", inputfile.read(4))[0]
            except:
                print("The file ended earlier than expected, at bin %d/%d."\
                      % (j, histogramBins))
            counts[j]=int(histogramData)
        t=np.arange(0,int(histogramBins))*float(resolution)
    inputfile.close()
    return t,counts,tagDataList

t,c,tagDataList = readphu(r"C:\Users\sylvain.finot\Cloud Neel\Data\2019-10-15 - T2455\TRCL2_300K_450nm_slit0-1_side0-1_HV5kV_spot2-5_f400kHz_pulse1-25us.phu")
tagD=np.array(tagDataList)
syncDivider = tagD[:,1][tagD[:,0]=="HistResDscr_HWSyncDivider(0)"][0]
duration = tagD[:,1][tagD[:,0]=="HistResDscr_MDescStopAfter(0)"][0] #en ms
syncRate = tagD[:,1][tagD[:,0]=="HistResDscr_SyncRate(0)"][0] #Hz
InputRate = tagD[:,1][tagD[:,0]=='HistResDscr_InputRate(0)'][0] #Hz
dataOffset = tagD[:,1][tagD[:,0]=='HistResDscr_DataOffset(0)'][0] #?

minHist = np.histogram((c[(c>0) & (c<max(c)/2)]),bins=int(max(c)/10))
maxHist = np.histogram((c[(c>max(c)/2)]),bins=int(max(c)/10))
maxc = maxHist[1][maxHist[0].argmax()]
minc =  minHist[1][minHist[0].argmax()]

plt.plot(t,normalize(c,minc,maxc))
