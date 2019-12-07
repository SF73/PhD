# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:57:54 2019

@author: sylvain.finot
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os.path

def  scaleSEMimage(file):

    with open(file,'r',encoding="Latin-1") as myfile:
        data =myfile.read()
    PixelWidth=eval(data[data.find('PixelWidth=')+11:data.find('PixelWidth=')+23])
    PixelHeight=eval(data[data.find('PixelHeight=')+12:data.find('PixelHeight=')+24])
    width=eval(data[data.find('ResolutionX=')+12:data.find('ResolutionX=')+16])
    height=eval(data[data.find('ResolutionY=')+12:data.find('ResolutionY=')+16])
    Acceleration=eval(data[data.find('HV=')+3:data.find('HV=')+8]) 
    #FullImage =  image.crop((0,0,width,height))
    Totallength_x = PixelWidth *width
    Totallength_y = height*PixelHeight
    xscale = np.linspace(-Totallength_x/2,Totallength_x/2,width)/1e-6
    yscale = np.linspace(-Totallength_y/2,Totallength_y/2,height)/1e-6
    im = Image.open(file).convert('I')
    im = im.crop((0,0,width,height))
    return xscale,yscale,Acceleration,im

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

def process_all():
    files=getListOfFiles(r'C:/Users/sylvain.finot/Documents/data/')
    mask = [x for x in files if (("Hyp" in x) & (x.endswith(".dat")))]
    for p in mask:
        try:
            make_linescan(p,autoclose=True)
        except:
            pass
        
def make_linescan(path,save=True,autoclose=False):
#    path=r'C:/Users/sylvain.finot/Documents/data/2019-03-08 - T2601 - 300K/Fil2/HYP1-T2601-300K-Vacc5kV-spot7-zoom6000x-gr600-slit0-2-t5ms-Fil1-cw380nm/Hyp.dat'
    dirname = os.path.dirname(path)
    hyppath = path
    specpath = os.path.join(dirname,'Hyp_X_axis.asc')
    filepath = os.path.join(dirname,'Hyp_SEM image after carto.tif')
    data = np.loadtxt(hyppath)
    xlen = int(data[0,0])
    ylen = int(data[1,0])
    wavelenght = np.loadtxt(specpath)
    wavelenght = wavelenght[:2048]
    xcoord = data[0,1:]
    ycoord = data[1,1:]
    CLdata = data[2:,1:] #tableau de xlen * ylen points (espace) et 2048 longueur d'onde CLdata[:,n] n = numero du spectr
    
    hypSpectrum = np.transpose(np.reshape(np.transpose(CLdata),(ylen,xlen ,len(wavelenght))), (0, 1, 2))
    average_axis = 0 #1 on moyenne le long du fil, 0 transversalement
    linescan = np.sum(hypSpectrum,axis=average_axis)
    linescan -= linescan.min()
#    linescan = np.log10(linescan)
    xscale_CL,yscale_CL,acc,image = scaleSEMimage(filepath)
    fig,(ax,bx)=plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios': [1, 3]})
    fig.subplots_adjust(top=0.98,bottom=0.11,left=0.1,right=0.82,hspace=0.0,wspace=0.2)
    newX = np.linspace(xscale_CL[int(xcoord.min())],xscale_CL[int(xcoord.max())],len(xscale_CL))
    newY = np.linspace(yscale_CL[int(ycoord.min())],yscale_CL[int(ycoord.max())],len(yscale_CL))
    
    nImage = np.array(image.crop((xcoord.min(),ycoord.min(),xcoord.max(),ycoord.max())))#-np.min(image)
#    minC = nImage.min()
#    maxC=nImage.max()
    ax.imshow(nImage,cmap='gray',vmin=0,vmax=65535,extent=[np.min(newX),np.max(newX),np.max(newY),np.min(newY)])
    ax.set_ylabel("distance (µm)")
    if average_axis==1:
        extent = [1239.842/wavelenght.max(),1239.842/wavelenght.min(),np.max(newY),np.min(newY)]
        im=bx.imshow(linescan,cmap='jet',extent=extent)
        bx.set_xlabel("energy (eV)")
        bx.set_ylabel("distance (µm)")
    else:
        extent = [np.min(newX),np.max(newX),1239.842/wavelenght.max(),1239.842/wavelenght.min()]
        im=bx.imshow(linescan.T,cmap='jet',extent=extent)
        bx.set_ylabel("Energy (eV)")
        bx.set_xlabel("distance (µm)")

    ax.set_aspect('auto')
    bx.set_aspect('auto')
    cbar_ax = fig.add_axes([0.85, 0.11, 0.05, 0.65])
    fig.colorbar(im, cax=cbar_ax)
    if save==True:
        savedir = os.path.join(dirname,'Saved')
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        fig.savefig(os.path.join(dirname,'Saved','linescan.png'),dpi=300)
    if autoclose==True:
        plt.close(fig)
def main():
    path = input("Enter the path of your file: ")
    path=path.replace('"','')
    path=path.replace("'",'')
#    path = r'C:/Users/sylvain.finot/Documents/data/2019-03-11 - T2597 - 5K/Fil3/TRCL-cw455nm/TRCL.dat'
    make_linescan(path)
    
if __name__ == '__main__':
    main()