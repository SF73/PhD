from PIL import Image
import numpy as np
import time
def  scaleSEMimage(file):
    """
    

    Parameters
    ----------
    file : .tif file
        Tiff SEM Image

    Returns
    -------
    xscale : numpy array
        X-coordinates of the pixels in micron
    yscale : numpy array
        Y-coordinates of the pixels in micron
    Acceleration : float
        Acceleration voltage of the SEM
    im : Image object
        SEM Image without infobar

    """
    with open(file,'r',encoding="Latin-1") as myfile:
        data =myfile.read()
    PixelWidth=eval(data[data.find('PixelWidth=')+11:data.find('PixelWidth=')+23]) #largeur d'un pixel
    PixelHeight=eval(data[data.find('PixelHeight=')+12:data.find('PixelHeight=')+24]) #hauteur d'un pixel
    width=eval(data[data.find('ResolutionX=')+12:data.find('ResolutionX=')+16]) #largeur en #pixel
    height=eval(data[data.find('ResolutionY=')+12:data.find('ResolutionY=')+16]) #hauteur en #pixel
    Acceleration=float(eval(data[data.find('HV=')+3:data.find('HV=')+8]))
    #FullImage =  image.crop((0,0,width,height))
    Totallength_x = PixelWidth *width #largeur de l'image en m
    Totallength_y = height*PixelHeight #hauteur en m
    xscale = np.linspace(-Totallength_x/2,Totallength_x/2,width)/1e-6
    yscale = np.linspace(-Totallength_y/2,Totallength_y/2,height)/1e-6
    im = Image.open(file).convert('I')
    im = im.crop((0,0,width,height))
    return xscale,yscale,Acceleration,im