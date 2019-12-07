from PIL import Image
import numpy as np

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