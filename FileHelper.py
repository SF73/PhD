import os.path
import numpy as np
import pandas as pd
def getListOfFiles(dirName,recursive=True):
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
            if not(recursive):continue
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

def getSpectro_info(path):
    #path=r"C:\Users\sylvain.finot\Cloud Neel\Data\2020\2020-02-20 - T2575\005K\Wire 1\Hyp_T005K_cw295nm_spot3-5_HV5kV_gr600_slit1mm_t010ms_zoom12000_spectro_info.asc"
    return pd.read_csv(path,delimiter='\t').to_numpy()

def casino(path):
    trajectories = []
    with open(path,'r') as f:
        for line in f:
            if line.startswith("BackScattered"):
                print(line)
                nextline = f.readline()
                print(nextline)
                N = int(nextline.split()[4])
                f.readline()
                f.readline()
                traj = []
                for n in range(N):
                    l = f.readline().split()
                    temp = [float(x) for x in l[:-1]]
                    #temp.append(l[-1]) #contient le label
                    traj.append(temp)
                trajectories.append(np.array(traj)) #trajectoire d'un electron
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for t in trajectories:
        ax.plot(t[:,0],t[:,1],-t[:,2],c='C0')