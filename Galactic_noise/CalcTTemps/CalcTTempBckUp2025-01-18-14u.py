import numpy as np
from Galactic_noise.GalaxyFunctions import *
import sys
from datetime import datetime
from os import listdir

#StNr, RefData, ChNrs=23,'V_Yr22','13'
#RefFile=[f for f in listdir(DataStorage+"/" + RefData + "/St" + str(StNr)) if (f[-3:]=="pkl" and f[13:15] in ChNrs.split(','))]#[0]
#print([f for f in listdir(DataStorage+"/" + RefData + "/St" + str(StNr)) if (f[-3:]=="pkl")])

def TTempCalc(FilePath, ChNrs,NSims=5000):
    """
    FilePath= location of the files e.g. "/pnfs/iihe/rno-g/store/user/jstoffels/Jobs/GalacticNoise/Data/V_Yr22/St23/"
    RefData= Datafolder of reference data
    StNr= station number
    ChNrs= Channels for which to calculate matching TTemp
    """
    DataStoragePath="/pnfs/iihe/rno-g/store/user/jstoffels/Jobs/GalacticNoise/Data/"
    ChNrs=[int(ChNr) for ChNr in ChNrs.split(',')]
    TTemps={ChNr:[150,290] for ChNr in ChNrs}
    FileNames=[f for f in listdir(DataStoragePath+FilePath) if (f[-3:]=="pkl" and int(f.split('_')[1].split('-')[0][6:8]) in ChNrs)]
    print("FileNames:",FileNames)
    DataVRMS={}
    for FileName in FileNames:
        GNData=pd.read_pickle(DataStoragePath+FilePath+FileName)
        DataVRMS[int(FileName.split('_')[1].split('-')[0][6:8])]=np.median(GNData['VRMS'])
        del GNData

    #Prepare the datapoints to simulate
    GNSim=pd.read_pickle(DataStoragePath+FilePath+FileNames[0])
    GNSim=GNSim[int(len(GNSim)/2 - NSims/2):int(len(GNSim)/2 + NSims/2)]
    GNSim=GNSim.assign(Unix=np.linspace(GNSim.iloc[0]['Unix'],GNSim.iloc[0]['Unix']+24*3600,NSims))

    #Initial VRMS vals
    #SimVRMS={13: [np.float64(0.0015434738770192828), np.float64(0.0019067007175157997)],16: [np.float64(0.0013675127610066637), np.float64(0.0013686593837145768)]}#{}
    print("Initial temp simulation")
    SimVRMS={}
    #In case you want to simulate different stations.
    # for FileName in FileNames:
    #     StNr,ChNr=int(FileName.split('-')[1][2:4]),int(FileName.split('-')[1][6:8])
    #     PDResultDicT0=GalacticNoiseVRMSCurve(StNr,[ChNr],GNSim,FFTFilter=True,
    #                                     Lowpass=True,ZeroAvg=True, ThermalNoiseT={ChNr:TTemps[ChNr][0] for ChNr in ChNrs},Plot=False,json="")
    #     PDResultDicT1=GalacticNoiseVRMSCurve(StNr,[ChNr],GNSim,FFTFilter=True,
    #                                     Lowpass=True,ZeroAvg=True, ThermalNoiseT={ChNr:TTemps[ChNr][1] for ChNr in ChNrs},Plot=False,json="")
    #     SimVRMS[int(FileName.split('-')[1][6:8])]=[np.mean(PDResultDicT0[str(ChNr)]['VRMS']),np.mean(PDResultDicT1[str(ChNr)]['VRMS'])]
    #     del PDResultDicT0,PDResultDicT1

    StNr=int(FileName.split('_')[1].split('-')[0][2:4])
    print('StNr test:',StNr)
    
    PDResultDicT0=GalacticNoiseVRMSCurve(StNr,ChNrs,GNSim,FFTFilter=True,
                                         Lowpass=True,ZeroAvg=True, ThermalNoiseT={ChNr:TTemps[ChNr][0] for ChNr in ChNrs},Plot=False,json="")
    PDResultDicT1=GalacticNoiseVRMSCurve(StNr,ChNrs,GNSim,FFTFilter=True,
                                         Lowpass=True,ZeroAvg=True, ThermalNoiseT={ChNr:TTemps[ChNr][1] for ChNr in ChNrs},Plot=False,json="")
    for ChNr in ChNrs:
        SimVRMS[ChNr]=[np.mean(PDResultDicT0[str(ChNr)]['VRMS']),np.mean(PDResultDicT1[str(ChNr)]['VRMS'])]

    del PDResultDicT0,PDResultDicT1
    
    for ChNr in ChNrs:
        print('')
        print('Ch',ChNr)
        print('Temps=',TTemps[ChNr],'SimVRMS=',SimVRMS[ChNr])
        print('Data=',DataVRMS[ChNr])
    
    print('Iterate:')
    for i in range(5):
        print('')
        print(15*'=')
        print('Iteration',i)
        print(15*'=')
        # for FileName in FileNames:
        # print('')
        # StNr,ChNr=int(FileName.split('-')[1][2:4]),int(FileName.split('-')[1][6:8])
        # print('St',StNr,', Ch',ChNr) # {str(ChNr):[150,290] for ChNr in ChNrs.split(',')}
        Rico={ChNr:(SimVRMS[ChNr][1]-SimVRMS[ChNr][0])/(TTemps[ChNr][1]-TTemps[ChNr][0]) for ChNr in ChNrs}
        T={ChNr:TTemps[ChNr][1]-(SimVRMS[ChNr][1]-DataVRMS[ChNr])/Rico[ChNr] for ChNr in ChNrs}
        # print(Rico)
        # print(T)
        # print(ChNrs)
        StNr=23
        PDResultDicT=GalacticNoiseVRMSCurve(StNr,ChNrs,GNSim,FFTFilter=True,
                                    Lowpass=True,ZeroAvg=True, ThermalNoiseT=T,Plot=False,json="")
        VRMS={ChNr:np.mean(PDResultDicT[str(ChNr)]['VRMS']) for ChNr in ChNrs}
        for ChNr in ChNrs:
            print('')
            print('Ch',ChNr)
            print('Temps=',TTemps[ChNr],'SimVRMS=',SimVRMS[ChNr])
            print('T=',np.round(T[ChNr],2),'VRMS=',np.round(1e3*VRMS[ChNr],3),'mV; Data=',np.round(1e3*DataVRMS[ChNr],3),"mV")
            TTemps[ChNr][1],SimVRMS[ChNr][1]=T[ChNr],VRMS[ChNr]
        # print('TTemps:',TTemps)
        # print('VRMS:',SimVRMS)
        del T,VRMS, Rico
    print(15*'=')
    print('Results')
    print(15*'=')
    print('TTemps:',TTemps)
    return {ChNr:TTemps[ChNr][1] for ChNr in ChNrs}

#TTempCalc("V_Yr22/St23/",'13,16')