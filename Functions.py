# # Description:
# This Jupyter notebook contains all python functions developed for this work in order
# to allow for easy importing. The common functions are organised per notebook in which they were first used.

# # Import modules:

import numpy as np
#import scipy as sc
import matplotlib.pyplot as plt
import os
#import scipy.integrate
#from scipy.optimize import curve_fit
#%matplotlib widget


# # Define Constants:

#DataPath="/pnfs/iihe/rno-g/data" #For the systematically updated rno-g data
DataPath="/pnfs/iihe/rno-g/data" #For the handcarry data
PNFSPath="/pnfs/iihe/rno-g/store/user/jstoffels/Jobs"
# ## Firmware changes run number

# Source: following message from slack group chat  rno-g ops <br>
# avinngaq is on station 22, nanoq is on station 11 and amaroq is on station 21

# Define a dictionary for easy reference of when each station started using the new firmware

FirmwareSwitch = { #Runs for each station that start using the new firmware
  "St11": 571,
  "St21": 753,
  "St22": 656
}

def TimeTrace(StNr,ChNr,Run,EvNr,Amplitude="V"):
    """Show the timetrace of Station StNr, channel ChNr for run Run, event EvNr. Units of Ampllitude can be V,mV or ADC"""
    path=Path(StNr,Run)
    if os.path.isfile(path+"/combined.root") and os.path.isfile(path+"/daqstatus.root"):   
        CombinedFile=GetCombinedFile(StNr,Run)
        RadiantData=CombinedFile['combined']['waveforms']['radiant_data[24][2048]'].array(library='np')
        EventNrs=CombinedFile['combined']['waveforms']['event_number'].array(library="np")
        #TriggerTimes=CombinedFile['combined']['header']["trigger_time"].array(library='np')
        EvIndex=np.where(EventNrs==EvNr)[0][0]
    elif os.path.isfile(path+"/waveforms.root"):
        WaveFormFile=GetWaveformsFile(StNr,Run)
        EventNrs=WaveFormFile['event_number'].array(library="np")
        EvIdx=np.where(EventNrs==EvNr)[0][0]
        RadiantData=WaveFormFile['waveforms']['radiant_data[24][2048]'].array(entry_start=EvIdx, entry_stop=EvIdx+1,library='np')
        #RadiantData=WaveFormFile['radiant_data[24][2048]'].array(library='np')
        EvIndex=0
    else:
        print("Root files not present")
        return
    
    if not EvNr in EventNrs:
        print("There is no event with this number")
        return
        
    

    sampling_rate=3.2 * (10**9) #Sampling rate in Hertz according to the python file of NuRadioReco.modules.io.rno_g
    TimeStep=1/sampling_rate #Time between two samples
    SamplingTimes=np.arange(0,len(RadiantData[0][0])*TimeStep,TimeStep)
    
    plt.figure(figsize=(20,5))
    
    if Amplitude=="V":
        plt.plot(SamplingTimes*(10**9),ADCtoVoltage(RadiantData[EvIndex][ChNr]),'-', label="Channel " + str(ChNr))
        plt.ylabel("Amplitude (V)",fontsize=40)#20)
    elif Amplitude=="mV":
        plt.plot(SamplingTimes*(10**9),1000*ADCtoVoltage(RadiantData[EvIndex][ChNr]),'-', label="Channel " + str(ChNr))
        plt.ylabel("Amplitude (mV)",fontsize=40)#20)
    elif Amplitude=="ADC":
        plt.plot(SamplingTimes*(10**9),RadiantData[EvIndex][ChNr],'-', label="Channel " + str(ChNr))
        plt.ylabel("Amplitude (ADC)",fontsize=40)#20)        
    #plt.plot(Energies,TritonEnergyLoss,'-',color='r', label="Triton")
    plt.title("Time trace of St" + str(StNr) + ", Ch" + str(ChNr) + ", Run " + str(Run) + ", Evt " + str(EvNr),fontsize=50)#25)
    #plt.ylim(-50,50)
    plt.xlim(0,np.max(SamplingTimes*(10**9)))
    plt.xlabel("Time (ns)",fontsize=40)#20)

    plt.xticks(fontsize=30)#15)
    plt.yticks(fontsize=30)#15)
    #plt.legend()
    plt.show()
    return

def TimeTraceFFT(StNr,ChNr,Run,EvNr,Amplitude="V",LogScale=False):
    """Show the FFT of the timetrace of Station StNr, channel ChNr for run Run, event EvNr. Units of Ampllitude can be V,mV or ADC"""
    import scipy.fft as scfft
    path=Path(StNr,Run)
    if os.path.isfile(path+"/combined.root") and os.path.isfile(path+"/daqstatus.root"):
        
        CombinedFile=GetCombinedFile(StNr,Run)
        RadiantData=CombinedFile['combined']['waveforms']['radiant_data[24][2048]'].array(library='np')
        EventNrs=CombinedFile['combined']['waveforms']['event_number'].array(library="np")
        #TriggerTimes=CombinedFile['combined']['header']["trigger_time"].array(library='np')
        EvIndex=np.where(EventNrs==EvNr)[0][0]
    elif os.path.isfile(path+"/waveforms.root"):
        WaveFormFile=GetWaveformsFile(StNr,Run)
        EventNrs=WaveFormFile['event_number'].array(library="np")
        EvIdx=np.where(EventNrs==EvNr)[0][0]
        RadiantData=WaveFormFile['waveforms']['radiant_data[24][2048]'].array(entry_start=EvIdx, entry_stop=EvIdx+1,library='np')
        #RadiantData=WaveFormFile['radiant_data[24][2048]'].array(library='np')
        EvIndex=0
    else:
        print("Root files not present")
        return
    
    if not EvNr in EventNrs:
        print("There is no event with this number")
        return
        
    sampling_rate=3.2 * (10**9) #Sampling rate in Hertz according to the python file of NuRadioReco.modules.io.rno_g
    TimeStep=1/sampling_rate #Time between two samples
    SamplingTimes=np.arange(0,len(RadiantData[0][0])*TimeStep,TimeStep)
    freq=scfft.fftfreq(len(SamplingTimes),(SamplingTimes[-1]-SamplingTimes[0])/len(SamplingTimes))
    freq=np.fft.fftshift(freq)
    
    plt.figure(figsize=(20,5))
    
    if Amplitude=="V":
        FFT=scfft.fft(ADCtoVoltage(RadiantData[EvIndex][ChNr]))
        FFT=np.fft.fftshift(FFT)
        plt.plot((freq*(10**-6))[int(len(freq)/2)+1:len(freq)],(2/len(SamplingTimes))*np.abs(FFT)[int(len(FFT)/2)+1:len(FFT)],'-', label="Channel " + str(ChNr))
        plt.ylabel("Amplitude (V)",fontsize=40)#20)
    elif Amplitude=="mV":
        FFT=scfft.fft(1000*ADCtoVoltage(RadiantData[EvIndex][ChNr]))
        FFT=np.fft.fftshift(FFT)
        plt.plot((freq*(10**-6))[int(len(freq)/2)+1:len(freq)],(2/len(SamplingTimes))*np.abs(FFT)[int(len(FFT)/2)+1:len(FFT)],'-', label="Channel " + str(ChNr))
        plt.ylabel("Amplitude (mV)",fontsize=40)#20)
    elif Amplitude=="ADC":
        FFT=scfft.fft(RadiantData[EvIndex][ChNr])
        FFT=np.fft.fftshift(FFT)
        plt.plot((freq*(10**-6))[int(len(freq)/2)+1:len(freq)],(2/len(SamplingTimes))*np.abs(FFT)[int(len(FFT)/2)+1:len(FFT)],'-', label="Channel " + str(ChNr))
        plt.ylabel("Amplitude (ADC)",fontsize=40)#20)  
    #plt.plot(Energies,TritonEnergyLoss,'-',color='r', label="Triton")
    plt.title("FFT of Time trace of St" + str(StNr) + ", Ch" + str(ChNr) + ", Run " + str(Run) + ", Evt " + str(EvNr),fontsize=50)#25)
    #plt.ylim(-50,50)
    plt.xlim(0,np.max(freq*(10**-6)))
    if LogScale:
        plt.yscale("log")
    #plt.xlim(400,450)
    plt.xlabel("Frequency (MHz)",fontsize=40)#20)

    plt.xticks(fontsize=30)#15)
    plt.yticks(fontsize=30)#15)
    #plt.legend()
    plt.show()

def Path(StNr,RunNr):
    """Returns path to the datafiles for station StNr, run RunNr."""
    global DataPath
    if os.path.isdir(DataPath + "/handcarry/station"+str(StNr)+"/run" + str(RunNr)):
        return DataPath + "/handcarry/station{}/run{}".format(StNr, int(RunNr))
    # print("Did not find a run with run number " + str(RunNr) + " for station " + str(StNr))
    return
            

def FilesStRun(StNr,RunNr):
    """Returns the daqstatus -, headers - & pedestal datafiles for run RunNR of station StNr"""
    import uproot
    path=Path(StNr,RunNr)
    DAQStatFile=uproot.open(path+"/daqstatus.root")
    HeadersFile=uproot.open(path+"/headers.root")
    PedestalFile=uproot.open(path+"/pedestal.root")
    WaveformsFile=uproot.open(path+"/waveforms.root")
    RunInfo=uproot.open(path+"/runinfo.root")
    return WaveformsFile, DAQStatFile, HeadersFile, PedestalFile, RunInfo

def GetCombinedFile(StNr,RunNr):
    """Returns the combined datafile for run RunNR of station StNr"""
    import uproot
    path=Path(StNr,RunNr)
    #print(path+"/combined.root")
    return uproot.open(path+"/combined.root")

def GetPedestalFile(StNr,RunNr):
    """Returns the pedestal datafile for run RunNR of station StNr"""
    import uproot
    path=Path(StNr,RunNr)
    return uproot.open(path+"/pedestal.root")

def GetWaveformsFile(StNr,RunNr):
    """Returns the waveforms datafile for run RunNR of station StNr"""
    import uproot
    path=Path(StNr,RunNr)
    WaveFormFile=uproot.open(path+"/waveforms.root")
    Key=None
    try:
        WaveFormFile['waveforms']
    except:
        None
    else:
        Key='waveforms'
    try:
        WaveFormFile['wf']
    except:
        None
    else:
        Key='wf'
    if Key==None:
        print("NonStandard Key in run",RunNr,", options are:")
        print(WaveFormFile.keys())
        return
    return WaveFormFile[Key]

def GetHeaderFile(StNr,RunNr):
    """Returns the header datafile for run RunNR of station StNr"""
    import uproot
    path=Path(StNr,RunNr)
    if path==None:
        return
    if not os.path.isfile(path+"/headers.root"):
        return None
    HeaderFile=uproot.open(path+"/headers.root")
    Key=None
    try:
        HeaderFile['header']
    except:
        None
    else:
        Key='header'
    try:
        HeaderFile['hdr']
    except:
        None
    else:
        Key='hdr'
    if Key==None:
        print("NonStandard Key in run",RunNr,", options are:")
        print(HeaderFile.keys())
        return
    return HeaderFile[Key]

def ADCtoVoltage(ADCCounts):
    """Converts the ADC counts ADCCounts to Volt."""
    ADC_Factor=0.000618
    ADC_Offset=-0.008133 #Set this to zero for now, otherwise the FFT becomes extremely weird
    return (ADC_Factor*ADCCounts + ADC_Offset)

def RunListRunSum(StNr):
    """Constructs a runlist from the zeuthen run summary with runs that have no comments"""
    import pandas as pd
    RunSum=pd.read_csv('rnog_run_summary.csv')
    NpRuns=np.hstack(RunSum[(RunSum.station==StNr) & (RunSum.comment.isnull())][["run"]].to_numpy(dtype=int))
    return NpRuns

# def RunList(StNr,Date0=None,Date1=None,Comments=False):
#     """Constructs a runlist between dates Date0 and Date1 with runs that have no comments if Comments was set to False"""
#     import pandas as pd
#     from datetime import datetime
#     T=datetime.utcfromtimestamp(UnixTime)
#     RunSum=pd.read_csv('rnog_run_summary.csv')
#     NpRuns=np.hstack(RunSum[(RunSum.station==StNr) & (RunSum.comment.isnull())][["run"]].to_numpy(dtype=int))
#     return NpRuns

def RunList(StNr,T0,T1,AllowComments=False):
    """Construct a run list of runs between two dates."""
    Run0,Run1=RunListMinMax(StNr,T0,T1)
    if AllowComments:
        return np.arange(Run0,Run1+1,dtype=int)
    else:
        # path=Path(StNr,RunNr)
        return np.array([Run for Run in np.arange(Run0,Run1+1,dtype=int) if Path(StNr,Run)!=None and (os.path.getsize(Path(StNr,Run) + '/aux/comment.txt')==0)])
    
def RunListMinMax(StNr,T0,T1):
    """Returns a list of runs between the two specified times (datetime object). Helper function for the RunList function."""
    import datetime
    import time
    
    MaxYear=23
    #print([RunNr[3:].isdigit() for RunNr in os.listdir(DataPath + "/handcarry/station" + str(StNr))])
    MaxRuns=max([int(RunNr[3:]) for RunNr in os.listdir(DataPath + "/handcarry/station" + str(StNr)) if RunNr[3:].isdigit()])
    
    T0,T1 =  time.mktime(T0.timetuple()), time.mktime(T1.timetuple())
    
    X0,X1=1,MaxRuns
    for i in range(int(np.ceil(np.log2(X1)))+1):
        Xmid=int((X1+X0)/2)
        # print(X0,Xmid,X1)
        if Xmid==X0 or Xmid==X1:# or X1-X0<=2:
            # if Xmid==0:
            #     Result0=X1
            #     break
            HeaderFile=GetHeaderFile(StNr,X0)
            if HeaderFile==None:
                Result0=X1
                break
            TX0=HeaderFile["trigger_time"].array(entry_start=0, entry_stop=1,library='np')[0]
            if T0<TX0:
                Result0=X0
            else:
                Result0=X1
            break
        HeaderFile=GetHeaderFile(StNr,Xmid)
        if (HeaderFile==None or len(HeaderFile["trigger_time"].array(entry_start=0, entry_stop=1,library='np'))==0 or np.isnan(HeaderFile["trigger_time"].array(entry_start=0, entry_stop=1,library='np')[0])):
            while HeaderFile==None or len(HeaderFile["trigger_time"].array(entry_start=0, entry_stop=1,library='np'))==0 or np.isnan(HeaderFile["trigger_time"].array(entry_start=0, entry_stop=1,library='np')[0]):
                Xmid-=1
                HeaderFile=GetHeaderFile(StNr,Xmid)
            if Xmid<=X0:
                Xmid=int((X1+X0)/2)
                while HeaderFile==None or len(HeaderFile["trigger_time"].array(entry_start=0, entry_stop=1,library='np'))==0 or np.isnan(HeaderFile["trigger_time"].array(entry_start=0, entry_stop=1,library='np')[0]):
                    Xmid+=1
                    HeaderFile=GetHeaderFile(StNr,Xmid)
                if Xmid>=X1:
                    print("Loop got stuck around X0,X1=",X0,X1)
                    return
        TX0=HeaderFile["trigger_time"].array(entry_start=0, entry_stop=1,library='np')[0]
        if TX0<=T0:
            X0,X1=Xmid,X1
            continue
        elif TX0>T0:
            X0,X1=X0,Xmid
        else:
            print("Error")

    X0,X1=1,MaxRuns
    for i in range(int(np.ceil(np.log2(X1)))+1):
        Xmid=int((X1+X0)/2)
        # print(X0,Xmid,X1)
        if Xmid==X0 or Xmid==X1:# or X1-X0<=2:
            HeaderFile=GetHeaderFile(StNr,X1)
            if HeaderFile==None:
                Result1=X0
                break
            TX1=HeaderFile["trigger_time"].array(entry_start=0, entry_stop=1,library='np')[0]
            if T1<TX1:
                Result1=X0
            else:
                Result1=X1
            break
        HeaderFile=GetHeaderFile(StNr,Xmid)
        if HeaderFile==None:
            while HeaderFile==None:
                Xmid-=1
                HeaderFile=GetHeaderFile(StNr,Xmid)
            if Xmid<=X0:
                Xmid=int((X1+X0)/2)
                while HeaderFile==None or len(HeaderFile["trigger_time"].array(entry_start=0, entry_stop=1,library='np'))==0:
                    Xmid+=1
                    HeaderFile=GetHeaderFile(StNr,Xmid)
                if Xmid>=X1:
                    print("Loop got stuck around X0,X1=",X0,X1)
                    return
        TX1=HeaderFile["trigger_time"].array(entry_start=0, entry_stop=1,library='np')[0]
        if TX1<T1:
            X0,X1=Xmid,X1
            continue
        elif TX1>T1:
            X0,X1=X0,Xmid
            
    return Result0,Result1