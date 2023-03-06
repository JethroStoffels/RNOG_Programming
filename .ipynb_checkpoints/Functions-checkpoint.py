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

DataPath="/pnfs/iihe/rno-g/data"


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
    
    else:
        return
    
    if not EvNr in EventNrs:
        print("There is no event with this number")
        return
        
    
    EvIndex=np.where(EventNrs==EvNr)[0][0]
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

def TimeTraceFFT(StNr,ChNr,Run,EvNr,Amplitude="V",LogScale=False):
    """Show the FFT of the timetrace of Station StNr, channel ChNr for run Run, event EvNr. Units of Ampllitude can be V,mV or ADC"""
    import scipy.fft as scfft
    path=Path(StNr,Run)
    if os.path.isfile(path+"/combined.root") and os.path.isfile(path+"/daqstatus.root"):
        
        CombinedFile=GetCombinedFile(StNr,Run)
        RadiantData=CombinedFile['combined']['waveforms']['radiant_data[24][2048]'].array(library='np')
        EventNrs=CombinedFile['combined']['waveforms']['event_number'].array(library="np")
        #TriggerTimes=CombinedFile['combined']['header']["trigger_time"].array(library='np')
    
    else:
        return
    
    if not EvNr in EventNrs:
        print("There is no event with this number")
        return
        
    EvIndex=np.where(EventNrs==EvNr)[0][0]
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

def Path(StNr,RunNr,System="U"):
    """Returns path to the datafiles for station StNr, run RunNr. Path differs for OS Unix (U) or Windows (W)."""
    global DataPath
    if System=="U":
        #"/mnt/c/Users/Jethro/Desktop/Master thesis/RNO_DATA_DIR/station11/run101/combined.root"
        return DataPath + "/station{}/run{}".format(StNr, RunNr)
    elif System=="W":
        #"/mnt/c/Users/Jethro/Desktop/Master thesis/RNO_DATA_DIR/station11/run101/combined.root"
        return DataPath + r"\station{}\run{}".format(StNr, RunNr)

def FilesStRun(StNr,RunNr):
    """Returns the combined -, daqstatus -, headers - & pedestal datafiles for run RunNR of station StNr"""
    import uproot
    path=Path(StNr,RunNr)
    CombinedFile=uproot.open(path+"/combined.root")
    DAQStatFile=uproot.open(path+"/daqstatus.root")
    HeadersFile=uproot.open(path+"/headers.root")
    PedestalFile=uproot.open(path+"/pedestal.root")
    return CombinedFile, DAQStatFile, HeadersFile, PedestalFile

def GetCombinedFile(StNr,RunNr):
    """Returns the combined datafile for run RunNR of station StNr"""
    import uproot
    path=Path(StNr,RunNr)
    return uproot.open(path+"/combined.root")

def GetPedestalFile(StNr,RunNr):
    """Returns the pedestal datafile for run RunNR of station StNr"""
    import uproot
    path=Path(StNr,RunNr)
    return uproot.open(path+"/pedestal.root")

def ADCtoVoltage(ADCCounts):
    """Converts the ADC counts ADCCounts to Volt."""
    ADC_Factor=0.000618
    ADC_Offset=-0.008133 #Set this to zero for now, otherwise the FFT becomes extremely weird
    return (ADC_Factor*ADCCounts + ADC_Offset)
