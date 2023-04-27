# # Description:
# This Jupyter notebook contains all python functions developed for the galactic noise study in order
# to allow for easy importing. The common functions are organised per notebook in which they were first used.

# # Import modules:
from Functions import *
#%matplotlib widget

# # Galaxy Stacking:



def ADCtoVoltageTemp(ADCCounts):
    """Temporary reimplementation of the ADC to Voltage conversion. Do not use this function!"""
    ADC_Factor=0.618
    ADC_Offset=-8.133 
    return (ADC_Factor*ADCCounts + ADC_Offset)

def LST(TriggerTimes,EvIdx):
    """Computes the Local Sidereal Time in decimal hours via the astropy module"""
    from datetime import datetime
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    from astropy import units as u
    observing_location = EarthLocation(lat=72.598265*u.deg, lon=-38.459936*u.deg)
    observing_time = Time(datetime.utcfromtimestamp(TriggerTimes[EvIdx]), scale='utc', location=observing_location)
    T=observing_time.sidereal_time('mean')
    return T.hour

def UTC(TriggerTimes,EvIdx):
    """Computes the UTC time in decimal hours"""
    from datetime import datetime
    T=datetime.utcfromtimestamp(TriggerTimes[EvIdx])
    return T.hour+T.minute/60 + T.second/3600

def LT(TriggerTimes,EvIdx):
    return (UTC(TriggerTimes,EvIdx)-2)%24

def TransitCurve(StNr,ChNr,Runs,NBins=24,ZeroAvg=True,TimeFormat="LST",Triggers=(5,5,5,5),StdCut=(-1,-1),FFTFilter=True,Lowpass=False,Plot=True):
    """
    Plots the Average V_RMS as a function of time of the day.
    Parameters:
    StNr,ChNr,Runs=Station number, channel number, list of runs 
    NBins=amount of bins to divide the full day in
    ZeroAvg=Boolean: if true, the timetraces will firs tbe zero averaged
    Lowpass= Boolean: if true, a butterworth lowpass filter will be applied in order to maintain only galactic noise dominated frequencies
    FFTFilter=Boolean: if true, applies a Notch filter to all frequency spectra to cut out frequencies which have shown to be potentially problematic
    TimeFormat= String: Dictates what timeformat the x-axis will be in. Options: "LST": local sidereal time, "LT": Local time & "UTC": UTC time
    Triggers=tupel of flags to dictate which triggers are allowed in the analysis. Events with different triggers are not used (0=has to be absent, 1=has to be present, anything else=both 0 and 1 can be used for analysis)
    StdCut=(AmtStd,StdCut) if larger than zero, all VRMS outliers above StdCut standard variations will be cut out of the analysis. This procedure is repeated AmtStd times.
    """
    (has_rf,has_ext,has_pps,has_soft)=Triggers
    NEvs=0
    EventRMS=np.array([]) #Array to store V_RMS value of each event
    EventTime=np.array([])#Array to store timestamp of each event
    #for (Run, EvNr) in TriggerFilterAlt(StNr, ChNr, Runs,has_rf,has_ext,has_pps,has_soft):
    FilteredRuns,FilteredEvNrs=TriggerFilterAlt(StNr, ChNr, Runs,has_rf,has_ext,has_pps,has_soft)
    for Run in FilteredRuns:
        path=Path(StNr,Run)
        #if os.path.isdir(path+"/combined.root"):
        if os.path.isfile(path+"/combined.root"):
            NEvs+=1
            #CombinedFile=GetCombinedFile(StNr,Runs[0])
            CombinedFile=GetCombinedFile(StNr,Run)
            RadiantData=CombinedFile['combined']['waveforms']['radiant_data[24][2048]'].array(library='np')
            EventNrs=CombinedFile['combined']['waveforms']['event_number'].array(library="np")
            TriggerTimes=CombinedFile['combined']['header']["trigger_time"].array(library='np')        
            
            RunIdx=np.where(FilteredRuns==Run)[0][0]

            for EvNr in FilteredEvNrs[RunIdx]:
                EvIdx=np.where(EventNrs==EvNr)[0][0]
                if np.isinf(TriggerTimes[EvIdx]):
                    print("Inf timestamp at: Run" + str(Run) + ", EvNr" + str(EvNr))
                    FilteredEvNrs[RunIdx]=np.delete(FilteredEvNrs[RunIdx],np.where(FilteredEvNrs[RunIdx]==EvNr)[0][0])
                    if len(FilteredEvNrs[RunIdx])==0:
                        FilteredRuns=np.delete(FilteredRuns, RunIdx)
                        FilteredEvNrs=np.delete(FilteredEvNrs, RunIdx)
                    continue
                VoltageTrace=ADCtoVoltage(RadiantData[EvIdx][ChNr]) #The timetrace data in voltage
                if ZeroAvg==True:
                    Vmean=np.mean(VoltageTrace)
                    VoltageTrace-=Vmean
                    #EventRMS=np.append(EventRMS,np.sqrt(np.mean([(V-Vmean)**2 for V in ADCtoVoltage(RadiantData[EvIdx][ChNr])])))
                if FFTFilter or Lowpass:
                    import scipy.fft as scfft
                    sampling_rate=3.2 * (10**9) #Sampling rate in Hertz according to the python file of NuRadioReco.modules.io.rno_g
                    TimeStep=1/sampling_rate #Time between two samples
                    SamplingTimes=np.arange(0,len(RadiantData[0][0])*TimeStep,TimeStep)
                    freq=scfft.fftfreq(len(SamplingTimes),(SamplingTimes[-1]-SamplingTimes[0])/len(SamplingTimes))
                    freq=np.fft.fftshift(freq)
                    TotalFilter=np.ones(len(freq))
                    if FFTFilter:
                        TotalFilter=np.multiply(TotalFilter,NotchFilters([403*10**6,120*10**6,807*10**6,1197*10**6],75,freq,sampling_rate))
                    if Lowpass:
                        CritFreq=110*10**6
                        TotalFilter=np.multiply(TotalFilter,LowpassButter(CritFreq,20,freq))
                    FFT=scfft.fft(VoltageTrace)
                    FFT=np.fft.fftshift(FFT)
                    FFT=np.array([FFT[i]*TotalFilter[i] for i in range(len(freq))])
                    VoltageTrace=np.abs(scfft.ifft(FFT))

                EventRMS=np.append(EventRMS,np.sqrt(np.mean([V**2 for V in VoltageTrace])))

                if TimeFormat=="LST":
                    EventTime=np.append(EventTime,LST(TriggerTimes,EvIdx))
                elif TimeFormat=="LT": #Greenland Timezone is UTC-3
                    EventTime=np.append(EventTime,LT(TriggerTimes,EvIdx))
                else:
                    print("Please enter a valid TimeFormat")
                    return


    if np.all(np.array(StdCut)>=0):
        for StdAmt in range(StdCut[0]):
            EventRMSStd=np.std(EventRMS)
            EventRMSMedian=np.median(EventRMS)
            for VRMS in EventRMS:
                if not EventRMSMedian - StdCut[1]*EventRMSStd<VRMS<EventRMSMedian + StdCut[1]*EventRMSStd:
                    #GroupedVRMS[i]=np.delete(GroupedVRMS[i], np.where(GroupedVRMS[i]==VRMS)[0][0])
                    FilteredRuns,FilteredEvNrs=StdCutRunEvtsFilter(np.where(EventRMS==VRMS)[0][0],FilteredRuns,FilteredEvNrs)
                    EventTime=np.delete(EventTime, np.where(EventRMS==VRMS)[0][0])
                    EventRMS=np.delete(EventRMS, np.where(EventRMS==VRMS)[0][0])
    #print(np.sum([EventRMS[i] for i in np.arange(len(EventTime)) if EventTime[i]<=.25 ]))
    
    EventTimeCounts, EventTimeBins=np.histogram(EventTime, bins=NBins,range=(0,24),density=False) #Storing timestamps in histogram format
    #Make a histogram of the V_RMS value fully added in its respective bin by adding V_RMS as a weigth to the additions to the histogram
    EventRMSCounts, EventRMSBins=np.histogram(EventTime, bins=NBins,range=(0,24),density=False,weights=EventRMS)    
    
    ##plt.hist(EventTime, bins=NBins,range=(0,24),density=False)
    
    ##plt.figure()
    ##plt.hist(EventTime, bins=NBins,range=(0,24),density=False, weights=EventRMS)
    
    #RMSBins=np.digitize(EventTime,EventTimeBins) #Array of idx of bin in which the timestamp of each event falls
    
    ##Make a histogram of the V_RMS value fully added in its respective bin by adding V_RMS as a weigth to the additions to the histogram
    #EventRMSCounts, EventRMSBins=np.histogram(RMSBins, bins=24,range=(0,24),density=False,weights=EventRMS)
    
    
    #Compute std per bin
    EventTimeDig=np.digitize(EventTime,EventTimeBins)
    GroupedVRMS=np.empty((NBins,),dtype=object)
    for i in range(len(EventTimeDig)):
        GroupedVRMS[EventTimeDig[i]-1]=np.append(GroupedVRMS[EventTimeDig[i]-1],EventRMS[i])
    ##Get rid of "None" entries in beginning of array
    for i in range(len(GroupedVRMS)):
        GroupedVRMS[i]=np.delete(GroupedVRMS[i], 0) 
    
    VRMSStd=np.array([np.std(GroupedVRMS[i]) if len(GroupedVRMS[i])!=0 else 0 for i in range(len(GroupedVRMS))])
    VRMSMedian=np.array([np.median(GroupedVRMS[i]) for i in range(len(GroupedVRMS))])

    
    MidBins= np.array([(EventTimeBins[i] + EventTimeBins[i+1])/2 for i in range(0,len(EventTimeBins)-1)])           
    VRMSAvg=np.array([EventRMSCounts[i]/EventTimeCounts[i]  if EventTimeCounts[i] !=0 else 0 for i in range(len(EventRMSCounts))])
    
    #if StdCut>=0:
    #    for i in range(len(GroupedVRMS)):
    #        for VRMS in GroupedVRMS[i]:
    #            if VRMS>VRMSMedian[i] + StdCut*VRMSStd[i] or VRMS<VRMSMedian[i] - StdCut*VRMSStd[i]:
    #                GroupedVRMS[i]=np.delete(GroupedVRMS[i], np.where(GroupedVRMS[i]==VRMS)[0][0])
    #    VRMSAvg=np.array([np.mean(GroupedVRMS[i]) for i in range(len(GroupedVRMS))])
    #    VRMSStd=np.array([np.std(GroupedVRMS[i]) if len(GroupedVRMS[i])!=0 else 0 for i in range(len(GroupedVRMS))]) 
    if Plot:
        plt.figure(figsize=(15,5))
        plt.figtext(0.2, 0.8, "Entries:" + str(np.sum(EventTimeCounts)), fontsize=18,bbox=dict(edgecolor='black', facecolor='none', alpha=0.2, pad=10.0))
        #plt.hist(RMSBins, bins=24,range=(0,24),density=False, weights=[EventRMS[i]/EventRMSCounts[i] for i in range(len(EventRMS))])
        plt.errorbar(MidBins,1000*VRMSAvg,yerr=1000*VRMSStd,fmt=".",zorder=2)
        #for i in range(len(GroupedVRMS)):
        #    plt.plot(MidBins[i]*np.ones(len(GroupedVRMS[i])),1000*GroupedVRMS[i],"r.", alpha=0.5,zorder=1)

        #plt.plot(MidBins,1000*VRMSAvg,'r.')
        plt.grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
        plt.title("V_RMS of Station " + str(StNr) + ", channel " + str(ChNr) + " for " + str(NEvs) + " events between run " + str(Runs[0]) + " and run " + str(Runs[-1]) + " throughout the day for " + str(NBins) + " bins")
        #plt.ylim(-50,50)
        #plt.xlim(0,np.max(SamplingTimes*(10**9)))
        plt.xlabel(TimeFormat + " Time (hrs)",fontsize=20)#20)
        plt.ylabel("V_RMS (mV)",fontsize=20)#20)
        plt.xticks(np.arange(0, 24, 1.0),fontsize=25)#15)
        plt.yticks(fontsize=25)#15)
        #plt.legend()
        plt.show()
    return MidBins, GroupedVRMS, FilteredRuns,FilteredEvNrs

def StdCutRunEvtsFilter(EvRMSIdx,Runs,EvIdxs):
    PastElements=0
    for RunIdx in range(len(Runs)):
        if PastElements + len(EvIdxs[RunIdx])>EvRMSIdx:
            SubEvIdx=EvRMSIdx-PastElements-1
            EvIdxs[RunIdx]=np.delete(EvIdxs[RunIdx],SubEvIdx)
            if len(EvIdxs[RunIdx])==0:
                Runs=np.delete(Runs, RunIdx)
                EvIdxs=np.delete(EvIdxs, RunIdx)
            return Runs, EvIdxs
        else:
            PastElements+=len(EvIdxs[RunIdx])
            continue
    return

def InfTimestampFilter(Runs,EventNrs):
    return

def VRMSTimeOfDay(StNr,ChNr,Runs,t0,t1,TimeFormat="LST"):
    """Plots the VRMS of each event of station StNr, channel ChNr during runs Runs which occur during times of the day t0 and t1 (t0<t1 & t0,t1 in [0,24]), given in timeformat TimeFormat (can be local sidereal time LST or local time LT)."""
    EventRMS=np.array([]) #Array to store V_RMS value of each event
    EventTime=np.array([])#Array to store timestamp of each event
    for Run in Runs:
        path=Path(StNr,Run)
        #if os.path.isdir(path+"/combined.root"):
        if os.path.isfile(path+"/combined.root"):

            CombinedFile=GetCombinedFile(StNr,Run)
            RadiantData=CombinedFile['combined']['waveforms']['radiant_data[24][2048]'].array(library='np')
            EventNrs=CombinedFile['combined']['waveforms']['event_number'].array(library="np")
            TriggerTimes=CombinedFile['combined']['header']["trigger_time"].array(library='np')  
            
            for EvNr in EventNrs:
                EvIdx=np.where(EventNrs==EvNr)[0][0]
                if TimeFormat=="LST":
                    t=LST(TriggerTimes,EvIdx)
                elif TimeFormat=="LT":
                    t=LT(TriggerTimes,EvIdx)
                else:
                    print("TimeFormat " + str(TimeFormat) + " is not supported.")
                
                if t0<=t<=t1:
                    Vmean=np.mean(ADCtoVoltage(RadiantData[EvIdx][ChNr]))
                    EventRMS=np.append(EventRMS,np.sqrt(np.mean([(V-Vmean)**2 for V in ADCtoVoltage(RadiantData[EvIdx][ChNr])])))

    #plt.figure(figsize=(15,5))
    plt.plot(np.arange(len(EventRMS)),1000*EventRMS,".")
    plt.title(r"$V_{RMS}$ for the events of Channel" + str(ChNr) + " for runs between " + str(t0) + " & " + str(t1) + " hr " + TimeFormat)
    plt.xlabel("Sample Nr")
    plt.ylabel(r"$V_{RMS}$ (mV)")
    plt.show()

def EvntsPerRun(StNr,ChNr,Runs):
    """Plots the amount of events per run for station StNr, channel ChNr during runs Runs."""
    NRuns=0
    AmountEvents=np.array([])
    ValidRuns=np.array([])
    for Run in Runs:
        path=Path(StNr,Run)
        #if os.path.isdir(path+"/combined.root"):
        if os.path.isfile(path+"/combined.root"):
            NRuns+=1
            
            CombinedFile=GetCombinedFile(StNr,Run)
            RadiantData=CombinedFile['combined']['waveforms']['radiant_data[24][2048]'].array(library='np')
            EventNrs=CombinedFile['combined']['waveforms']['event_number'].array(library="np")
            #TriggerTimes=CombinedFile['combined']['header']["trigger_time"].array(library='np')        
            AmountEvents=np.append(AmountEvents,len(EventNrs))  
            ValidRuns=np.append(ValidRuns,Run)  
            
    plt.figure()
    plt.plot(ValidRuns,AmountEvents,".")
    plt.figtext(0.2, 0.8, "Entries:" + str(np.sum(AmountEvents)), fontsize=18,bbox=dict(edgecolor='black', facecolor='none', alpha=0.2, pad=10.0))
    plt.xlabel("Run Nr")
    plt.ylabel("Amount of events")
    plt.title("Amount of events per run for Station " + str(StNr) + ", channel " + str(ChNr))
    plt.yscale("log")
    plt.show()

def TrigInfo(StNr,ChNr,RunNr):
    """Reads in the TriggerInfo databranch"""
    CombinedFile=GetCombinedFile(StNr,RunNr)
    TriggerInfo=CombinedFile['combined']['header']['trigger_info'].array(library='np')
    return TriggerInfo


# # Investigating Transit Curves:

def TransitCurves(StNr,ExtraChNr,Runs,NBins=24,ZeroAvg=True,TimeFormat="LST"):
    """Plots the Vrms values as a function of LST for the upwards facing LPDA's and a chosen channel. This function is deprecated, use TransitCurve instead!"""
    ChNrs=[13,16,19,ExtraChNr]
    NRuns=0
    EventRMS= np.empty((4,0),float) #Array to store V_RMS value of each event for all four channels
    EventTime=np.array([])#Array to store timestamp of each event
    for Run in Runs:
        path=Path(StNr,Run)
        #if os.path.isdir(path+"/combined.root"):
        if os.path.isfile(path+"/combined.root"):
            NRuns+=1
        
            CombinedFile=GetCombinedFile(StNr,Run)
            RadiantData=CombinedFile['combined']['waveforms']['radiant_data[24][2048]'].array(library='np')
            EventNrs=CombinedFile['combined']['waveforms']['event_number'].array(library="np")
            TriggerTimes=CombinedFile['combined']['header']["trigger_time"].array(library='np')        
                
            for EvNr in EventNrs:
                EvIdx=np.where(EventNrs==EvNr)[0][0]
                #EventRMS=np.append(EventRMS,np.sqrt(np.mean([V**2 for V in ADCtoVoltage(RadiantData[EvIdx][ChNr])])))
                if ZeroAvg==True:
                    VmeanList=[np.mean(ADCtoVoltage(RadiantData[EvIdx][ChNr])) for ChNr in ChNrs]
                    VRMSList=[[np.sqrt(np.mean([(V-VmeanList[i])**2 for V in ADCtoVoltage(RadiantData[EvIdx][ChNrs[i]])]))] for i in range(len(ChNrs))]
                else:
                    VRMSList=[[np.sqrt(np.mean([V**2 for V in ADCtoVoltage(RadiantData[EvIdx][ChNr])]))] for ChNr in ChNrs ]
                EventRMS=np.concatenate((EventRMS,np.array(VRMSList)),axis=1)
                
                if TimeFormat=="LST":
                    EventTime=np.append(EventTime,LST(TriggerTimes,EvIdx))
                elif TimeFormat=="LT": #Greenland Timezone is UTC-3
                    EventTime=np.append(EventTime,(UTC(TriggerTimes,EvIdx)-2)%24)
                else:
                    print("Please enter a valid TimeFormat")
                    return
                
                
    #print(np.sum([EventRMS[i] for i in np.arange(len(EventTime)) if EventTime[i]<=.25 ]))
    
    EventTimeCounts, EventTimeBins=np.histogram(EventTime, bins=NBins,range=(0,24),density=False) #Storing timestamps in histogram format
    #Make a histogram of the V_RMS value fully added in its respective bin by adding V_RMS as a weigth to the additions to the histogram
    
    EventRMSCountsList=np.array([np.histogram(EventTime, bins=NBins,range=(0,24),density=False,weights=EventRMS[ChIdx]) for ChIdx in np.arange(4)],dtype=object)
    #Indexable EventRMSCountsList[ChNr][0 for Counts, 1 for Bins][BinIdx]
    
    #RMSBins=np.digitize(EventTime,EventTimeBins) #Array of idx of bin in which the timestamp of each event falls
    
    ##Make a histogram of the V_RMS value fully added in its respective bin by adding V_RMS as a weigth to the additions to the histogram
    #EventRMSCounts, EventRMSBins=np.histogram(RMSBins, bins=24,range=(0,24),density=False,weights=EventRMS)
    
    #Compute std per bin
    EventTimeDig=np.digitize(EventTime,EventTimeBins)
    GroupedVRMS=np.empty((4,NBins),dtype=object)
    for ChIdx in range(4):
        for i in range(len(EventTimeDig)):
            #print("ChIdx: " + str(ChIdx) + ", EventTimeDig[i]-1:" + str(EventTimeDig[i]-1) + ",i: " + str(i))
            GroupedVRMS[ChIdx][EventTimeDig[i]-1]=np.append(GroupedVRMS[ChIdx][EventTimeDig[i]-1],EventRMS[ChIdx][i])
    ##Get rid of "None" entries in beginning of array
    for ChIdx in range(4):
        for i in range(len(GroupedVRMS[ChIdx])):
            GroupedVRMS[ChIdx][i]=np.delete(GroupedVRMS[ChIdx][i], 0) 
    VRMSStd=np.array([[np.std(GroupedVRMS[ChIdx][i]) if len(GroupedVRMS[ChIdx][i])!=0 else 0 for i in range(len(GroupedVRMS[ChIdx]))] for ChIdx in range(4)])
    
    MidBins= np.array([(EventTimeBins[i] + EventTimeBins[i+1])/2 for i in range(0,len(EventTimeBins)-1)])           
    VRMSAvg=np.array([[EventRMSCountsList[ChIdx][0][i]/EventTimeCounts[i]  if EventTimeCounts[i] !=0 else 0 for i in range(len(EventTimeCounts))] for ChIdx in np.arange(4)])
    #Indexabl as VRMSAvg[ChNr][BinIdx]
    if False:
        plt.figure(figsize=(15,5))
        plt.figtext(0.2, 0.8, "Entries:" + str(np.sum(EventTimeCounts)), fontsize=18,bbox=dict(edgecolor='black', facecolor='none', alpha=0.2, pad=10.0))
        #plt.plot(SamplingTimes*(10**9),TimeTrace,'-')#, label="Channel " + str(ChNr))
        #plt.plot(Energies,TritonEnergyLoss,'-',color='r', label="Triton")
        #plt.hist(RMSBins, bins=24,range=(0,24),density=False, weights=[EventRMS[i]/EventRMSCounts[i] for i in range(len(EventRMS))])
        plt.plot(MidBins,VRMSAvg[0],'.')
        plt.grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
        plt.title("V_RMS of Station " + str(StNr) + ", channel " + str(ChNrs[0]) + " for " + str(NRuns) + " runs between run " + str(Runs[0]) + " and run " + str(Runs[-1]) + " throughout the day for " + str(NBins) + " bins")
        #plt.ylim(-50,50)
        #plt.xlim(0,np.max(SamplingTimes*(10**9)))
        plt.xlabel("Time (hrs)",fontsize=20)#20)
        plt.ylabel("V_RMS (V)",fontsize=20)#20)
        plt.xticks(np.arange(0, 24, 1.0),fontsize=25)#15)
        plt.yticks(fontsize=25)#15)
        #plt.legend()
        plt.show()
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle("Transit curve for StNr " + str(StNr) + ", for " + str(np.sum(EventTimeCounts)) + " entries", fontsize=25)
    #fig.text(0.30, 0.87, r"Null hypothesis $H_0$: dice are functioning properly and follow a uniform distribution", fontsize=10)
    #plt.subplots_adjust(top=0.82)
    #fig.title(r"Null hypothesis $H_0$: dice are functioning properly and follow a uniform distribution")
    
    for i in np.arange(4):
        yidx=i%2
        if i <2:
            xidx=0
        else:
            xidx=1
        
        #for j in range(len(GroupedVRMS[i])):
            #axs[xidx, yidx].plot(MidBins[j]*np.ones(len(GroupedVRMS[i][j])),1000*GroupedVRMS[i][j])
        axs[xidx, yidx].errorbar(MidBins,1000*VRMSAvg[i],yerr=1000*VRMSStd[i],fmt=".")
        axs[xidx, yidx].plot(MidBins,1000*VRMSAvg[i],'r.')
        axs[xidx, yidx].grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
        #axs[0, 0].legend()
        axs[xidx, yidx].set_title("Channel " + str(ChNrs[i]), fontsize=20)
        if xidx==1:
            axs[xidx, yidx].set_xlabel(TimeFormat + " Time (hrs)",fontsize=19)
        if yidx==0:
            axs[xidx, yidx].set_ylabel(r"$V_{RMS}$ in mV",fontsize=19)
        axs[xidx, yidx].set_xticks(np.arange(0, 24, 1.0))
        axs[xidx, yidx].xaxis.set_tick_params(labelsize=18)
        axs[xidx, yidx].yaxis.set_tick_params(labelsize=18)
        #axs[xidx, yidx].set_yticks(fontsize=25)

    fig.tight_layout()
    #fig.subplots_adjust(hspace=0.4)
    #plt.figtext(0.25, 0.01, "The analysis indicates a " + str(np.round(2*100*PVal,4)) + r"% chance that $H_0$ is true", fontsize=18)
    #plt.text(200,-200,"The analysis indicates a " + str(2*100*PVal) + "% chance that H_0 is true")
    plt.show()

def RunStartTime(StNr, Run):
    """Returns the start time of a run in Unix time"""
    CombinedFile=GetCombinedFile(StNr,Run)
    TriggerTimes=CombinedFile['combined']['header']["trigger_time"].array(library='np') 
    #print(datetime.utcfromtimestamp(TriggerTimes[0]))
    return TriggerTimes[0]

def RunEndTime(StNr, Run):
    """Returns the end time of a run in Unix time"""
    CombinedFile=GetCombinedFile(StNr,Run)
    TriggerTimes=CombinedFile['combined']['header']["trigger_time"].array(library='np') 
    #print(datetime.utcfromtimestamp(TriggerTimes[0]))
    return TriggerTimes[-1]

def RunsTimeRanges(StNr, Runs):
    """Returns the time ranges of a list of runs in Unix time"""
    TimeRanges=np.empty((0,2),float)
    for Run in Runs:
        path=Path(StNr,Run)
        if os.path.isfile(path+"/combined.root"):
            CombinedFile=GetCombinedFile(StNr,Run)
            TriggerTimes=CombinedFile['combined']['header']["trigger_time"].array(library='np')
            TimeRanges=np.concatenate((TimeRanges,np.array([[TriggerTimes[0],TriggerTimes[-1]]])),axis=0)
        
    #print(datetime.utcfromtimestamp(TriggerTimes[0]))
    return TimeRanges

def TimedRunList(OGStNr,StNr,RunsRange=[0,100]):
    """Highly specific function that returns runs of a station StNr that occured during runs RunsRange of another station OGStNr"""
    RunTimes=[RunStartTime(OGStNr, RunsRange[0]),RunStartTime(OGStNr, RunsRange[1])]
    RunList=np.array([])
    for Run in np.arange(0,1000,dtype=int):
        path=Path(StNr,Run)
        if os.path.isfile(path+"/combined.root"):
            if RunTimes[0] < RunStartTime(StNr, Run) < RunTimes[1]:
                RunList=np.append(RunList,Run)
    return RunList

def RunsInTimeframe(StNr,t0,t1,Runs=np.arange(1000)):
    """Returns a list of all runs of this station which are within the given LST range [t0,t1] and in the run range Runs"""
    RunList=np.array([],dtype=int) #Array to store run values
    EventList=[]
    for Run in Runs:
        path=Path(StNr,Run)
        #if os.path.isdir(path+"/combined.root"):
        if os.path.isfile(path+"/combined.root"):

            CombinedFile=GetCombinedFile(StNr,Run)
            RadiantData=CombinedFile['combined']['waveforms']['radiant_data[24][2048]'].array(library='np')
            EventNrs=CombinedFile['combined']['waveforms']['event_number'].array(library="np")
            TriggerTimes=CombinedFile['combined']['header']["trigger_time"].array(library='np') 
            for EvNr in EventNrs:
                EvIdx=np.where(EventNrs==EvNr)[0][0]
                if t0<=LST(TriggerTimes,EvIdx)<=t1:
                    if not Run in RunList:
                        RunList=np.append(RunList,int(Run))
                        EventList.append([EvNr])
                    else:
                        EventList[-1].append(EvNr)  

    return RunList, EventList


# # TransCurveEnvInsp:

def GrafanaData(Nr,FileName):
    '''Returns Temperature data of the place with number Nr in the cvs file.'''
    import csv
    from datetime import datetime
    with open(FileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        Value=np.array([])
        Time=np.array([])
        for row in csv_reader:
            if line_count == 0:
                Name=row[Nr+1]
                line_count += 1
                #print("length of row is " + str(len(row)))
            elif len(row)==0:
                break
            else:
                #print(row)
                if row[Nr+1]!="":
                    #time=row[0]
                    time=dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
                    DataCount=0
                    data=float(row[Nr+1])
                    Time=np.append(Time,time)
                    Value=np.append(Value,data)
                    #for data in row[1:]:
                    #    if data!='':
                    #        1+1
                    #        Time=np.append(Time,time)
                    #        Temp=np.append(Temp,data)
                    
                #print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                #print(row)
                line_count += 1
            if line_count%50000==0:
                print("progress of " + Name + " " + str(Nr) + "/" + str(len(row)-1) +" : " + str(np.round(100*line_count/477430,2)) + '%', end="\r")
        #print(f'Processed {line_count} lines.')
        #print(Data)
        return Name, Time, Value

def DailyGrafanaAvg(LocNr,FileName,TimeRanges,NBins=24,TimeFormat="UTC"):
    "Plots the average of a physical quantity at location number LocNr provided as a csv file FileName from Grafana as a function of LST during a set time ranges TimeRanges (with shape (x,2) in Unix time)"
    from datetime import datetime
    Name, AllTime, AllVal = GrafanaData(LocNr,FileName)
    
    Val=np.array([]) #Array to store value of each sample in the timeranges
    Time=np.array([])#Array to store timestamp of each sample in the timeranges
    
    for i in np.arange(len(AllTime)):
        for TimeRange in TimeRanges:
            if datetime.utcfromtimestamp(TimeRange[0])<AllTime[i]<datetime.utcfromtimestamp(TimeRange[1]) and (not AllTime[i] in Time):
                Val=np.append(Val,AllVal[i])
                T=AllTime[i].hour+AllTime[i].minute/60 + AllTime[i].second/3600
                if TimeFormat=="UTC":
                    Time=np.append(Time, T)        
                elif TimeFormat=="LT": #Greenland Timezone is UTC-3
                    Time=np.append(Time, (T-2)%24)
                else:
                    print("Enter a valid TimeFormat")
    TimeCounts, TimeBins=np.histogram(Time, bins=NBins,range=(0,24),density=False) #Storing timestamps in histogram format
    #Make a histogram of the V_RMS value fully added in its respective bin by adding V_RMS as a weigth to the additions to the histogram
    ValCounts, ValBins=np.histogram(Time, bins=NBins,range=(0,24),density=False,weights=Val)    

   
    MidBins= np.array([(TimeBins[i] + TimeBins[i+1])/2 for i in range(0,len(TimeBins)-1)])           
    ValAvg=np.array([ValCounts[i]/TimeCounts[i]  if TimeCounts[i] !=0 else 0 for i in range(len(ValCounts))])
    
    #Compute std per bin
    TimeDig=np.digitize(Time,TimeBins)
    GroupedVal=np.empty((NBins,),dtype=object)
    for i in range(len(TimeDig)):
        GroupedVal[TimeDig[i]-1]=np.append(GroupedVal[TimeDig[i]-1],Val[i])
    ##Get rid of "None" entries in beginning of array
    for i in range(len(GroupedVal)):
        GroupedVal[i]=np.delete(GroupedVal[i], 0) 
    ValStd=np.array([np.std(GroupedVal[i]) for i in range(len(GroupedVal))])
    
    plt.figure(figsize=(15,5))
    plt.figtext(0.2, 0.8, "Entries:" + str(np.sum(TimeCounts)), fontsize=18,bbox=dict(edgecolor='black', facecolor='none', alpha=0.2, pad=10.0))
    #for i in range(len(GroupedVal)):
    #    plt.plot(MidBins[i]*np.ones(len(GroupedVal[i])),GroupedVal[i])
    plt.errorbar(MidBins,ValAvg,yerr=ValStd,fmt=".")
    plt.grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
    plt.title("Average value for " + str(Name) + " with " + str(NBins) + " bins")
    #plt.ylim(-50,50)
    #plt.xlim(0,np.max(SamplingTimes*(10**9)))
    plt.xlabel(TimeFormat + " Time (hrs)",fontsize=20)#20)
    plt.ylabel("A.U.",fontsize=20)#20)
    plt.xticks(np.arange(0, 24, 1.0),fontsize=25)#15)
    plt.yticks(fontsize=25)#15)
    #plt.legend()
    plt.show()
    return Name, Time, Val

def TriggerFilter(StNr, ChNr, Runs,has_rf0,has_rf1,has_ext,has_pps,has_soft):
    """Returns the runs following the trigger demands"""
    RunsEvts=np.empty((1,2),dtype=int)
    Requirements=[has_rf0,has_rf1,has_ext,has_pps,has_soft]
           
    for Run in Runs:
        path=Path(StNr,Run)
        if os.path.isfile(path+"/combined.root") :
            CombinedFile=GetCombinedFile(StNr,Run)
            EventNrs=CombinedFile['combined']['waveforms']['event_number'].array(library="np")
            TriggerInfo=TrigInfo(StNr,ChNr,Run)
            for EvIdx in range(len(EventNrs)):
                Triggers=[TriggerInfo["trigger_info.radiant_info.RF_window[2]"][EvIdx][0],TriggerInfo["trigger_info.radiant_info.RF_window[2]"][EvIdx][1],TriggerInfo["trigger_info.ext_trigger"][EvIdx],TriggerInfo["trigger_info.pps_trigger"][EvIdx],TriggerInfo["trigger_info.force_trigger"][EvIdx]] 
                if all([Triggers[i]==Requirements[i] or not Requirements[i] in [1,0] for i in range(len(Triggers))]):
                    #RunsEvts=np.append(RunsEvts,[Run,EvIdx])
                    RunsEvts=np.concatenate((RunsEvts,np.array([[Run,EventNrs[EvIdx]]])),axis=0)

    return RunsEvts[1:]

def TriggerFilterAlt(StNr, ChNr, Runs,has_rf,has_ext,has_pps,has_soft):
    """Returns the runs and events following the trigger demands"""
    #RunsEvts=np.empty((1,2),dtype=int)
    FilteredRuns=np.array([],dtype=int)
    FilteredEvNrs=[]
    Requirements=[has_rf,has_ext,has_pps,has_soft]
           
    for Run in Runs:
        path=Path(StNr,Run)
        if os.path.isfile(path+"/combined.root") and os.path.isfile(path+"/daqstatus.root"):
            CombinedFile=GetCombinedFile(StNr,Run)
            EventNrs=CombinedFile['combined']['waveforms']['event_number'].array(library="np")
            TriggerInfo=TrigInfo(StNr,ChNr,Run)
            for EvIdx in range(len(EventNrs)):
                if TriggerInfo['trigger_info.which_radiant_trigger'][EvIdx]<-100:
                    HasRFTrigger=0
                else:
                    HasRFTrigger=1
                Triggers=[HasRFTrigger,TriggerInfo["trigger_info.ext_trigger"][EvIdx],TriggerInfo["trigger_info.pps_trigger"][EvIdx],TriggerInfo["trigger_info.force_trigger"][EvIdx]] 
                if all([Triggers[i]==Requirements[i] or not Requirements[i] in [1,0] for i in range(len(Triggers))]):
                    #RunsEvts=np.append(RunsEvts,[Run,EvIdx])
                    if not Run in FilteredRuns:
                        FilteredRuns=np.append(FilteredRuns,int(Run))
                        FilteredEvNrs.append(np.array([],dtype=int))
                    FilteredEvNrs[-1]=np.append(FilteredEvNrs[-1],int(EventNrs[EvIdx]))
                    #RunsEvts=np.concatenate((RunsEvts,np.array([[Run,EventNrs[EvIdx]]])),axis=0)

    #return RunsEvts[1:]
    return FilteredRuns,FilteredEvNrs


# # Frequency windows:

def NotchFilters(CritFreqs,Q,Freqs,SamplingRate):
    """Returns a combination of Notch filters with critical frequencies CritFreqs, a characteristic factor Q typical for a Notch filter. The resulting filter is evaluated at frequencies Freqs, with a sampling rate SamplingRate."""
    from scipy import signal
    NotchFilters=np.empty((1,len(Freqs)),dtype=object)
    for freq in CritFreqs:
        b,a=signal.iirnotch(freq, Q, fs=SamplingRate)
        w, h = signal.freqs(b, a,worN=2*np.pi*Freqs)
        freqa, h = signal.freqz(b, a, worN=2*np.pi*Freqs,fs=2*np.pi*SamplingRate)        
        NotchFilters=np.concatenate((NotchFilters,np.array([np.abs(h)])),axis=0)
    NotchFilters=np.delete(NotchFilters,0,0)
    TotalFilter=np.prod(NotchFilters,axis=0)
    #plt.figure()
    #plt.plot(Freqs,TotalFilter)
    #plt.xlim(0,1.5*10**9)
    #plt.xlim(3*10**8,5*10**8)
    #plt.show()
    return TotalFilter

def LowpassButter(CritFreq,N,Freqs):
    """Returns a lowpaww Butterworth filter at critical frequency CritFreq and a scaling factor N, which is evaluated at frequencies Freqs."""
    from scipy import signal
    b,a=signal.butter(N, 2*np.pi*CritFreq, btype='low', analog=True, output='ba', fs=None)
    w, h = signal.freqs(b, a,worN=2*np.pi*Freqs)
    #plt.figure()
    #plt.plot(Freqs,TotalFilter)
    #plt.xlim(0,1.5*10**9)
    #plt.xlim(3*10**8,5*10**8)
    #plt.show()
    return np.abs(h)


# # GalaxyNoiseSpectrum:

def GalacticNoiseSpectrum(StNr,ChNr,Run,EvtNr,Plot=False):
    """Plots the VRMS of galactic noise as a function of time of the day for station StNr, channel ChNr during run Run for event EvtNr by using the NuRadioReco software."""
    #import NuRadioReco.modules.channelGalacticNoiseSpectrum
    import Galactic_noise.channelGalacticNoiseSpectrum as channelGalacticNoiseSpectrum
    from NuRadioReco.detector import detector
    from NuRadioReco.utilities import units
    from NuRadioReco.framework import event,station, channel
    import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
    from datetime import datetime
    from datetime import timedelta
    import scipy.fft as scfft
    
    CombinedFile=GetCombinedFile(StNr,Run)
    RadiantData=CombinedFile['combined']['waveforms']['radiant_data[24][2048]'].array(library='np')
    EventNrs=CombinedFile['combined']['waveforms']['event_number'].array(library="np")
    TriggerTimes=CombinedFile['combined']['header']["trigger_time"].array(library='np')  
    
    EvIdx=np.where(EventNrs==EvtNr)[0][0]
    Date=datetime.utcfromtimestamp(TriggerTimes[EvIdx])# - timedelta(hours=2, minutes=0)
    
    #Obtaining path to relevant json file for detector description
    detpath = os.path.dirname(detector.__file__)
    detpath+="/RNO_G/RNO_season_2021.json"
    
    GNDetector = detector.Detector(json_filename = detpath)
    GNDetector.update(Date) #date in example
    GNEvent=event.Event(Run,EvtNr)
    GNStation=station.Station(StNr)
    GNStation.set_station_time(Date)
    GNChannel=channel.Channel(ChNr)
    GNChannel.set_trace(trace=np.zeros(2048), sampling_rate=3.2 * units.GHz)
    GNStation.add_channel(GNChannel) 
    
    
    channelGalacticNoiseAdder = channelGalacticNoiseSpectrum.channelGalacticNoiseAdder()
    channelGalacticNoiseAdder.begin(debug=False,n_side=4,interpolation_frequencies=np.arange(10 * units.MHz, 1100 * units.MHz,100*units.MHz))
    GalacticNoiseTrace=channelGalacticNoiseAdder.run(GNEvent,GNStation,GNDetector,passband=[10 * units.MHz, 1000 * units.MHz])
    
    sampling_rate=3.2 * (10**9) #Sampling rate in Hertz according to the python file of NuRadioReco.modules.io.rno_g
    TimeStep=1/sampling_rate #Time between two samples
    SamplingTimes=np.arange(0,len(RadiantData[0][0])*TimeStep,TimeStep)
    GalacticNoiseSpec=scfft.fft(GalacticNoiseTrace)
    GalacticNoiseSpec=np.fft.fftshift(GalacticNoiseSpec)
    freq=scfft.fftfreq(len(SamplingTimes),(SamplingTimes[-1]-SamplingTimes[0])/len(SamplingTimes))
    freq=np.fft.fftshift(freq)
    
    #Incorporate hardware response
    #from NuRadioReco.framework.base_trace import BaseTrace
    #dummy_trace = BaseTrace()
    #dummy_trace.set_trace(np.zeros(2048), 3.2*units.GHz)
    #frequencies = dummy_trace.get_frequencies()
    hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
    GalacticNoiseSpec *= hardwareResponseIncorporator.get_filter(freq*units.Hz, GNStation.get_id(), GNChannel.get_id(), GNDetector, temp=293.15, sim_to_data=True, phase_only=False, mode=None, mingainlin=None)
    
    ## zero first bins to avoid DC offset
    #trace_fft[0] = 0
    
    
    if Plot:
        plt.figure(figsize=(20,5))
        plt.title("Galactic noise spectrum of Station " + str(StNr) + ", channel " + str(ChNr) + ", run " + str(Run) + ", event " + str(EvtNr) + " at " + str(Date.replace(microsecond=0) ) + " UTC",fontsize=20)
        #plt.plot(GNChannel.get_frequencies()/units.MHz,np.abs(GalacticNoiseSpec))
        plt.plot(freq,np.abs(GalacticNoiseSpec))
        plt.xlim(0,1.5*10**9)
        plt.show()
    return freq, np.abs(GalacticNoiseSpec)

def GalacticNoiseVRMSCurve(StNr,ChNr,RunEvts,FFTFilter=False,Lowpass=False,Plot=True):
    sampling_rate=3.2 * (10**9) #Sampling rate in Hertz according to the python file of NuRadioReco.modules.io.rno_g
    EventRMS=np.array([])
    EventTime=np.array([])
    for (RunNr,EvNr) in RunEvts:
        CombinedFile, DAQStatFile, HeadersFile, PedestalFile=FilesStRun(StNr,RunNr)
        RadiantData=CombinedFile['combined']['waveforms']['radiant_data[24][2048]'].array(library='np')
        EventNrs=CombinedFile['combined']['waveforms']['event_number'].array(library="np")
        TriggerTimes=CombinedFile['combined']['header']["trigger_time"].array(library='np')  
        
        EvIdx=np.where(EventNrs==EvNr)[0][0]
        EventTime=np.append(EventTime,LST(TriggerTimes,EvIdx))
        
        GNFreq,GNSpec=GalacticNoiseSpectrum(22,13,RunNr,EvNr,Plot=False)
        
        if Lowpass:
            CritFreq=110*10**6
            GNSpec=np.array([GNSpec[i]*LowpassButter(CritFreq,20,GNFreq)[i] for i in range(len(GNFreq))])
        if FFTFilter:
            GNSpec=np.array([GNSpec[i]*NotchFilters([403*10**6,120*10**6,807*10**6,1197*10**6],75,GNFreq,sampling_rate)[i] for i in range(len(GNFreq))])
            
            
        EventRMS=np.append(EventRMS,np.sqrt(np.sum(np.abs(GNSpec)**2)))
        
        #plt.figure(figsize=(20,5))
        #plt.title("Galactic noise spectrum of Station " + str(StNr) + ", channel " + str(ChNr) + ", run " + str(RunNr) + ", event " + str(EvNr),fontsize=20)
        #plt.plot(GNFreq,np.abs(GNSpec))
        #plt.xlim(0,1.5*10**9)
        #plt.show()
    
    
    if Plot:
        plt.figure(figsize=(15,5))
        #plt.figtext(0.2, 0.8, "Entries:" + str(np.sum(EventTimeCounts)), fontsize=18,bbox=dict(edgecolor='black', facecolor='none', alpha=0.2, pad=10.0))
        #plt.hist(RMSBins, bins=24,range=(0,24),density=False, weights=[EventRMS[i]/EventRMSCounts[i] for i in range(len(EventRMS))])
        #plt.errorbar(MidBins,1000*VRMSAvg,yerr=1000*VRMSStd,fmt=".",zorder=2)
        #for i in range(len(GroupedVRMS)):
        #    plt.plot(MidBins[i]*np.ones(len(GroupedVRMS[i])),1000*GroupedVRMS[i],"r.", alpha=0.5,zorder=1)

        #plt.plot(MidBins,1000*VRMSAvg,'r.')
        plt.plot(EventTime,EventRMS,'.', markersize=20)
        plt.grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
        plt.title("Galactic noise V_RMS of Station " + str(StNr) + ", channel " + str(ChNr) + " for " + str(len(RunEvts)) + " events")
        #plt.ylim(0.8*np.min(EventRMS),1.2*np.max(EventRMS))
        #plt.xlim(0,np.max(SamplingTimes*(10**9)))
        plt.xlabel("LST Time (hrs)",fontsize=20)#20)
        plt.ylabel("V_RMS (mV)",fontsize=20)#20)
        plt.xticks(np.arange(0, 24, 1.0),fontsize=25)#15)
        plt.yticks(fontsize=25)#15)
        #plt.legend()
        plt.show()
    return EventTime,EventRMS
