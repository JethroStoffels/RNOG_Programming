# # Description:
# This Jupyter notebook contains all python functions developed for the galactic noise study in order
# to allow for easy importing. The common functions are organised per notebook in which they were first used.

# # Import modules:
from Functions import *
import pandas as pd
# # Galaxy Stacking:

def LST(UnixTime):
    """Computes the Local Sidereal Time in decimal hours via the astropy module
    Parameters:
    TriggerTimes=The trigger time data of the event in Unix time.  
    EvIdx=index of the relevant event in the EventNumbers data for this data run
    """
    from datetime import datetime
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    from astropy import units as u
    observing_location = EarthLocation(lat=72.598265*u.deg, lon=-38.459936*u.deg)
    observing_time = Time(datetime.utcfromtimestamp(UnixTime), scale='utc', location=observing_location)
    T=observing_time.sidereal_time('mean')
    return T.hour

def UTC(UnixTime):
    """Computes the UTC time in decimal hours
    Parameters:
    TriggerTimes=The trigger time data of the event in Unix time.  
    EvIdx=index of the relevant event in the EventNumbers data for this data run
    """
    from datetime import datetime
    T=datetime.utcfromtimestamp(UnixTime)
    return T.hour+T.minute/60 + T.second/3600

def LT(UnixTime):
    """Computes the Local Time at Summit Station for an event
    Parameters:
    TriggerTimes=The trigger time data of the event in Unix time.  
    EvIdx=index of the relevant event in the EventNumbers data for this data run
    """
    return (UTC(UnixTime)-2)%24

def HardCut(Results,Threshold):
    """
    Performs threshold cut on the results passed on.
    Parameters:
    Results=PD data array on which the cut should be performed.
    Threshold=Value above which all waveforms should be rejected.
    """
    ####Quality cuts:
    ##Hard cut
    # if HardCut>0:
    #     print("Applying hard cut")
    #     Results = Results[Results['VRMS'] <= HardCut]
    return Results[Results['VRMS'] <= HardCut]
 
def StdCut(Results,StdCut):
    """
    Performs multiple sigma cuts on the results passed on.
    Parameters:
    Results=PD data array on which the cut should be performed.
    StdCut=(AmtStd,StdCut) if larger than zero, all VRMS outliers above StdCut standard variations will be cut out of the analysis. This procedure is repeated AmtStd times.
    """
    BinStat=Results.groupby(["Bin"]).agg({"VRMS":[np.median,np.std]})
    ##Std Cut
    if np.all(np.array(StdCut)>0):
        print("Applying Quality cut")
        for StdAmt in range(StdCut[0]):
            BinStat=Results.groupby(["Bin"]).agg({"VRMS":[np.median,np.std]})
            for idx,row in BinStat.iterrows():
                if  pd.isna(row[('VRMS', 'median')])  or pd.isna(row[('VRMS', 'std')]):
                    continue
                median,std=row[('VRMS', 'median')],row[('VRMS', 'std')]
                Filter=np.logical_and((Results["Bin"]==idx),(((Results["VRMS"])<(median - StdCut[1]*std))   | (Results["VRMS"] > (median + StdCut[1]*std))) )
                Results = Results.drop(Results[Filter].index) 
    return Results

def TransitCurve(StNr,ChNr,Runs,NBins=4*24,ZeroAvg=True,TimeFormat="LST",Triggers=(5,5,5,5),StdCut=(-1,-1),FFTFilter=True,Lowpass=False,HardCut=-0.0035,EventSkip=1,Plot=True):
    """
    Plots the Average V_RMS as a function of time of the day.
    Parameters:
    StNr,ChNr,Runs=Station number, channel number, list of runs to analyse
    NBins=amount of bins to divide the full day in
    ZeroAvg=Boolean: if true, the timetraces will firs tbe zero averaged
    TimeFormat= String: Dictates what timeformat the x-axis will be in. Options: "LST": local sidereal time, "LT": Local time & "UTC": UTC time
    Triggers=tupel of flags to dictate which triggers are allowed in the analysis. Events with different triggers are not used (0=has to be absent, 1=has to be present, anything else=both 0 and 1 can be used for analysis)
    StdCut=(AmtStd,StdCut) if larger than zero, all VRMS outliers above StdCut standard variations will be cut out of the analysis. This procedure is repeated AmtStd times.
    FFTFilter=Boolean: if true, applies a Notch filter to all frequency spectra to cut out frequencies which have shown to be potentially problematic
    Lowpass= Boolean: if true, a butterworth lowpass filter will be applied in order to maintain only galactic noise dominated frequencies
    HardCut= a hardcut (in Volt) to be performed on all VRMS values. Any event with a VRMS value exceeding this is excluded from the analysis.
    EventSkip=skips fraction of data to look at. EventSkip=5 => only analyse every 5th event => only analyse 20% of the data.
    """
    (has_rf,has_ext,has_pps,has_soft)=Triggers
    NRuns=0
    EventRMS=np.array([]) #Array to store V_RMS value of each event
    EventTimeUnix=np.array([])#Array to store timestamp of each event
    EventTimeLST=np.array([])#Array to store timestamp of each event
    EventTimeLT=np.array([])#Array to store timestamp of each event
    FilteredRuns,FilteredEvNrs=TriggerFilter(StNr, ChNr, Runs,has_rf,has_ext,has_pps,has_soft)
    if len(FilteredEvNrs)==0: 
        print("WARNING: empty dataframe returned since no events passed trigger filter")
        return pd.DataFrame(columns=["RunNr", "EvNr", "VRMS","Unix","LST","LT"]) 
    # #Decimate amount of events if EventSkip>1
    if EventSkip>1:
        for i in range(len(FilteredEvNrs)):
            FilteredEvNrs[i]=FilteredEvNrs[i][::EventSkip]
    for Run in FilteredRuns:
        path=Path(StNr,Run)
        #if os.path.isfile(path+"/combined.root"):
        if os.path.isfile(path+"/waveforms.root") and os.path.isfile(path+"/headers.root"):
            NRuns+=1
            # #If CombinedFile exists:
            # CombinedFile=GetCombinedFile(StNr,Run)
            # RadiantData=CombinedFile['combined']['waveforms']['radiant_data[24][2048]'].array(library='np')
            # EventNrs=CombinedFile['combined']['waveforms']['event_number'].array(library="np")
            # TriggerTimes=CombinedFile['combined']['header']["trigger_time"].array(library='np')
            
            #Read in data (RadiantData is read in on a per event basis in order to save memory)
            WaveFormFile=GetWaveformsFile(StNr,Run)
            HeaderFile=GetHeaderFile(StNr,Run)
            EventNrs=WaveFormFile['event_number'].array(library="np")
            TriggerTimes=HeaderFile["trigger_time"].array(library='np')
            RunIdx=np.where(FilteredRuns==Run)[0][0]

            for EvNr in FilteredEvNrs[RunIdx]:
                EvIdx=np.where(EventNrs==EvNr)[0][0]
                RadiantData=WaveFormFile['radiant_data[24][2048]'].array(entry_start=EvIdx, entry_stop=EvIdx+1,library='np')
                #Check if the relevant timestamp is not inf or nan
                if np.isinf(TriggerTimes[EvIdx]) or np.isnan(TriggerTimes[EvIdx]):
                    #print("Inf or nan timestamp at: Run" + str(Run) + ", EvNr" + str(EvNr))
                    FilteredEvNrs[RunIdx]=np.delete(FilteredEvNrs[RunIdx],np.where(FilteredEvNrs[RunIdx]==EvNr)[0][0])
                    if len(FilteredEvNrs[RunIdx])==0:
                        FilteredRuns=np.delete(FilteredRuns, RunIdx)
                        FilteredEvNrs=np.delete(FilteredEvNrs, RunIdx)
                    continue
                VoltageTrace=ADCtoVoltage(RadiantData[0][ChNr]) #Convert timetrace data from ADC to voltage (index on 0 since only one event is read in from WaveFormsFile)
                if ZeroAvg==True: #Zero average the timetrace 
                    Vmean=np.mean(VoltageTrace)
                    VoltageTrace-=Vmean
                #If a filter is required: convert to frequency domain and apply filter
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
                        CritFreq=150*(10**6)
                        TotalFilter=np.multiply(TotalFilter,LowpassButter(CritFreq,20,freq))
                    FFT=scfft.fft(VoltageTrace)
                    FFT=np.fft.fftshift(FFT)
                    FFT=np.array([FFT[i]*TotalFilter[i] for i in range(len(freq))])
                    VoltageTrace=np.abs(scfft.ifft(FFT))
                EventRMS=np.append(EventRMS,np.sqrt(np.mean([V**2 for V in VoltageTrace])))
                EventTimeUnix=np.append(EventTimeUnix,TriggerTimes[EvIdx])
                # EventTimeLST=np.append(EventTime,LST(TriggerTimes,EvIdx))
                # EventTimeLT=np.append(EventTime,LT(TriggerTimes,EvIdx))

            del HeaderFile, RadiantData, EventNrs,TriggerTimes
    EventTimeLST=[LST(UnixTime) for UnixTime in EventTimeUnix]
    EventTimeLT=[LT(UnixTime) for UnixTime in EventTimeUnix] 
    Runs=np.concatenate([int(FilteredRuns[i])*np.ones(len(FilteredEvNrs[i]),dtype=int) for i in range(len(FilteredEvNrs))],axis=0)
    Results=pd.DataFrame({"RunNr":Runs,"EvNr":np.concatenate(FilteredEvNrs,axis=0),"VRMS":EventRMS,"Unix":EventTimeUnix,"LST":EventTimeLST,"LT":EventTimeLT})
    
    if HardCut>0:
        Results=HardCut(Results,Threshold)
        
    if np.all(np.array(StdCut)>0):
        Results=StdCut(Results,StdCut)

    BinEdges=np.linspace(0,24,NBins+1,True)
    Results['Bin'] = pd.cut(Results['LST'], BinEdges)
    #MidBins=np.array([(BinEdges[i] + BinEdges[i+1])/2 for i in range(0,len(BinEdges)-1)]))
    BinStat=Results.groupby(["Bin"],observed=False).agg({"VRMS":['median',(lambda x: np.std(x)/np.sqrt(len(x)))]})
    

    if Plot:
        NEntries=Results.shape[0]
        
#         EventRMSCounts, EventRMSBins=np.histogram(EventTime, bins=NBins,range=(0,24),density=False,weights=EventRMS)
        BinStat=Results.groupby(["Bin"]).agg({"VRMS":[np.median,(lambda x: np.std(x)/np.sqrt(len(x))) ]})
        PlotFilter=BinStat[('VRMS', 'median')].isnull()
        VRMSMed=BinStat[('VRMS', 'median')].drop(BinStat[('VRMS', 'median')][PlotFilter].index)
        VRMSStd=BinStat[('VRMS', '<lambda_0>')].drop(BinStat[('VRMS', 'median')][PlotFilter].index).fillna(0)
        MidBins=np.array([(BinEdges[i] + BinEdges[i+1])/2 for i in range(0,len(BinEdges)-1)])[np.logical_not(PlotFilter)]
        
        plt.figure(figsize=(15,5))
        plt.figtext(0.2, 0.8, "Entries:" + str(Results.shape[0]), fontsize=18,bbox=dict(edgecolor='black', facecolor='none', alpha=0.2, pad=10.0))
        plt.errorbar(MidBins,1000*VRMSMed,yerr=1000*VRMSStd,fmt=".",zorder=2)
        # for i in range(len(GroupedVRMS)):
        #     plt.plot(MidBins[i]*np.ones(len(GroupedVRMS[i])),1000*GroupedVRMS[i],"r.", alpha=0.5,zorder=1)
        plt.grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
        plt.title("V_RMS of Station " + str(StNr) + ", channel " + str(ChNr) + " for " + str(NRuns) + " runs between run " + str(np.min(Runs)) + " and run " + str(np.max(Runs)) + " throughout the day for " + str(NBins) + " bins")
        plt.xlabel(TimeFormat + " Time (hrs)",fontsize=20)#20)
        plt.ylabel("V_RMS (mV)",fontsize=20)#20)
        plt.xticks(np.arange(0, 24, 1.0),fontsize=25)#15)
        plt.yticks(fontsize=25)#15)
        plt.show()
    return Results

def TransitCurvePlot(StNr,ChNr,FilePath,TimeFormat="LST",NBins=4*24,StdCut=(-1,-1),HardCut=-0.0035,SavePath="",Format="pdf"):
    """Plots transit curve results from npy files.
    Parameters:
    StNr=Station number.
    ChNr= Channel Number
    FileId= name of the file
    TimeFormat= String: Dictates what timeformat is written the x-axis, does not transform unix time to the chosen timeformat, visual effect only! Options: "LST": local sidereal time, "LT": Local time & "UTC": UTC time
    """
    import matplotlib as mpl
    #Read in stored results
    GNData=pd.read_pickle(FilePath+".pkl")
    
    BinEdges=np.linspace(0,24,NBins+1,True)
    GNData['Bin'] = pd.cut(GNData[TimeFormat], BinEdges)
    
    #Compute relevant statistics
    # BinStat=GNData.groupby(["Bin"]).agg({"VRMS":[np.median,(lambda x: np.std(x)/np.sqrt(len(x))) ]})
    # BinStat=GNData.groupby(["Bin"]).agg({"VRMS":[np.median,(lambda x: np.sqrt(np.pi/2)*(1.4826*np.median(np.abs(x-np.median(x))))/np.sqrt(len(x))) ]})
    BinStat=GNData.groupby(["Bin"],observed=False).agg({"VRMS":['median',(lambda x: np.sqrt(np.pi/2)*(1.4826*np.median(np.abs(x-np.median(x))))/np.sqrt(len(x))) ]})
    PlotFilter=BinStat[('VRMS', 'median')].isnull()
    VRMSMed=BinStat[('VRMS', 'median')].drop(BinStat[('VRMS', 'median')][PlotFilter].index)
    VRMSStd=BinStat[('VRMS', '<lambda_0>')].drop(BinStat[('VRMS', 'median')][PlotFilter].index).fillna(0)
    MidBins=np.array([(BinEdges[i] + BinEdges[i+1])/2 for i in range(0,len(BinEdges)-1)])[np.logical_not(PlotFilter)]
    
    if HardCut>0:
        GNData=HardCut(GNData,Threshold)
        
    if np.all(np.array(StdCut)>0):
        GNData=StdCut(GNData,StdCut)

    #Count the amount of entries in the transit curve
    NEntries=GNData.shape[0]
    
    plt.figure(figsize=(15,5))
    
    plt.minorticks_on()
    ax=plt.gca()
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    
    plt.figtext(0.2, 0.8, "Entries:" + str(np.sum(NEntries)), fontsize=18,bbox=dict(edgecolor='black', facecolor='none', alpha=0.2, pad=10.0))
    plt.errorbar(MidBins,1000*VRMSMed,yerr=1000*VRMSStd,fmt=".",zorder=2)
    plt.grid(which="both",color='grey', linestyle='-', linewidth=1,alpha=0.5)
    plt.title("Transit curve for station " + str(StNr) + ", antenna " + str(ChNr),fontsize=25)
    plt.xlabel(TimeFormat + " Time (hrs)",fontsize=25)#20)
    plt.ylabel("V_RMS (mV)",fontsize=25)#20)
    plt.xticks(np.arange(0, 25, 2.0),fontsize=25)#15)
    plt.yticks(fontsize=25)#15)
    plt.xlim(0,24)
    if SavePath!="":
        import datetime
        plt.savefig(SavePath + "TransitCurveSt"+str(StNr)+"Ch" + str(ChNr)+f"_{datetime.datetime.now().strftime('%y-%m-%d_%H%M')}"+"."+Format, format=Format, bbox_inches="tight")
    plt.show()
    return

def TrigInfo(StNr,RunNr):
    """Reads in the TriggerInfo databranch.
    StNr= Station number
    ChNr= Channel number
    RunNr= Run number
    """
    # #If headerfile is present
    # CombinedFile=GetCombinedFile(StNr,RunNr)
    # TriggerInfo=CombinedFile['combined']['header']['trigger_info'].array(library='np')
    HeaderFile=GetHeaderFile(StNr,RunNr)
    TriggerInfo=HeaderFile['trigger_info'].array(library='np')
    return TriggerInfo

# # TransCurveEnvInsp:

def GrafanaData(Nr,FileName):
    """Returns Temperature data of the place with number Nr in the cvs file.
    Parameters:
    Nr= Column number of data read in from the csv file. Dictates what station is read in.
    FileName= Name of the csv file
    """
    import csv
    import datetime as dt
    with open(FileName, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        Value=np.array([])
        Time=np.array([])
        for row in csv_reader:
            if line_count == 0: #The first row contains the name of the station
                Name=row[Nr+1]
                line_count += 1
            elif len(row)==0: #Stop when there's no data left to read
                break
            else:
                if row[Nr+1]!="": #If row of data is not empty, read in data
                    time=dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
                    DataCount=0
                    data=float(row[Nr+1])
                    Time=np.append(Time,time)
                    Value=np.append(Value,data)
                line_count += 1
            if line_count%50000==0:
                print("progress of " + Name + " " + str(Nr) + "/" + str(len(row)-1) +" : " + str(np.round(100*line_count/477430,2)) + '%', end="\r")
        return Name, Time, Value
    
def DailyGrafanaAvg(LocNr,FileName,StNr,Runs,NBins=24,TimeFormat="UTC"):
    """Plots the average of a physical quantity at location number LocNr provided as a csv file FileName from Grafana as a function of LST during a set time ranges TimeRanges (with shape (x,2) in Unix time).
    Parameters:
    LocNr= Column number of data read in from the csv file. Dictates what station is read in.
    FileName= Name of the csv file.
    StNr= Station number.
    Runs= List of runs for which this analysis should be performed.
    NBins= amount of bins the plot should be divided in.
    TimeFormat= String: Dictates what timeformat the x-axis will be in. Options: "LT": Local time & "UTC": UTC time
    """
    from datetime import datetime
    #Read in data
    Name, AllTime, AllVal = GrafanaData(LocNr,FileName)
    RunListUnixTimes=np.array([(GetHeaderFile(StNr,Run)["trigger_time"].array(library='np')[0],GetHeaderFile(StNr,Run)["trigger_time"].array(library='np')[-1]) for Run in Runs])
    Val=np.array([]) #Array to store value of each sample in the timeranges
    Time=np.array([])#Array to store timestamp of each sample in the timeranges
    
    #Collect the environmental datapoints that occured during one of the runs under study
    for i in np.arange(len(AllTime)):
        for TimeRange in RunListUnixTimes:
            if datetime.utcfromtimestamp(TimeRange[0])<AllTime[i]<datetime.utcfromtimestamp(TimeRange[1]) and (not AllTime[i] in Time):
                Val=np.append(Val,AllVal[i])
                T=AllTime[i].hour+AllTime[i].minute/60 + AllTime[i].second/3600
                if TimeFormat=="UTC":
                    Time=np.append(Time, T)        
                elif TimeFormat=="LT": #Greenland Timezone is UTC-3
                    Time=np.append(Time, (T-2)%24)
                else:
                    print("Enter a valid TimeFormat")
                    
    #Make a histogram of the V_RMS value fully added in its respective bin by adding V_RMS as a weigth to the additions to the histogram
    TimeCounts, TimeBins=np.histogram(Time, bins=NBins,range=(0,24),density=False) #Storing timestamps in histogram format
    ValCounts, ValBins=np.histogram(Time, bins=NBins,range=(0,24),density=False,weights=Val)    
    MidBins= np.array([(TimeBins[i] + TimeBins[i+1])/2 for i in range(0,len(TimeBins)-1)])           
    ValAvg=np.array([ValCounts[i]/TimeCounts[i]  if TimeCounts[i] !=0 else 0 for i in range(len(ValCounts))])
    
    #Group in bins
    TimeDig=np.digitize(Time,TimeBins)
    GroupedVal=np.empty((NBins,),dtype=object)
    for i in range(len(TimeDig)):
        GroupedVal[TimeDig[i]-1]=np.append(GroupedVal[TimeDig[i]-1],Val[i])
    ##Get rid of "None" entries in beginning of array
    for i in range(len(GroupedVal)):
        GroupedVal[i]=np.delete(GroupedVal[i], 0) 
    #Compute std per bin
    ValStd=np.array([np.std(GroupedVal[i])/np.sqrt(len(GroupedVal[i])) if len(GroupedVal[i])!=0 else 0 for i in range(len(GroupedVal))])
    
    plt.figure(figsize=(15,5))
    plt.figtext(0.2, 0.8, "Entries:" + str(np.sum(TimeCounts)), fontsize=18,bbox=dict(edgecolor='black', facecolor='none', alpha=0.2, pad=10.0))
    plt.errorbar(MidBins,ValAvg,yerr=ValStd,fmt=".")
    plt.grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
    plt.title("Average value for " + str(Name) + " with " + str(NBins) + " bins")
    plt.xlabel(TimeFormat + " Time (hrs)",fontsize=20)#20)
    plt.ylabel("A.U.",fontsize=20)#20)
    plt.xticks(np.arange(0, 24, 1.0),fontsize=25)#15)
    plt.yticks(fontsize=25)#15)
    plt.show()
    return Name, Time, Val

def TriggerFilter(StNr, ChNr, Runs,has_rf,has_ext,has_pps,has_soft):
    """Returns the runs and events following the trigger demands.
    Parameters:
    StNr= Station number.
    ChNr=Channel number.
    Runs= List of runs for which this analysis should be performed.
    has_rf= if 0 or 1: demands the data to have an rf trigger of False or True. If it is any other number, it is ignored.
    has_ext= if 0 or 1: demands the data to have an ext trigger of False or True. If it is any other number, it is ignored.
    has_pps= if 0 or 1: demands the data to have a pps trigger of False or True. If it is any other number, it is ignored.
    has_soft= if 0 or 1: demands the data to have an forced trigger of False or True. If it is any other number, it is ignored.
    """
    FilteredRuns=np.array([],dtype=int)
    FilteredEvNrs=[]
    Requirements=[has_rf,has_ext,has_pps,has_soft]
    for Run in Runs:
        path=Path(StNr,Run)
        # if os.path.isfile(path+"/combined.root"):
        if os.path.isfile(path+"/waveforms.root"):
            # # If Combined file is present
            # CombinedFile=GetCombinedFile(StNr,Run)
            # EventNrs=CombinedFile['combined']['waveforms']['event_number'].array(library="np")
            WaveformsFile=GetWaveformsFile(StNr,Run)
            if WaveformsFile==None:
                continue
            EventNrs=WaveformsFile['event_number'].array(library="np")    
            TriggerInfo=TrigInfo(StNr,Run)
            for EvIdx in range(len(EventNrs)):
                #Try to use the which_radiant_trigger, if it is not present just give it the surface_trigger value
                try:
                    TriggerInfo['trigger_info.which_radiant_trigger'][EvIdx]
                except:
                    HasRFTrigger=TriggerInfo['trigger_info.surface_trigger'][EvIdx]  #Placeholder instant filter laten passen, vraag na waar je anders RF trigger kan testen
                    #if HasRFTrigger:
                    #print(TriggerInfo.keys())
                    #continue
                # print("surface_trigger:",TriggerInfo['trigger_info.surface_trigger'][EvIdx],", TriggerInfo['trigger_info.which_radiant_trigger'][EvIdx]<-100", TriggerInfo['trigger_info.which_radiant_trigger'][EvIdx]<-100)
                else:
                    if TriggerInfo['trigger_info.which_radiant_trigger'][EvIdx]<-100:
                        HasRFTrigger=0
                    else:
                        HasRFTrigger=1
                Triggers=[HasRFTrigger,TriggerInfo["trigger_info.ext_trigger"][EvIdx],TriggerInfo["trigger_info.pps_trigger"][EvIdx],TriggerInfo["trigger_info.force_trigger"][EvIdx]] 
                #If the trigger requirements are met, add run/event to filtered list
                if all([Triggers[i]==Requirements[i] or not Requirements[i] in [1,0] for i in range(len(Triggers))]):
                    if not Run in FilteredRuns:
                        FilteredRuns=np.append(FilteredRuns,int(Run))
                        FilteredEvNrs.append(np.array([],dtype=int))
                    FilteredEvNrs[-1]=np.append(FilteredEvNrs[-1],int(EventNrs[EvIdx]))
                else:
                    continue
    return FilteredRuns,FilteredEvNrs


# # Frequency windows:

def NotchFilters(CritFreqs,Q,Freqs,SamplingRate):
    """Returns a combination of Notch filters with critical frequencies CritFreqs, a characteristic factor Q typical for a Notch filter. The resulting filter is evaluated at frequencies Freqs, with a sampling rate SamplingRate.
    Parameters:
    CritFreqs= List of frequencies that need to be filtered out.
    Q=Characteristic Q factor, specific to the Notch filter.
    Freqs= Frequencies to evaluate the filter at
    SamplingRate= Sampling rate to be used.
    """
    from scipy import signal
    #Collect all Notch filters for the different critical frequencies
    NotchFilters=np.empty((1,len(Freqs)),dtype=object)
    for freq in CritFreqs:
        b,a=signal.iirnotch(freq, Q, fs=SamplingRate)
        w, h = signal.freqs(b, a,worN=2*np.pi*Freqs)
        freqa, h = signal.freqz(b, a, worN=2*np.pi*Freqs,fs=2*np.pi*SamplingRate)        
        NotchFilters=np.concatenate((NotchFilters,np.array([np.abs(h)])),axis=0)
    NotchFilters=np.delete(NotchFilters,0,0)
    #Produce a total filter by multiplying all the filters together
    TotalFilter=np.prod(NotchFilters,axis=0)
    return TotalFilter

def LowpassButter(CritFreq,N,Freqs):
    """Returns a lowpass Butterworth filter at critical frequency CritFreq and a scaling factor N, which is evaluated at frequencies Freqs.
    Parameters:
    CritFreq=Critical frequency up to where the Lowpass Butterworth filter has to filter out.
    N= Scaling factor, characterisitc for the Butterworth filter.
    Freqs=Frequencie at which this filter has to be evaluated.
    """
    from scipy import signal
    b,a=signal.butter(N, 2*np.pi*CritFreq, btype='low', analog=True, output='ba', fs=None)
    w, h = signal.freqs(b, a,worN=2*np.pi*Freqs)
    return np.abs(h)


# # GalaxyNoiseSpectrum:

def SimNoiseTrace(StNr,ChNr,Run,EvtNr,UnixTime=None,ThermalNoiseT=275,Plot=False,GNDetector=None,GNStation=None,channelGalacticNoiseAdder=None):
    """Simulates a timetrace containing only galactic (and thermal) noise.
    Parameters:
    StNr= Station number
    ChNr= Channel number
    Run= Data run for which this should be simulated
    EvtNr= Event Number for which this should be simulated
    ThermalNoise= Boolean, if True: thermal noise is included in the simulation.
    """    
    import NuRadioReco.modules.channelGalacticNoiseAdder as ChannelGalacticNoiseAdder
    import NuRadioReco.modules.channelGenericNoiseAdder as ChannelGenericNoiseAdder
    import NuRadioReco.examples.cr_efficiency_analysis.helper_cr_eff as hcr
    from NuRadioReco.detector import detector
    from NuRadioReco.utilities import units
    from NuRadioReco.framework import event,station, channel
    import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
    from datetime import datetime
    from datetime import timedelta
    import scipy.fft as scfft 
    
    
    # CombinedFile=GetCombinedFile(StNr,Run)
    # RadiantData=CombinedFile['combined']['waveforms']['radiant_data[24][2048]'].array(library='np')
    # EventNrs=CombinedFile['combined']['waveforms']['event_number'].array(library="np")
    # TriggerTimes=CombinedFile['combined']['header']["trigger_time"].array(library='np')  
    
    if UnixTime==None:
        #Import data
        WaveFormFile=GetWaveformsFile(StNr,Run)
        HeaderFile=GetHeaderFile(StNr,Run)
        EventNrs=WaveFormFile['waveforms']['event_number'].array(library="np")
        TriggerTimes=HeaderFile['header']["trigger_time"].array(library='np')
        EvIdx=np.where(EventNrs==EvtNr)[0][0]
        UnixTime=TriggerTimes[EvIdx]
        
    Date=datetime.utcfromtimestamp(UnixTime)# - timedelta(hours=2, minutes=0)
    #define relevant parameters
    SamplingTimes=(1/(3.2*1e9))*np.arange(2048) #SampleTimes in seconds

    if GNDetector==None:
        #Obtaining path to relevant json file for detector description
        detpath = os.path.dirname(detector.__file__)
        detpath+="/RNO_G/RNO_season_2022.json"
        GNDetector = detector.Detector(json_filename = detpath)#,antenna_by_depth=False)
    GNDetector.update(Date) #date in example
        
    if GNStation==None:
        GNStation=station.Station(StNr)
        GNChannel=channel.Channel(ChNr)
        GNChannel.set_trace(trace=np.zeros(2048), sampling_rate=3.2 * units.GHz)
        GNStation.add_channel(GNChannel)
    else:
        GNChannel=GNStation.get_channel(ChNr)
    GNStation.set_station_time(Date)
    GNChannel.set_trace(trace=np.zeros(2048), sampling_rate=3.2 * units.GHz)
    
    GNEvent=event.Event(Run,EvtNr)
    
    #Set relevant thermal noise parameters to the default value from the example
    Noise_min_freq,Noise_max_freq,Noise_df=10 * units.MHz,1100 * units.MHz,100*units.MHz
    # Noise_min_freq,Noise_max_freq,Noise_df=10 * units.MHz,150 * units.MHz,10*units.MHz
    if ThermalNoiseT!=0:
        channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
        channelGenericNoiseAdder.begin()
        channelGenericNoiseAdder.run(GNEvent, GNStation, GNDetector, amplitude=hcr.calculate_thermal_noise_Vrms(T_noise=ThermalNoiseT, T_noise_max_freq=Noise_max_freq, T_noise_min_freq=Noise_min_freq),min_freq=Noise_min_freq, max_freq=Noise_max_freq,type='rayleigh')

    #Add Galactic noise
    if channelGalacticNoiseAdder==None:
        channelGalacticNoiseAdder = ChannelGalacticNoiseAdder.channelGalacticNoiseAdder()
        
    # channelGalacticNoiseAdder.begin(debug=False,n_side=16,interpolation_frequencies=np.arange(Noise_min_freq, Noise_max_freq, Noise_df))
    channelGalacticNoiseAdder.begin(debug=False,n_side=16,interpolation_frequencies=np.concatenate((np.arange(10 * units.MHz,150 * units.MHz,20 * units.MHz),np.arange(150 * units.MHz,1200 * units.MHz,200 * units.MHz)))) #For interpolation range over entire range but also focused on freq regime where GN is dominant 
    
    channelGalacticNoiseAdder.run(GNEvent,GNStation,GNDetector,passband=[10 * units.MHz, 1000 * units.MHz])
    # channelGalacticNoiseAdder.run(GNEvent,GNStation,GNDetector,passband=[10 * units.MHz, 150 * units.MHz])
    
    #Plot results    
    if Plot:        
            plt.figure(figsize=(20,5))
            plt.title("Galactic" + (" + thermal" if ThermalNoise else "") + "noise of Station " + str(StNr) + ", channel " + str(ChNr) + ", run " + str(Run) + ", event " + str(EvtNr) + " at " + str(Date.replace(microsecond=0) ) + " UTC, using NuRadioMC PreHardware",fontsize=20)
            plt.plot(1e9*SamplingTimes,GNChannel.get_trace())
            plt.xlabel("Time (ns)", fontsize=20)
            plt.ylabel("Amplitude (V)", fontsize=20)
            plt.show()
    
    #Convolve result with the hardwareresponses
    hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
    hardwareResponseIncorporator.run(GNEvent, GNStation, GNDetector, sim_to_data=True)

    #Plot results    
    if Plot:        
            plt.figure(figsize=(20,5))
            plt.title("Galactic" + (" + thermal" if ThermalNoise else "") + " noise of Station " + str(StNr) + ", channel " + str(ChNr) + ", run " + str(Run) + ", event " + str(EvtNr) + " at " + str(Date.replace(microsecond=0) ) + " UTC, using NuRadioMC",fontsize=20)
            plt.plot(1e9*SamplingTimes,GNChannel.get_trace())
            plt.xlabel("Time (ns)", fontsize=20)
            plt.ylabel("Amplitude (V)", fontsize=20)
            plt.show()
    return SamplingTimes, GNChannel.get_trace()

def GalacticNoiseVRMSCurve(StNr,ChNrs,PDData,FFTFilter=False,Lowpass=False,ZeroAvg=True, ThermalNoiseT=None,Plot=True,json="",dAngle=0):
    """Simulates a transit curve containing galactic and/or thermal noise.
    Parameters:
    StNr= Station number
    ChNr= Channel number
    Rusn= List of runs for which this should be simulated
    EvtNrs= Event Numbers for which this should be simulated
    FFTFilter=Boolean: if true, applies a Notch filter to all frequency spectra to cut out frequencies which have shown to be potentially problematic
    Lowpass= Boolean: if true, a butterworth lowpass filter will be applied in order to maintain only galactic noise dominated frequencies
    ZeroAvg=ZeroAverages the simulates noise timetrace before its VRMS is calculated
    ThermalNoise= Boolean, if True: thermal noise is included in the simulation.
    json=str, if "": use pymongo detector description, if "normal": use RNO_season_YEAR.json description else: use json description with this name
    """   
    
    import NuRadioReco.modules.channelGalacticNoiseAdder as ChannelGalacticNoiseAdder
    import NuRadioReco.modules.channelGenericNoiseAdder as ChannelGenericNoiseAdder
    import NuRadioReco.examples.cr_efficiency_analysis.helper_cr_eff as hcr
    from NuRadioReco.utilities import units
    from NuRadioReco.framework import event, station, channel
    from NuRadioReco.detector import detector
    from datetime import datetime
    import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
    # import scipy.fft as scfft
    sampling_rate=3.2 * (1e9) #Sampling rate in Hertz according to the python file of NuRadioReco.modules.io.rno_g
    
    RMSDic= {str(key):np.array([]) for key in ChNrs}
    Date=datetime.utcfromtimestamp(PDData["Unix"].iloc[0])# - timedelta(hours=2, minutes=0)
    
    #Obtaining path to relevant json file for detector description
    if json=="":
        kwargs = {"source": "rnog_mongo",'select_stations':StNr}
        GNDetector = detector.Detector(**kwargs)
        GNDetector.update(Date)
        if dAngle!=0:
            for ChNr in ChNrs:
                GNDetector.get_channel_orientation(StNr, ChNr)
                GNDetector._Detector__buffered_stations[StNr]["channels"][ChNr]['channel_position']['rotation']['phi']+=int(dAngle)
                print('Ch',ChNr,'orientation:',GNDetector.get_channel_orientation(StNr, ChNr))
    elif json=="normal":
        detpath = os.path.dirname(detector.__file__)
        detpath+="/RNO_G/RNO_season_{0}.json".format(Date.year)
        GNDetector = detector.Detector(json_filename = detpath)#,antenna_by_depth=False)
        print("Using","RNO_season_{0}.json".format(Date.year))
        GNDetector.update(Date)
    else:
        detpath=os.path.dirname(detector.__file__) + "/RNO_G/"
        detpath+=json
        GNDetector = detector.Detector(json_filename = detpath)#,antenna_by_depth=False)
        print("Using",json)
        GNDetector.update(Date)


    #Defining the instances of classes necessary for the simulation

    
    
    GNStation=station.Station(StNr)
    # GNStation=GNDetector.get_station(StNr)
        
    for ChNr in ChNrs:
        GNChannel=channel.Channel(ChNr)
        GNChannel.set_trace(trace=np.zeros(2048), sampling_rate=GNDetector.get_sampling_frequency(StNr, ChNr))
        GNStation.add_channel(GNChannel) 
    
    channelGalacticNoiseAdder = ChannelGalacticNoiseAdder.channelGalacticNoiseAdder()
    channelGalacticNoiseAdder.begin(debug=False,n_side=16,interpolation_frequencies=np.arange(40 * units.MHz,160 * units.MHz,10 * units.MHz)) #For interpolation range over entire range but also focused on freq regime where GN is dominant 
    hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
    # channelGalacticNoiseAdder._channelGalacticNoiseAdder__antenna_pattern_provider.load_antenna_pattern(GNDetector.get_antenna_model(GNStation.get_id(), GNChannel.get_id()))
    N=0
    for idx,row in PDData.iterrows():
        # print('Data Progress:',np.round(100*N/len(PDData),2),'%', end="\r")
        # N+=1
                
        GNEvent=event.Event(row["RunNr"],row["EvNr"])
        Date=datetime.utcfromtimestamp(row["Unix"])
        GNDetector.update(Date) #date in example
        GNStation.set_station_time(Date)

        channelGalacticNoiseAdder.run(GNEvent,GNStation,GNDetector,passband=[50 * units.MHz, 150 * units.MHz])
        # channelGalacticNoiseAdder.run(GNEvent,GNStation,GNDetector,passband=[10 * units.MHz, 150 * units.MHz])
        
        #Set relevant thermal noise parameters to the default value from the example
        Noise_min_freq,Noise_max_freq,Noise_df=10 * units.MHz,1100 * units.MHz,100*units.MHz
        # Noise_min_freq,Noise_max_freq,Noise_df=10 * units.MHz,150 * units.MHz,10*units.MHz
        if ThermalNoiseT==None:
            continue
        elif type(ThermalNoiseT)==int or type(ThermalNoiseT)==float:
            ThermalNoiseT={ChNr:ThermalNoiseT for ChNr in ChNrs}
            #ADD THERMAL NOISE!!!!
        elif type(ThermalNoiseT)==dict:
            channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
            channelGenericNoiseAdder.begin()
            Amplitudes={ChNr: hcr.calculate_thermal_noise_Vrms(T_noise=ThermalNoiseT[ChNr], T_noise_max_freq=Noise_max_freq, T_noise_min_freq=Noise_min_freq) for ChNr in ChNrs}
            channelGenericNoiseAdder.run(GNEvent, GNStation, GNDetector, amplitude=Amplitudes,min_freq=Noise_min_freq, max_freq=Noise_max_freq,type='rayleigh')
        else:
            print('Warning: No thermal noise was added')
        
        hardwareResponseIncorporator.run(GNEvent, GNStation, GNDetector, sim_to_data=True)
        for channel in  GNStation.iter_channels():
            SamplingTimes, GNTrace=channel.get_times()*(units.ns/units.s), channel.get_trace()
            if ZeroAvg==True: #Zero average the timetrace 
                    GNTrace-=np.mean(GNTrace)
            #If a filter is required: convert to frequency domain and apply filter
            if FFTFilter or Lowpass:
                    import scipy.fft as scfft
                    GNFreq=scfft.fftfreq(len(SamplingTimes),(SamplingTimes[-1]-SamplingTimes[0])/len(SamplingTimes))
                    TotalFilter=np.ones(len(GNFreq))
                    if FFTFilter:
                        TotalFilter=np.multiply(TotalFilter,NotchFilters([403*10**6,120*10**6,807*10**6,1197*10**6],75,GNFreq,sampling_rate))
                    if Lowpass:
                        CritFreq=150*(10**6)
                        TotalFilter=np.multiply(TotalFilter,LowpassButter(CritFreq,20,GNFreq))
                    GNFFT=scfft.fft(GNTrace)
                    GNFFT=np.array([GNFFT[i]*TotalFilter[i] for i in range(len(GNFreq))])
                    GNTrace=np.abs(scfft.ifft(GNFFT))
            RMSDic[str(channel.get_id())]=np.append(RMSDic[str(channel.get_id())],np.sqrt(np.mean([V**2 for V in GNTrace])))
            channel.set_trace(trace=np.zeros(2048), sampling_rate=3.2 * units.GHz)
            
    PDSimResultDic={str(key):PDData.assign(VRMS=RMSDic[str(key)]) for key in ChNrs}                
    if Plot:
        plt.figure(figsize=(15,5))
        plt.plot(PDData["LST"],1000*EventRMS,'.', markersize=20)
        plt.grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
        plt.title("Galactic noise V_RMS of Station " + str(StNr) + ", channel " + str(ChNr) + " for " + str(len(EventRMS)) + " events")
        plt.xlabel("LST Time (hrs)",fontsize=20)#20)
        plt.ylabel("V_RMS (mV)",fontsize=20)#20)
        plt.xticks(np.arange(0, 24, 1.0),fontsize=25)#15)
        plt.yticks(fontsize=25)#15)
        plt.show()
    return PDSimResultDic        
    
##### DataSimComparison

def TransitCurveSingular(StNr,ChNr, DataSample,DataFileId,NBins=24):
    """ Plots the data transit curve results from files stored away in the JobResults folder structure.
    Parameters:
    StNr= Station number.
    ChNr= Channel number
    DataSample= String that can be "C" for combined, "HC" for handcarry or "S" for Satellite 
    DataFileId= Name of the file where the data is stored.
    NBins= Amount of bins the transit curve should have.
    """
    #Plots the transit curves from JobsResults for a file with name "FileId, station StNr, channel ChNr, binned in NBins bins(set NBins=0 for the already binned data)"
    #DataSample can be "C" for combined, "HC" for handcarry or "S" for Satellite     
    import numpy as np
    import matplotlib.pyplot as plt

    #Read in data. If NBins=0, it is assumed that the data is already grouped
    if NBins !=0:
        EventRMS=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventRMS_" + str(DataFileId) + ".npy",allow_pickle=True)
        EventTime=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventTime_" + str(DataFileId) + ".npy",allow_pickle=True)
        MidBins, GroupedVRMS = GroupVRMS(EventVRMS,EventTime,NBins)
        DataEntries=len(EventRMS)
    else:
        GroupedVRMS=np.load("JobResults/Data/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "GroupedVRMS_" + str(DataFileId) + ".npy",allow_pickle=True)
        MidBins=np.load("JobResults/Data/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "MidBins_" + str(DataFileId) + ".npy",allow_pickle=True)
        DataEntries=0
        for i in range(len(GroupedVRMS)):
            DataEntries+=len(GroupedVRMS[i])
    
    #Compute statistics
    VRMSAvg=np.array([np.mean(GroupedVRMS[i]) if len(GroupedVRMS[i])!=0 else 0 for i in range(len(GroupedVRMS))])
    VRMSStd=np.array([np.std(GroupedVRMS[i])/len(GroupedVRMS[i]) if len(GroupedVRMS[i])!=0 else 0 for i in range(len(GroupedVRMS))])

    plt.figure(figsize=(15,5))
    if NBins !=0:
        plt.errorbar(MidBins,VRMSAvg,yerr=VRMSStd,fmt=".",zorder=2,markersize=10)
        plt.title("Simulation of Galactic noise for station " + str(StNr) + ", antenna " + str(ChNr),fontsize=25)
        plt.figtext(0.2, 0.8, "Simulations:" + str(len(EventRMS)), fontsize=18,bbox=dict(edgecolor='black', facecolor='none', alpha=0.2, pad=10.0))
    else:
        plt.errorbar(MidBins,1000*VRMSAvg,yerr=1000*VRMSStd,fmt=".",zorder=2,markersize=10)
        plt.title("Transit curve for station " + str(StNr) + ", antenna " + str(ChNr),fontsize=25)
        plt.figtext(0.2, 0.8, "Entries:" + str(DataEntries), fontsize=18,bbox=dict(edgecolor='black', facecolor='none', alpha=0.2, pad=10.0))
    plt.grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
    plt.xlabel("LST Time (hrs)",fontsize=20)#20)
    plt.ylabel("V_RMS (mV)",fontsize=20)#20)
    plt.xticks(np.arange(0, 24, 1.0),fontsize=25)#15)
    plt.yticks(fontsize=25)#15)
    plt.show()
    return
        
def TransitCurveComparison(StNr,ChNr,DataSample, DataFileId,SimFileId):
    """ Plots the data and simulation transit curve results from files stored away in the JobResults folder structure.
    Parameters:
    StNr= Station number.
    ChNr= Channel number
    DataSample= String that can be "C" for combined, "HC" for handcarry or "S" for Satellite 
    DataFileId= Name of the file where the data is stored.
    SimFileId= Name of the file where the simulated results are stored.
    """
    import numpy as np
    import matplotlib.pyplot as plt
        
    #Import data
    SimEventRMS=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventRMS_" + str(SimFileId) + ".npy",allow_pickle=True)
    SimEventTime=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventTime_" + str(SimFileId) + ".npy",allow_pickle=True)
    DataGroupedVRMS=np.load("JobResults/Data/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "GroupedVRMS_" + str(DataFileId) + ".npy",allow_pickle=True)
    DataMidBins=np.load("JobResults/Data/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "MidBins_" + str(DataFileId) + ".npy",allow_pickle=True)
    NBins=len(DataMidBins)
       
    #Group data
    SimMidBins, SimGroupedVRMS = GroupVRMS(SimEventRMS,SimEventTime,NBins)
        
    #Calculate statistics
    SimVRMSAvg=np.array([np.mean(SimGroupedVRMS[i]) if len(SimGroupedVRMS[i])!=0 else 0 for i in range(len(SimGroupedVRMS))])
    SimVRMSStd=np.array([np.std(SimGroupedVRMS[i])/np.sqrt(len(SimGroupedVRMS[i])) if len(SimGroupedVRMS[i])!=0 else 0 for i in range(len(SimGroupedVRMS))])
    DataVRMSAvg=np.array([np.mean(DataGroupedVRMS[i]) if len(DataGroupedVRMS[i])!=0 else 0 for i in range(len(DataGroupedVRMS))])
    DataVRMSStd=np.array([np.std(DataGroupedVRMS[i])/np.sqrt(len(DataGroupedVRMS[i])) if len(DataGroupedVRMS[i])!=0 else 0 for i in range(len(DataGroupedVRMS))])
    DataVRMSMedian=np.array([np.median(DataGroupedVRMS[i]) if len(DataGroupedVRMS[i])!=0 else 0 for i in range(len(DataGroupedVRMS))])
    SimVRMSMedian=np.array([np.median(SimGroupedVRMS[i]) if len(SimGroupedVRMS[i])!=0 else 0 for i in range(len(SimGroupedVRMS))])
        
    #Compute amount of entries in the transit curve
    DataEntries=0
    for i in range(len(DataGroupedVRMS)):
        DataEntries+=len(DataGroupedVRMS[i])
              
    #Set simulated transit curve onto the same average as the data and keep track of this offset
    SimOffset=np.mean(SimVRMSAvg)-np.mean(DataVRMSAvg)
    SimVRMSAvg-=SimOffset

    if SimOffset>0:
        OffsetStr="-" + str(np.round(1000*SimOffset,3))
    else:
        OffsetStr= "+" + str(np.abs(np.round(1000*SimOffset,3)))
    fig, axs = plt.subplots(2,1,figsize=(15,7.5), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle("Transit curve for station " + str(StNr) + ", antenna " + str(ChNr),x=0.5,y=0.94,fontsize=25)
    plt.subplots_adjust(hspace=0)
    for ax in axs:
        ax.grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
        ax.set_xticks(np.arange(0, 24, 1.0))
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
    axs[0].set_ylabel("V_RMS (mV)",fontsize=20)
    axs[1].set_ylabel("Data/Sim",fontsize=20)
    plt.xticks(np.arange(0, 24, 1.0),fontsize=25)#15)
    plt.xlabel("LST Time (hrs)",fontsize=20)#20)
    axs[0].errorbar(DataMidBins,1000*DataVRMSAvg,yerr=1000*DataVRMSStd,fmt=".",zorder=2,label="Data")#,markersize=15)
    axs[0].errorbar(SimMidBins,1000*SimVRMSAvg,yerr=1000*SimVRMSStd,fmt=".",zorder=2,label="Simulation " + OffsetStr + "mV")#,markersize=15)
    VRMSRatio=DataVRMSAvg/SimVRMSAvg
    VRMSRatioStd=VRMSRatio*np.sqrt((DataVRMSStd/DataVRMSAvg)**2 + (SimVRMSStd/SimVRMSAvg)**2)
    axs[1].axhline(y = 1, color = 'k', linestyle = ':',alpha=1)
    axs[1].errorbar(SimMidBins,VRMSRatio,yerr=1000*SimVRMSStd,fmt=".",zorder=2,label="Simulation " + OffsetStr + "mV")#,markersize=15)
    axs[0].legend(loc="lower left",fontsize=15)
    return

def TransitCurveComparisonCleanPlot(StNr,ChNr, NBins=4*24,TimeFormat="LST"):
    """ Plots the data and simulation transit curve results from files stored away in the JobResults folder structure. Clean version for the paper.
    Parameters:
    StNr= Station number.
    ChNr= Channel number
    DataSample= String that can be "C" for combined, "HC" for handcarry or "S" for Satellite 
    DataFileId= Name of the file where the data is stored.
    SimFileId= Name of the file where the simulated results are stored.
    Scaling= "Int" for scaling by integrated transit curve, "pdf" to make both a pdf & "Avg" by matching baselines
     """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    # from matplotlib.ticker import AutoMinorLocator
    
    #Import data   
    GNData=pd.read_pickle("JobResults/Data/St"+str(StNr)+"/St"+str(StNr)+"Ch"+str(ChNr)+".pkl")
    GNSim=pd.read_pickle("JobResults/Sim/St"+str(StNr)+"/St"+str(StNr)+"Ch"+str(ChNr)+".pkl")
           
    BinEdges=np.linspace(0,24,NBins+1,True)
    GNData['Bin'] = pd.cut(GNData[TimeFormat], BinEdges)
    GNSim['Bin'] = pd.cut(GNSim[TimeFormat], BinEdges)
    
    #Compute relevant statistics
    DataBinStat=GNData.groupby(["Bin"]).agg({"VRMS":[np.median,(lambda x: np.std(x)/np.sqrt(len(x))) ]})
    DataPlotFilter=DataBinStat[('VRMS', 'median')].isnull()
    DataVRMSMed=DataBinStat[('VRMS', 'median')].drop(DataBinStat[('VRMS', 'median')][DataPlotFilter].index)
    DataVRMSStd=DataBinStat[('VRMS', '<lambda_0>')].drop(DataBinStat[('VRMS', 'median')][DataPlotFilter].index).fillna(0)
    
    SimBinStat=GNSim.groupby(["Bin"]).agg({"VRMS":[np.median,(lambda x: np.std(x)/np.sqrt(len(x))) ]})
    SimPlotFilter=SimBinStat[('VRMS', 'median')].isnull()
    SimVRMSMed=SimBinStat[('VRMS', 'median')].drop(SimBinStat[('VRMS', 'median')][SimPlotFilter].index)
    SimVRMSStd=SimBinStat[('VRMS', '<lambda_0>')].drop(SimBinStat[('VRMS', 'median')][SimPlotFilter].index).fillna(0)
    
    MidBins=np.array([(BinEdges[i] + BinEdges[i+1])/2 for i in range(0,len(BinEdges)-1)])[np.logical_not(DataPlotFilter)]  
        
    plt.figure(figsize=(15,5))
    mpl.rcParams['axes.linewidth'] = 5
    plt.title("Transit curve for station " + str(StNr) + ", antenna " + str(ChNr),fontsize=25,weight="bold")
    plt.subplots_adjust(hspace=0)

    plt.xticks(np.arange(0, 25, 2.0),fontsize=25,weight="bold")
    plt.xticks(np.arange(0, 25, 1.0),fontsize=25,weight="bold")
    plt.yticks(fontsize=25,weight="bold")
    plt.minorticks_on()
    ax=plt.gca()
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    if TimeFormat=="LST":
        TimeLabel="Local Sidereal"
    elif TimeFormat=="LT":
        TimeLabel="Local"
    plt.xlabel(TimeLabel + " Time (h)",fontsize=20,weight="bold")#20)
    plt.ylabel("V_RMS (mV)",fontsize=20,weight="bold")#20)
    plt.grid(which="both",color='grey', linestyle='-', linewidth=1,alpha=0.5)
    plt.xticks(np.arange(0, 25, 2.0),fontsize=25,weight="bold")
    plt.errorbar(MidBins,1e3*DataVRMSMed,yerr=1e3*DataVRMSStd,linestyle="",marker=".",color="#7565ad",zorder=2,label="Data")
    plt.errorbar(MidBins,1e3*SimVRMSMed,yerr=1e3*SimVRMSStd,linestyle="",marker="x",markersize=5,color="#d76caa",zorder=2,label="Sim")
    plt.legend(loc="lower left",fontsize=15,prop={"weight":"bold","size":15})
    
    plt.xlim(0,24)
    
    plt.savefig("Figures/TransitCurveSt"+str(StNr)+"Ch" + str(ChNr)+".svg", format="svg", bbox_inches="tight")
    return
    
def TransitCurveRatioSingular(StNr,ChNr1,ChNr2,DataSample,DataFileId1=0,DataFileId2=0,SimFileId1=0,SimFileId2=0):
    """ Plots the ratio of the data transit curve or simulation transit curve results for ChNr1/ChNr2 from files stored away in the JobResults folder structure.
    Parameters:
    StNr= Station number.
    ChNr1= Channel number for the VRMS value in the enumerator
    ChNr2= Channel number for the VRMS value in the denominator
    DataSample= String that can be "C" for combined, "HC" for handcarry or "S" for Satellite 
    DataFileId= Name of the file where the data is stored.
    SimFileId= Name of the file where the simulated results are stored.
    """  
    import numpy as np
    import matplotlib.pyplot as plt
        
    #Import data 
    if isinstance(DataFileId1, str) and isinstance(DataFileId2, str):
        GroupedVRMS1=np.load("JobResults/Data/St" + str(StNr) + "Ch" + str(ChNr1) + "/" + DataSample + "GroupedVRMS_" + str(DataFileId1) + ".npy",allow_pickle=True)
        GroupedVRMS2=np.load("JobResults/Data/St" + str(StNr) + "Ch" + str(ChNr2) + "/" + DataSample + "GroupedVRMS_" + str(DataFileId2) + ".npy",allow_pickle=True)
        MidBins=np.load("JobResults/Data/St" + str(StNr) + "Ch" + str(ChNr1) + "/" + DataSample + "MidBins_" + str(DataFileId1) + ".npy",allow_pickle=True)
    #Import simulated results
    elif isinstance(SimFileId1, str) and isinstance(SimFileId2, str):
        SimEventRMS1=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr1) + "/" + DataSample + "SimEventRMS_" + str(SimFileId1) + ".npy",allow_pickle=True)
        SimEventTime1=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr1) + "/" + DataSample + "SimEventTime_" + str(SimFileId1) + ".npy",allow_pickle=True)
        SimEventRMS2=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr2) + "/" + DataSample + "SimEventRMS_" + str(SimFileId2) + ".npy",allow_pickle=True)
        SimEventTime2=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr2) + "/" + DataSample + "SimEventTime_" + str(SimFileId2) + ".npy",allow_pickle=True)
        
        #Group simulated results
        NBins=4*24
        MidBins1, GroupedVRMS1 = GroupVRMS(SimEventRMS1,SimEventTime1,NBins)
        MidBins2, GroupedVRMS2 = GroupVRMS(SimEventRMS2,SimEventTime2,NBins)
        MidBins=MidBins1   
    else:
        print("No Filenames given")
        return
    if len(GroupedVRMS1) != len(GroupedVRMS2):
        print("Error: Non matching Grouped VRMS length")
            
    #Count entries in each grouped dataset
    Entries1=0
    for i in range(len(GroupedVRMS1)):
        Entries1+=len(GroupedVRMS1[i])
    Entries2=0
    for i in range(len(GroupedVRMS2)):
        Entries2+=len(GroupedVRMS2[i])    

    #Calculate statistics    
    VRMSAvg1=np.array([np.mean(GroupedVRMS1[i]) if len(GroupedVRMS1[i])!=0 else 0 for i in range(len(GroupedVRMS1))])
    VRMSStd1=np.array([np.std(GroupedVRMS1[i])/np.sqrt(len(GroupedVRMS1[i])) if len(GroupedVRMS1[i])!=0 else 0 for i in range(len(GroupedVRMS1))])
    VRMSAvg2=np.array([np.mean(GroupedVRMS2[i]) if len(GroupedVRMS2[i])!=0 else 0 for i in range(len(GroupedVRMS2))])
    VRMSStd2=np.array([np.std(GroupedVRMS2[i])/np.sqrt(len(GroupedVRMS2[i])) if len(GroupedVRMS2[i])!=0 else 0 for i in range(len(GroupedVRMS2))])
    
    #Calculate ratio & std
    VRMSRatio=VRMSAvg1/VRMSAvg2
    VRMSRatioStd=VRMSRatio*np.sqrt((VRMSStd1/VRMSAvg1)**2 + (VRMSStd2/VRMSAvg2)**2)
        
    plt.figure(figsize=(15,5))
    plt.errorbar(MidBins,VRMSRatio,yerr=VRMSRatioStd,fmt=".",zorder=2,markersize=10)
    plt.title("Transit curve ratio for St" + str(StNr) + "Ch" + str(ChNr1) + "/St" + str(StNr) + "Ch" + str(ChNr2),fontsize=25)
    plt.figtext(0.2, 0.8, "Entries: " + str(Entries1) + "/" + str(Entries2), fontsize=18,bbox=dict(edgecolor='black', facecolor='none', alpha=0.2, pad=10.0))    
    plt.grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
    plt.xlabel("LST Time (hrs)",fontsize=20)#20)
    plt.ylabel("VRMS Ratio",fontsize=20)#20)
    plt.xticks(np.arange(0, 24, 1.0),fontsize=25)#15)
    plt.yticks(fontsize=25)#15)
    plt.show()
    return
    
def TransitCurveRatioComparison(StNr,ChNr1,ChNr2,DataSample,DataFileId1=0,DataFileId2=0,SimFileId1=0,SimFileId2=0):
    """ Plots the ratios of the data and simulation transit curve results for ChNr1/ChNr2 from files stored away in the JobResults folder structure in order to compare the ratios.
    Parameters:
    StNr= Station number.
    ChNr1= Channel number for the VRMS value in the enumerator
    ChNr2= Channel number for the VRMS value in the denominator
    DataSample= String that can be "C" for combined, "HC" for handcarry or "S" for Satellite 
    DataFileId1= Name of the file where the data is stored for ChNr1.
    DataFileId2= Name of the file where the data is stored for ChNr2.
    SimFileId1= Name of the file where the simulated results are stored for ChNr1.
    SimFileId2= Name of the file where the simulated results are stored for ChNr2.
    """  
    import numpy as np
    import matplotlib.pyplot as plt
    import NuRadioReco.examples.cr_efficiency_analysis.helper_cr_eff as hcr
    from NuRadioReco.utilities import units
        
    #Import data & simulation results
    DataGroupedVRMS1=np.load("JobResults/Data/St" + str(StNr) + "Ch" + str(ChNr1) + "/" + DataSample + "GroupedVRMS_" + str(DataFileId1) + ".npy",allow_pickle=True)
    DataGroupedVRMS2=np.load("JobResults/Data/St" + str(StNr) + "Ch" + str(ChNr2) + "/" + DataSample + "GroupedVRMS_" + str(DataFileId2) + ".npy",allow_pickle=True)
    DataMidBins=np.load("JobResults/Data/St" + str(StNr) + "Ch" + str(ChNr1) + "/" + DataSample + "MidBins_" + str(DataFileId1) + ".npy",allow_pickle=True)
    SimEventRMS1=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr1) + "/" + DataSample + "SimEventRMS_" + str(SimFileId1) + ".npy",allow_pickle=True)
    SimEventTime1=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr1) + "/" + DataSample + "SimEventTime_" + str(SimFileId1) + ".npy",allow_pickle=True)
    SimEventRMS2=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr2) + "/" + DataSample + "SimEventRMS_" + str(SimFileId2) + ".npy",allow_pickle=True)
    SimEventTime2=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr2) + "/" + DataSample + "SimEventTime_" + str(SimFileId2) + ".npy",allow_pickle=True)
      
    #Group simulated results
    NBins=len(DataMidBins)
    SimMidBins1, SimGroupedVRMS1 = GroupVRMS(SimEventRMS1,SimEventTime1,NBins)
    SimMidBins2, SimGroupedVRMS2 = GroupVRMS(SimEventRMS2,SimEventTime2,NBins)
    SimMidBins=SimMidBins1
        
    #Count amount of entries for each dataset and simulation result
    SimEntries1=len(SimEventRMS1)
    SimEntries2=len(SimEventRMS2)    
    DataEntries1=0
    for i in range(len(DataGroupedVRMS1)):
        DataEntries1+=len(DataGroupedVRMS1[i])
    DataEntries2=0
    for i in range(len(DataGroupedVRMS2)):
        DataEntries2+=len(DataGroupedVRMS2[i])  
    
    #Calculate data statistics
    DataVRMSAvg1=np.array([np.mean(DataGroupedVRMS1[i]) if len(DataGroupedVRMS1[i])!=0 else 0 for i in range(len(DataGroupedVRMS1))])
    DataVRMSStd1=np.array([np.std(DataGroupedVRMS1[i])/np.sqrt(len(DataGroupedVRMS1[i])) if len(DataGroupedVRMS1[i])!=0 else 0 for i in range(len(DataGroupedVRMS1))])
    DataVRMSAvg2=np.array([np.mean(DataGroupedVRMS2[i]) if len(DataGroupedVRMS2[i])!=0 else 0 for i in range(len(DataGroupedVRMS2))])
    DataVRMSStd2=np.array([np.std(DataGroupedVRMS2[i])/np.sqrt(len(DataGroupedVRMS2[i])) if len(DataGroupedVRMS2[i])!=0 else 0 for i in range(len(DataGroupedVRMS2))])

    #Calculate data ratio & std
    DataVRMSRatio=DataVRMSAvg1/DataVRMSAvg2
    DataVRMSRatioStd=DataVRMSRatio*np.sqrt((DataVRMSStd1/DataVRMSAvg1)**2 + (DataVRMSStd2/DataVRMSAvg2)**2)
        
    #Calculate simulation statistics
    SimVRMSAvg1=np.array([np.mean(SimGroupedVRMS1[i]) if len(SimGroupedVRMS1[i])!=0 else 0 for i in range(len(SimGroupedVRMS1))])
    SimVRMSStd1=np.array([np.std(SimGroupedVRMS1[i])/np.sqrt(len(SimGroupedVRMS1[i])) if len(SimGroupedVRMS1[i])!=0 else 0 for i in range(len(SimGroupedVRMS1))])
    SimVRMSAvg2=np.array([np.mean(SimGroupedVRMS2[i]) if len(SimGroupedVRMS2[i])!=0 else 0 for i in range(len(SimGroupedVRMS2))])
    SimVRMSStd2=np.array([np.std(SimGroupedVRMS2[i])/np.sqrt(len(SimGroupedVRMS2[i])) if len(SimGroupedVRMS2[i])!=0 else 0 for i in range(len(SimGroupedVRMS2))])

    #Set simulated transit curve onto the same average as the data and keep track of this offset
    SimOffset1=np.mean(SimVRMSAvg1)-np.mean(DataVRMSAvg1)
    SimVRMSAvg1-=SimOffset1
    SimOffset2=np.mean(SimVRMSAvg2)-np.mean(DataVRMSAvg2)
    SimVRMSAvg2-=SimOffset2        
        
    #Calculate simulated ratio and Data/Sim ratio
    SimVRMSRatio=SimVRMSAvg1/SimVRMSAvg2
    SimVRMSRatioStd=SimVRMSRatio*np.sqrt((SimVRMSStd1/SimVRMSAvg1)**2 + (SimVRMSStd2/SimVRMSAvg2)**2)   
    VRMSRatio=DataVRMSRatio/SimVRMSRatio
    VRMSRatioStd=VRMSRatio*np.sqrt((DataVRMSRatioStd/DataVRMSRatio)**2 + (SimVRMSRatioStd/SimVRMSRatio)**2)
        
    fig, axs = plt.subplots(2,1,figsize=(15,7.5), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle("Transit curve ratio for St" + str(StNr) + "Ch" + str(ChNr1) + "/St" + str(StNr) + "Ch" + str(ChNr2),x=0.5,y=0.94,fontsize=25)
    plt.subplots_adjust(hspace=0)
    for ax in axs:
        ax.grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
        ax.set_xticks(np.arange(0, 24, 1.0))
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
    axs[0].set_ylabel("V_RMS ratio",fontsize=20)
    axs[1].set_ylabel("Data/Sim",fontsize=20)
    plt.xticks(np.arange(0, 24, 1.0),fontsize=25)#15)
    plt.xlabel("LST Time (hrs)",fontsize=20)#20)
    axs[0].errorbar(DataMidBins,DataVRMSRatio,yerr=DataVRMSRatioStd,fmt=".",zorder=2,label="Data")#,markersize=15)
    axs[0].errorbar(SimMidBins,SimVRMSRatio,yerr=SimVRMSRatioStd,fmt=".",zorder=2,label="Simulation")# +str(np.round(1000*SimOffset,3)) + "mV")
    axs[1].axhline(y = 1, color = 'k', linestyle = ':',alpha=1)
    axs[1].errorbar(SimMidBins,VRMSRatio,yerr=VRMSRatioStd,fmt=".",zorder=2)# ,label="Simulation -" +str(np.round(1000*SimOffset,3)) + "mV",markersize=15)
    axs[0].legend(loc="lower left",fontsize=15)  
    return
    
def GroupVRMS(EventVRMS,EventTime,NBins):
    """Groupes a list of VRMS values and EventTimes into NBins bins.
    Parameters:
    EventVRMS= List of VRMS values that need to be grouped 
    EventTime= List of the respective Event times corresponding to the values in EventVRMS
    NBins= Amount of bins to group the values in.
    """
    EventTimeCounts, EventTimeBins=np.histogram(EventTime, bins=NBins,range=(0,24),density=False) #Storing timestamps in histogram format
    EventTimeDig=np.digitize(EventTime,EventTimeBins)
    GroupedVRMS=np.empty((NBins,),dtype=object)
    for i in range(len(EventTimeDig)):
        GroupedVRMS[EventTimeDig[i]-1]=np.append(GroupedVRMS[EventTimeDig[i]-1],EventVRMS[i])
        
    ##Get rid of "None" entries in beginning of array
    for i in range(len(GroupedVRMS)):
        GroupedVRMS[i]=np.delete(GroupedVRMS[i], 0) 
    MidBins= np.array([(EventTimeBins[i] + EventTimeBins[i+1])/2 for i in range(0,len(EventTimeBins)-1)]) 
        
    return MidBins, GroupedVRMS

def GNSimError(StNr,ChNr,DataSample,SimFileIdMin, SimFileIdZero,SimFileIdPlus,NBins=4*24,Plot=False):
    """ Plots the simulated transit curves to compare the impact of moving the radio skymap temperatures 1 standard deviation up or down. 
    Parameters:
    StNr= Station number.
    ChNr= Channel number
    DataSample= String that can be "C" for combined, "HC" for handcarry or "S" for Satellite 
    SimFileIdMin= Name of the file where the simulated results are stored for the radioskymap -1 standard deviation.
    SimFileIdZero= Name of the file where the simulated results are stored for the standard radioskymap.
    SimFileIdPlus= Name of the file where the simulated results are stored for the radioskymap +1 standard deviation.
    NBins= Amount of bins to group the values in.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import NuRadioReco.examples.cr_efficiency_analysis.helper_cr_eff as hcr
    from NuRadioReco.utilities import units
        
    #Import simulation results
    SimEventRMSMin=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventRMS_" + str(SimFileIdMin) + ".npy",allow_pickle=True)
    SimEventTimeMin=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventTime_" + str(SimFileIdMin) + ".npy",allow_pickle=True)
      
    SimEventRMSZero=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventRMS_" + str(SimFileIdZero) + ".npy",allow_pickle=True)
    SimEventTimeZero=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventTime_" + str(SimFileIdZero) + ".npy",allow_pickle=True)

    SimEventRMSPlus=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventRMS_" + str(SimFileIdPlus) + ".npy",allow_pickle=True)
    SimEventTimePlus=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventTime_" + str(SimFileIdPlus) + ".npy",allow_pickle=True)
        
    #Group simulation results
    MidBinsMin, GroupedVRMSMin = GroupVRMS(SimEventRMSMin,SimEventTimeMin,NBins)
    MidBinsZero, GroupedVRMSZero = GroupVRMS(SimEventRMSZero,SimEventTimeZero,NBins)
    MidBinsPlus, GroupedVRMSPlus = GroupVRMS(SimEventRMSPlus,SimEventTimePlus,NBins)
       
    #Calculate statistics
    SimVRMSAvgMin=np.array([np.mean(GroupedVRMSMin[i]) if len(GroupedVRMSMin[i])!=0 else 0 for i in range(len(GroupedVRMSMin))])
    SimVRMSStdMin=np.array([np.std(GroupedVRMSMin[i])/np.sqrt(len(GroupedVRMSMin[i])) if len(GroupedVRMSMin[i])!=0 else 0 for i in range(len(GroupedVRMSMin))])
    SimVRMSAvgZero=np.array([np.mean(GroupedVRMSZero[i]) if len(GroupedVRMSZero[i])!=0 else 0 for i in range(len(GroupedVRMSZero))])
    SimVRMSStdZero=np.array([np.std(GroupedVRMSZero[i])/np.sqrt(len(GroupedVRMSZero[i])) if len(GroupedVRMSZero[i])!=0 else 0 for i in range(len(GroupedVRMSZero))])
    SimVRMSAvgPlus=np.array([np.mean(GroupedVRMSPlus[i]) if len(GroupedVRMSPlus[i])!=0 else 0 for i in range(len(GroupedVRMSPlus))])
    SimVRMSStdPlus=np.array([np.std(GroupedVRMSPlus[i])/np.sqrt(len(GroupedVRMSPlus[i])) if len(GroupedVRMSPlus[i])!=0 else 0 for i in range(len(GroupedVRMSPlus))])

    #GN simulation error calculation
    GNUpperError,GNLowerError=SimVRMSAvgPlus-SimVRMSAvgZero,SimVRMSAvgZero-SimVRMSAvgMin
    GNSimError=np.array([GNLowerError,GNUpperError])
    
    if Plot:
        AvgError=np.array([np.mean([GNUpperError[i],GNLowerError[i]]) for i in range(len(GNLowerError))])
        AvgRelError=np.mean(AvgError/SimVRMSAvgZero)
        plt.figure(figsize=(15,5))
        plt.figtext(0.6, 0.2, "AvgError:" + str(np.round(np.mean(10**6*AvgError),2)) + "µV; AvgRelError:" + str(np.round(100*AvgRelError,2)) + "%", fontsize=16,bbox=dict(edgecolor='black', facecolor='none', alpha=0.2, pad=10.0))
        plt.errorbar(MidBinsPlus,1000*SimVRMSAvgPlus,yerr=1000*SimVRMSStdPlus,fmt=".",zorder=2,label="Simulation +1 sigma")#,markersize=15)
        plt.errorbar(MidBinsZero,1000*SimVRMSAvgZero,yerr=1000*SimVRMSStdZero,fmt=".",zorder=2,label="Simulation ")#,markersize=15)
        plt.errorbar(MidBinsMin,1000*SimVRMSAvgMin,yerr=1000*SimVRMSStdMin,fmt=".",zorder=2,label="Simulation -1 sigma")#,markersize=15)
        plt.grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
        plt.title("Transit curve for station " + str(StNr) + ", antenna " + str(ChNr),fontsize=25)
        plt.xlabel("LST Time (hrs)",fontsize=20)#20)
        plt.ylabel("V_RMS (mV)",fontsize=20)#20)
        plt.xticks(np.arange(0, 24, 1.0),fontsize=25)#15)
        plt.yticks(fontsize=25)#15)
        plt.legend(loc="lower left",fontsize=15)
        plt.show()
    return GNSimError
    
def PlotErrors(GNSimError,StNr,ChNr):
    """ Plots the errors on the transit curve induced by moving the radioskymap 1 standard deviation up or down.
    Parameters:
    GNSimError= List of the upper and lower errors on the transit curves. 
    StNr= Station number.
    ChNr= Channel number.
    NBins= Amount of bins to group the values in.
    """
    NBins=len(GNSimError[0])
    Stepsize=24.0/NBins
    MidBins=np.arange(0,24,Stepsize) + Stepsize/2

    plt.figure(figsize=(15,5))
    plt.plot(MidBins,1000*GNSimError[0],'r.',label="Lower error")
    plt.plot(MidBins,1000*GNSimError[1],'b.', label="Upper error")
    plt.plot(MidBins,1000*np.mean(GNSimError[0])*np.ones(len(MidBins)),'r-.')
    plt.plot(MidBins,1000*np.mean(GNSimError[1])*np.ones(len(MidBins)),'b-.')
    plt.grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
    plt.title("GN errors for Transit curve for station " + str(StNr) + ", antenna " + str(ChNr),fontsize=25)
    plt.xlabel("LST Time (hrs)",fontsize=20)#20)
    plt.ylabel("V_RMS (mV)",fontsize=20)#20)
    plt.xticks(np.arange(0, 24, 1.0),fontsize=25)#15)
    plt.yticks(fontsize=25)#15)
    plt.legend(loc="lower left",fontsize=15)
    plt.show()    
    return

def TNSimFrac(StNr,ChNr,DataSample,SimFileIdMin, SimFileIdZero,SimFileIdPlus,NBins=4*24,Plot=False):
    """ Plots the fraction of the simulated transit curves where the thermal noise temperature has been moved up and down by 1 standard deviation in order to check for a change in the transit curve form.
    Parameters:
    StNr= Station number
    ChNr= Channel number
    DataSample= String that can be "C" for combined, "HC" for handcarry or "S" for Satellite 
    SimFileIdMin= Name of the file where the simulated results are stored for the thermal noise temperature -1 standard deviation.
    SimFileIdZero= Name of the file where the simulated results are stored for the standard thermal noise temperature.
    SimFileIdPlus= Name of the file where the simulated results are stored for the thermal noise temperature +1 standard deviation.
    NBins= Amount of bins to group the values in.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import NuRadioReco.examples.cr_efficiency_analysis.helper_cr_eff as hcr
    from NuRadioReco.utilities import units
        
    #Import simulation results
    SimEventRMSMin=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventRMS_" + str(SimFileIdMin) + ".npy",allow_pickle=True)
    SimEventTimeMin=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventTime_" + str(SimFileIdMin) + ".npy",allow_pickle=True)
      
    SimEventRMSZero=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventRMS_" + str(SimFileIdZero) + ".npy",allow_pickle=True)
    SimEventTimeZero=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventTime_" + str(SimFileIdZero) + ".npy",allow_pickle=True)

    SimEventRMSPlus=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventRMS_" + str(SimFileIdPlus) + ".npy",allow_pickle=True)
    SimEventTimePlus=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventTime_" + str(SimFileIdPlus) + ".npy",allow_pickle=True)
    
    #Group simulation results
    MidBinsMin, GroupedVRMSMin = GroupVRMS(SimEventRMSMin,SimEventTimeMin,NBins)
    MidBinsZero, GroupedVRMSZero = GroupVRMS(SimEventRMSZero,SimEventTimeZero,NBins)
    MidBinsPlus, GroupedVRMSPlus = GroupVRMS(SimEventRMSPlus,SimEventTimePlus,NBins)
        
    #Calculate statistics
    SimVRMSAvgMin=np.array([np.mean(GroupedVRMSMin[i]) if len(GroupedVRMSMin[i])!=0 else 0 for i in range(len(GroupedVRMSMin))])
    SimVRMSStdMin=np.array([np.std(GroupedVRMSMin[i])/np.sqrt(len(GroupedVRMSMin[i])) if len(GroupedVRMSMin[i])!=0 else 0 for i in range(len(GroupedVRMSMin))])
    SimVRMSAvgZero=np.array([np.mean(GroupedVRMSZero[i]) if len(GroupedVRMSZero[i])!=0 else 0 for i in range(len(GroupedVRMSZero))])
    SimVRMSStdZero=np.array([np.std(GroupedVRMSZero[i])/np.sqrt(len(GroupedVRMSZero[i])) if len(GroupedVRMSZero[i])!=0 else 0 for i in range(len(GroupedVRMSZero))])
    SimVRMSAvgPlus=np.array([np.mean(GroupedVRMSPlus[i]) if len(GroupedVRMSPlus[i])!=0 else 0 for i in range(len(GroupedVRMSPlus))])
    SimVRMSStdPlus=np.array([np.std(GroupedVRMSPlus[i])/np.sqrt(len(GroupedVRMSPlus[i])) if len(GroupedVRMSPlus[i])!=0 else 0 for i in range(len(GroupedVRMSPlus))])

    #TN simulation error calculation
    GNUpperError,GNLowerError=SimVRMSAvgPlus-SimVRMSAvgZero,SimVRMSAvgZero-SimVRMSAvgMin
    GNSimError=np.array([GNLowerError,GNUpperError])
    
    if Plot:
        AvgError=np.array([np.mean([GNUpperError[i],GNLowerError[i]]) for i in range(len(GNLowerError))])
        AvgRelError=np.mean(AvgError/SimVRMSAvgZero)
        plt.figure(figsize=(15,5))
        plt.figtext(0.6, 0.2, "AvgError:" + str(np.round(np.mean(10**6*AvgError),2)) + "µV; AvgRelError:" + str(np.round(100*AvgRelError,2)) + "%", fontsize=16,bbox=dict(edgecolor='black', facecolor='none', alpha=0.2, pad=10.0))
        plt.errorbar(MidBinsPlus,1000*SimVRMSAvgPlus,yerr=1000*SimVRMSStdPlus,fmt=".",zorder=2,label="Simulation +1sigma")#,markersize=15)
        plt.errorbar(MidBinsZero,1000*SimVRMSAvgZero,yerr=1000*SimVRMSStdZero,fmt=".",zorder=2,label="Simulation ")#,markersize=15)
        plt.errorbar(MidBinsMin,1000*SimVRMSAvgMin,yerr=1000*SimVRMSStdMin,fmt=".",zorder=2,label="Simulation -1 sigma")#,markersize=15)
        plt.grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
        plt.title("Transit curve for station " + str(StNr) + ", antenna " + str(ChNr),fontsize=25)
        plt.xlabel("LST Time (hrs)",fontsize=20)#20)
        plt.ylabel("V_RMS (mV)",fontsize=20)#20)
        plt.xticks(np.arange(0, 24, 1.0),fontsize=25)#15)
        plt.yticks(fontsize=25)#15)
        plt.legend(loc="lower left",fontsize=15)
        plt.show()
            
        VRMSRatioUp=SimVRMSAvgPlus/SimVRMSAvgZero
        VRMSRatioUpStd=VRMSRatioUp*np.sqrt((SimVRMSStdPlus/SimVRMSAvgPlus)**2 + (SimVRMSStdZero/SimVRMSAvgZero)**2)
        VRMSRatioDown=SimVRMSAvgZero/SimVRMSAvgMin
        VRMSRatioDownStd=VRMSRatioDown*np.sqrt((SimVRMSStdZero/SimVRMSAvgZero)**2 + (SimVRMSStdMin/SimVRMSAvgMin)**2)            
        plt.figure(figsize=(15,5))
        plt.errorbar(MidBinsPlus,VRMSRatioUp,yerr=VRMSRatioUpStd,fmt=".",zorder=2,label="300K/275K")#,markersize=15)
        plt.errorbar(MidBinsZero,VRMSRatioDown,yerr=VRMSRatioDownStd,fmt=".",zorder=2,label="275K/250K ")#,markersize=15)
        plt.grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
        plt.title("Transit curve fraction for station " + str(StNr) + ", antenna " + str(ChNr),fontsize=25)
        plt.xlabel("LST Time (hrs)",fontsize=20)#20)
        plt.ylabel("Sim fraction",fontsize=20)#20)
        plt.xticks(np.arange(0, 24, 1.0),fontsize=25)#15)
        plt.yticks(fontsize=25)#15)
        plt.legend(loc="lower left",fontsize=15)
        plt.show()
    return GNSimError
    
def SimSysError(StNr,ChNr,DataSample,GNSimFileIdMin, GNSimFileIdZero,GNSimFileIdPlus,TNSimFileIdMin, TNSimFileIdZero, TNSimFileIdPlus,NBins=4*24,Plot=False):
    """ Computes the propagated systematic noise error on the thermal noise and the galactic noise and subsequently combines them.
    Parameters:
    StNr= Station number
    ChNr= Channel number
    DataSample= String that can be "C" for combined, "HC" for handcarry or "S" for Satellite 
    GNSimFileIdMin= Name of the file where the simulated results are stored for the radioskymap -1 standard deviation.
    GNSimFileIdZero= Name of the file where the simulated results are stored for the standard radioskymap.
    GNSimFileIdPlus= Name of the file where the simulated results are stored for the radioskymap +1 standard deviation.
    TNSimFileIdMin= Name of the file where the simulated results are stored for the thermal noise temperature -1 standard deviation.
    TNSimFileIdZero= Name of the file where the simulated results are stored for the standard thermal noise temperature.
    TNSimFileIdPlus= Name of the file where the simulated results are stored for the thermal noise temperature +1 standard deviation.
    NBins= Amount of bins to group the values in.
    """
    import NuRadioReco.examples.cr_efficiency_analysis.helper_cr_eff as hcr
    from NuRadioReco.utilities import units
       
    #Take the upwards facing or downwards facing LPDA corresponding to ChNr. (assumed thermal noise simulation to o,ly differ for upwards & downwards facing LPDAs)
    if ChNr in [13,16,19]:
            TNChNr=19
    elif ChNr in [14,17,20]:
            TNChNr=20
    else:
        print("Channel number not in list for this study")
        return
        
    #Import simulation results
    TNSimEventRMSMin=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(TNChNr) + "/" + DataSample + "SimEventRMS_" + str(TNSimFileIdMin) + ".npy",allow_pickle=True)
    TNSimEventTimeMin=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(TNChNr) + "/" + DataSample + "SimEventTime_" + str(TNSimFileIdMin) + ".npy",allow_pickle=True)
      
    TNSimEventRMSZero=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(TNChNr) + "/" + DataSample + "SimEventRMS_" + str(TNSimFileIdZero) + ".npy",allow_pickle=True)
    TNSimEventTimeZero=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(TNChNr) + "/" + DataSample + "SimEventTime_" + str(TNSimFileIdZero) + ".npy",allow_pickle=True)

    TNSimEventRMSPlus=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(TNChNr) + "/" + DataSample + "SimEventRMS_" + str(TNSimFileIdPlus) + ".npy",allow_pickle=True)
    TNSimEventTimePlus=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(TNChNr) + "/" + DataSample + "SimEventTime_" + str(TNSimFileIdPlus) + ".npy",allow_pickle=True)

    GNSimEventRMSMin=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventRMS_" + str(GNSimFileIdMin) + ".npy",allow_pickle=True)
    GNSimEventTimeMin=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventTime_" + str(GNSimFileIdMin) + ".npy",allow_pickle=True)
      
    GNSimEventRMSZero=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventRMS_" + str(GNSimFileIdZero) + ".npy",allow_pickle=True)
    GNSimEventTimeZero=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventTime_" + str(GNSimFileIdZero) + ".npy",allow_pickle=True)

    GNSimEventRMSPlus=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventRMS_" + str(GNSimFileIdPlus) + ".npy",allow_pickle=True)
    GNSimEventTimePlus=np.load("JobResults/Sim/St" + str(StNr) + "Ch" + str(ChNr) + "/" + DataSample + "SimEventTime_" + str(GNSimFileIdPlus) + ".npy",allow_pickle=True)
       
    #Group TN simulation results
    TNMidBinsMin, TNGroupedVRMSMin = GroupVRMS(TNSimEventRMSMin,TNSimEventTimeMin,NBins)
    TNMidBinsZero, TNGroupedVRMSZero = GroupVRMS(TNSimEventRMSZero,TNSimEventTimeZero,NBins)
    TNMidBinsPlus, TNGroupedVRMSPlus = GroupVRMS(TNSimEventRMSPlus,TNSimEventTimePlus,NBins)
        
    #Compute statistics
    TNSimVRMSAvgMin=np.array([np.mean(TNGroupedVRMSMin[i]) if len(TNGroupedVRMSMin[i])!=0 else 0 for i in range(len(TNGroupedVRMSMin))])
    TNSimVRMSStdMin=np.array([np.std(TNGroupedVRMSMin[i])/np.sqrt(len(TNGroupedVRMSMin[i])) if len(TNGroupedVRMSMin[i])!=0 else 0 for i in range(len(TNGroupedVRMSMin))])
    TNSimVRMSAvgZero=np.array([np.mean(TNGroupedVRMSZero[i]) if len(TNGroupedVRMSZero[i])!=0 else 0 for i in range(len(TNGroupedVRMSZero))])
    TNSimVRMSStdZero=np.array([np.std(TNGroupedVRMSZero[i])/np.sqrt(len(TNGroupedVRMSZero[i])) if len(TNGroupedVRMSZero[i])!=0 else 0 for i in range(len(TNGroupedVRMSZero))])
    TNSimVRMSAvgPlus=np.array([np.mean(TNGroupedVRMSPlus[i]) if len(TNGroupedVRMSPlus[i])!=0 else 0 for i in range(len(TNGroupedVRMSPlus))])
    TNSimVRMSStdPlus=np.array([np.std(TNGroupedVRMSPlus[i])/np.sqrt(len(TNGroupedVRMSPlus[i])) if len(TNGroupedVRMSPlus[i])!=0 else 0 for i in range(len(TNGroupedVRMSPlus))])

    #TN Error calculation simulation
    TNUpperError,TNLowerError=TNSimVRMSAvgPlus-TNSimVRMSAvgZero,TNSimVRMSAvgZero-TNSimVRMSAvgMin
    TNSimErrorAvg=np.mean([TNLowerError,TNUpperError],0)
        
    #Group GN simulation results
    GNMidBinsMin, GNGroupedVRMSMin = GroupVRMS(GNSimEventRMSMin,GNSimEventTimeMin,NBins)
    GNMidBinsZero, GNGroupedVRMSZero = GroupVRMS(GNSimEventRMSZero,GNSimEventTimeZero,NBins)
    GNMidBinsPlus, GNGroupedVRMSPlus = GroupVRMS(GNSimEventRMSPlus,GNSimEventTimePlus,NBins)
      
    #Compute statistics
    GNSimVRMSAvgMin=np.array([np.mean(GNGroupedVRMSMin[i]) if len(GNGroupedVRMSMin[i])!=0 else 0 for i in range(len(GNGroupedVRMSMin))])
    GNSimVRMSStdMin=np.array([np.std(GNGroupedVRMSMin[i])/np.sqrt(len(GNGroupedVRMSMin[i])) if len(GNGroupedVRMSMin[i])!=0 else 0 for i in range(len(GNGroupedVRMSMin))])
    GNSimVRMSAvgZero=np.array([np.mean(GNGroupedVRMSZero[i]) if len(GNGroupedVRMSZero[i])!=0 else 0 for i in range(len(GNGroupedVRMSZero))])
    GNSimVRMSStdZero=np.array([np.std(GNGroupedVRMSZero[i])/np.sqrt(len(GNGroupedVRMSZero[i])) if len(GNGroupedVRMSZero[i])!=0 else 0 for i in range(len(GNGroupedVRMSZero))])
    GNSimVRMSAvgPlus=np.array([np.mean(GNGroupedVRMSPlus[i]) if len(GNGroupedVRMSPlus[i])!=0 else 0 for i in range(len(GNGroupedVRMSPlus))])
    GNSimVRMSStdPlus=np.array([np.std(GNGroupedVRMSPlus[i])/np.sqrt(len(GNGroupedVRMSPlus[i])) if len(GNGroupedVRMSPlus[i])!=0 else 0 for i in range(len(GNGroupedVRMSPlus))])

    #GN simulation error calculation 
    GNUpperError,GNLowerError=GNSimVRMSAvgPlus-GNSimVRMSAvgZero,GNSimVRMSAvgZero-GNSimVRMSAvgMin
    GNSimErrorAvg=np.mean([GNLowerError,GNUpperError],0)
    
    #Print results
    PrintTN=np.round(10**6*np.mean(TNSimErrorAvg),2)
    PrintGN=np.round(10**6*np.mean(GNSimErrorAvg),2)
    PrintTot=np.round(10**6*np.sqrt(np.mean(TNSimErrorAvg)**2 + np.mean(GNSimErrorAvg)**2),2)
    print("Station",StNr,", Channel",ChNr,":",PrintTN,"µV TN +",PrintGN,"µV GN =>",PrintTot,"µV Tot")
    
    if Plot:
        AvgError=np.array([np.mean([GNUpperError[i],GNLowerError[i]]) for i in range(len(GNLowerError))])
        AvgRelError=np.mean(AvgError/SimVRMSAvgZero)
        plt.figure(figsize=(15,5))
        plt.figtext(0.6, 0.2, "AvgError:" + str(np.round(np.mean(10**6*AvgError),2)) + "µV; AvgRelError:" + str(np.round(100*AvgRelError,2)) + "%", fontsize=16,bbox=dict(edgecolor='black', facecolor='none', alpha=0.2, pad=10.0))
        plt.errorbar(MidBinsPlus,1000*SimVRMSAvgPlus,yerr=1000*SimVRMSStdPlus,fmt=".",zorder=2,label="Simulation +1sigma")#,markersize=15)
        plt.errorbar(MidBinsZero,1000*SimVRMSAvgZero,yerr=1000*SimVRMSStdZero,fmt=".",zorder=2,label="Simulation ")#,markersize=15)
        plt.errorbar(MidBinsMin,1000*SimVRMSAvgMin,yerr=1000*SimVRMSStdMin,fmt=".",zorder=2,label="Simulation -1 sigma")#,markersize=15)
        plt.grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
        plt.title("Transit curve for station " + str(StNr) + ", antenna " + str(ChNr),fontsize=25)
        plt.xlabel("LST Time (hrs)",fontsize=20)#20)
        plt.ylabel("V_RMS (mV)",fontsize=20)#20)
        plt.xticks(np.arange(0, 24, 1.0),fontsize=25)#15)
        plt.yticks(fontsize=25)#15)
        plt.legend(loc="lower left",fontsize=15)
        plt.show()
            
        VRMSRatioUp=SimVRMSAvgPlus/SimVRMSAvgZero
        VRMSRatioUpStd=VRMSRatioUp*np.sqrt((SimVRMSStdPlus/SimVRMSAvgPlus)**2 + (SimVRMSStdZero/SimVRMSAvgZero)**2)
        VRMSRatioDown=SimVRMSAvgZero/SimVRMSAvgMin
        VRMSRatioDownStd=VRMSRatioDown*np.sqrt((SimVRMSStdZero/SimVRMSAvgZero)**2 + (SimVRMSStdMin/SimVRMSAvgMin)**2)            
        plt.figure(figsize=(15,5))
        plt.errorbar(MidBinsPlus,VRMSRatioUp,yerr=VRMSRatioUpStd,fmt=".",zorder=2,label="300K/275K")#,markersize=15)
        plt.errorbar(MidBinsZero,VRMSRatioDown,yerr=VRMSRatioDownStd,fmt=".",zorder=2,label="275K/250K ")#,markersize=15)
        plt.grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
        plt.title("Transit curve fraction for station " + str(StNr) + ", antenna " + str(ChNr),fontsize=25)
        plt.xlabel("LST Time (hrs)",fontsize=20)#20)
        plt.ylabel("Sim fraction",fontsize=20)#20)
        plt.xticks(np.arange(0, 24, 1.0),fontsize=25)#15)
        plt.yticks(fontsize=25)#15)
        plt.legend(loc="lower left",fontsize=15)
        plt.show()
    return



# brown (RGB: 90, 40, 40) to white (RGB: 255, 255, 255)
BkgGradBlueRGB=(143,184,226)
BkgGradPinkRGB=(221,162,201)

PurpleLinesRGB=(85,81,162)
PurpleDarkRGB=(117,101,173)
PurpleLightRGB=(131,120,183)

PinkLightRGB=(217,128,181)
PinkDarkRGB=(215,108,170)
def MakeCM(rgb1,rgb2):
    """Make a new colormap"""
    r1,g1,b1=rgb1
    r2,g2,b2=rgb2
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(r1/256,r2/256, N)
    vals[:, 1] = np.linspace(g1/256,g2/256, N)
    vals[:, 2] = np.linspace(b1/256,b2/256, N)
    newcmp = mpl.colors.ListedColormap(vals)
    return newcmp

def MakeCM3(rgb1,rgb2,rgb3,N=256):
    """Make a new colormap"""
    r1,g1,b1=rgb1
    r2,g2,b2=rgb2
    r3,g3,b3=rgb3
    vals = np.ones((N, 4))
    HalfPoint=int(N/2)
    vals[:HalfPoint, 0] = np.linspace(r1/256,r2/256, HalfPoint)
    vals[:HalfPoint, 1] = np.linspace(g1/256,g2/256, HalfPoint)
    vals[:HalfPoint, 2] = np.linspace(b1/256,b2/256, HalfPoint)
    vals[HalfPoint:, 0] = np.linspace(r2/256,r3/256, HalfPoint)
    vals[HalfPoint:, 1] = np.linspace(g2/256,g3/256, HalfPoint)
    vals[HalfPoint:, 2] = np.linspace(b2/256,b3/256, HalfPoint)
    newcmp = mpl.colors.ListedColormap(vals)
    return newcmp

def MakeCM4(rgb1,rgb2,rgb3,rgb4,N=256):
    """Make a new colormap"""
    import matplotlib as mpl
    r1,g1,b1=rgb1
    r2,g2,b2=rgb2
    r3,g3,b3=rgb3
    r4,g4,b4=rgb4
    vals = np.ones((N, 4))
    FPoint=int(N/3)
    SPoint=int(2*N/3)
    vals[:FPoint, 0] = np.linspace(r1/256,r2/256, FPoint)
    vals[:FPoint, 1] = np.linspace(g1/256,g2/256, FPoint)
    vals[:FPoint, 2] = np.linspace(b1/256,b2/256, FPoint)
    vals[FPoint:SPoint, 0] = np.linspace(r2/256,r3/256, FPoint)
    vals[FPoint:SPoint, 1] = np.linspace(g2/256,g3/256, FPoint)
    vals[FPoint:SPoint, 2] = np.linspace(b2/256,b3/256, FPoint)
    vals[SPoint:, 0] = np.linspace(r3/256,r4/256, FPoint+1)
    vals[SPoint:, 1] = np.linspace(g3/256,g4/256, FPoint+1)
    vals[SPoint:, 2] = np.linspace(b3/256,b4/256, FPoint+1)
    newcmp = mpl.colors.ListedColormap(vals)
    return newcmp

#===========================================

def TransitCurveAnyComparison(FileId1,FileId2,NBins=4*24,TimeFormat="LST",ZeroAvg=False):
    """ Plots the curves for two files.
    Parameters:
    DataFileId= Name of the file where the data is stored.
    SimFileId= Name of the file where the simulated results are stored.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as scst
    
    #Import data   
    GN1=pd.read_pickle(FileId1)
    GN2=pd.read_pickle(FileId2)
       
    BinEdges=np.linspace(0,24,NBins+1,True)
    GN1['Bin'] = pd.cut(GN1[TimeFormat], BinEdges)
    GN2['Bin'] = pd.cut(GN2[TimeFormat], BinEdges)
    
    #Compute relevant statistics
    GN1BinStat=GN1.groupby(["Bin"]).agg({"VRMS":[np.median,(lambda x: np.std(x)/np.sqrt(len(x))) ]})
    GN1PlotFilter=GN1BinStat[('VRMS', 'median')].isnull()
    GN1VRMSMed=GN1BinStat[('VRMS', 'median')].drop(GN1BinStat[('VRMS', 'median')][GN1PlotFilter].index)
    GN1VRMSStd=GN1BinStat[('VRMS', '<lambda_0>')].drop(GN1BinStat[('VRMS', 'median')][GN1PlotFilter].index).fillna(0)
    
    GN2BinStat=GN2.groupby(["Bin"]).agg({"VRMS":[np.median,(lambda x: np.std(x)/np.sqrt(len(x))) ]})
    GN2PlotFilter=GN2BinStat[('VRMS', 'median')].isnull()
    GN2VRMSMed=GN2BinStat[('VRMS', 'median')].drop(GN2BinStat[('VRMS', 'median')][GN2PlotFilter].index)
    GN2VRMSStd=GN2BinStat[('VRMS', '<lambda_0>')].drop(GN2BinStat[('VRMS', 'median')][GN2PlotFilter].index).fillna(0)
    
    MidBins=np.array([(BinEdges[i] + BinEdges[i+1])/2 for i in range(0,len(BinEdges)-1)])[np.logical_not(GN1PlotFilter)]
    
    #Count the amount of entries in the transit curve
    # NEntries=GN1.shape[0]
    
    if ZeroAvg: 
        Offset=np.median(GN2VRMSMed)-np.median(GN1VRMSMed)
        GN2VRMSMed-=Offset
        
    
    plt.figure(figsize=(15,5))
    if ZeroAvg:
        plt.figtext(0.2, 0.8, r"$\Delta$ $V_{RMS}$:" + str(np.round(1e3*Offset,3)) + "mV", fontsize=18,bbox=dict(edgecolor='black', facecolor='none', alpha=0.2, pad=10.0))
    plt.errorbar(MidBins,1000*GN1VRMSMed,yerr=1000*GN1VRMSStd,fmt=".",zorder=2,label=FileId1.split("/")[-1][:-4])
    plt.errorbar(MidBins,1000*GN2VRMSMed,yerr=1000*GN2VRMSStd,fmt=".",zorder=2,label=FileId2.split("/")[-1][:-4])
    plt.grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
    plt.title("Transit curve for files " + FileId1.split("/")[-1][:-4] + " and " + FileId2.split("/")[-1][:-4],fontsize=25)
    plt.xlabel(TimeFormat + " Time (hrs)",fontsize=20)#20)
    plt.ylabel("V_RMS (mV)",fontsize=20)#20)
    plt.xticks(np.arange(0, 25, 1.0),fontsize=25)#15)
    plt.yticks(fontsize=25)#15)
    plt.xlim(0,24)
    plt.legend(loc="lower left",fontsize=15)
    plt.show()
    return

