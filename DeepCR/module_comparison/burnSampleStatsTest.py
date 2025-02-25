from NuRadioReco.modules.RNO_G.stationDeepCRVariables import stationDeepCRVariables
from NuRadioReco.modules.channelSignalReconstructor import channelSignalReconstructor
from NuRadioReco.framework.parameters import channelParameters as chp, stationParametersRNOG as stpRNOG, stationParameters as stp
import NuRadioReco
import matplotlib.pyplot as plt
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
import pandas as pd
import numpy as np
from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
import datetime
import sys
import time
import json

def calc_vars(list_of_root_files, burn_sample, station_id=11, channel_ids=[0,1,2,3], debug=False):
    StartTime=time.time()
    
    det = detector.Detector(json_filename = "/user/jstoffels/software/DeepCR/NuRadioMC/NuRadioReco/detector/RNO_G/RNO_season_2021.json")
    det.update(datetime.datetime(2022, 10, 1))

    # """ read in data """
    
    stationDeepCRVariables = NuRadioReco.modules.RNO_G.stationDeepCRVariables.stationDeepCRVariables()
    stationDeepCRVariables.begin(channel_ids = channel_ids)
    
    channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
    channelSignalReconstructor.begin()
    
    readRNOGData = NuRadioReco.modules.io.RNO_G.readRNOGDataMattak.readRNOGData(load_run_table=False)
    readRNOGData.begin(list_of_root_files,apply_baseline_correction='none')
    
    avg_snrs=np.array([])
    impulsivities={ch_id:np.array([]) for ch_id in channel_ids} 
    max_a_norm=np.array([])
    coh_snrs=np.array([])
    
    for run_nr, event_ids in burn_sample.items():
        if not (run_nr in [root_file[-4:] for root_file in list_of_root_files]):
            continue
        for event_id in event_ids[:10]: #Take first 10 events of the run for debugging purposes
            event=readRNOGData.get_event_by_index(int(event_id),run_nr=int(run_nr))
            if event==None:
                print('None event')
                continue
            # print('Run',event.get_run_number(), ", event", event.get_id())
            station=event.get_station(station_id)
            for ch in station.iter_channels():
                if not ch.get_id() in channel_ids:
                    station.remove_channel(ch.get_id())
            channelSignalReconstructor.run(event,station,det)
            stationDeepCRVariables.run(event,station,det)
            
            avg_snrs = np.append(avg_snrs, np.mean([station.get_channel(ch_id)[chp.SNR]['peak_2_peak_amplitude_split_noise_rms'] for ch_id in channel_ids]))
            coh_snrs = np.append(coh_snrs, station[stpRNOG.coherent_snr])
            max_a_norm = np.append(max_a_norm, station[stp.channels_max_amplitude_norm])
            for ch_id in channel_ids:
                impulsivities[ch_id] = np.append(impulsivities[ch_id], station.get_channel(ch_id)[chp.impulsivity])
            del event
    
    CalcTime=time.time()-StartTime
    print("Computing time: " + str(int(np.floor(CalcTime/3600))) + "hr" + str(int(60*((CalcTime/3600)%1))) + "min for",len(avg_snrs), 'events (',np.round(CalcTime/len(avg_snrs),2),'s/evt )')
    
    if debug:            
        print('avg_snrs')
        print(avg_snrs)
        print('coh_snrs')
        print(coh_snrs)
        print('impulsivities')
        for ch_id in channel_ids:
            print(5*' ',ch_id,':',impulsivities[ch_id])
        print('max_a_norm')
        print(max_a_norm)

        Format='png'
        
        plt.figure()
        plt.hist(avg_snrs)
        plt.xlabel('avg_snr')
        plt.savefig('Figures/NuRadioMC_' + "avg_snr"+f"_{datetime.datetime.now().strftime('%y-%m-%d_%H%M')}"+"."+Format, format=Format, bbox_inches="tight")
        plt.close()
        
        plt.figure()
        plt.hist(coh_snrs)
        plt.xlabel('coh_snr')
        plt.savefig('Figures/NuRadioMC_' + "coh_snr"+f"_{datetime.datetime.now().strftime('%y-%m-%d_%H%M')}"+"."+Format, format=Format, bbox_inches="tight")
        plt.close()
        
        plt.figure()
        plt.hist(max_a_norm)
        plt.xlabel('max_a_norm')
        plt.savefig('Figures/NuRadioMC_' + "max_a_norm"+f"_{datetime.datetime.now().strftime('%y-%m-%d_%H%M')}"+"."+Format, format=Format, bbox_inches="tight")
        plt.close()
        
        plt.figure()
        for ch_id in channel_ids:
            plt.hist(impulsivities[ch_id],label='Ch'+str(ch_id))
        plt.xlabel('impulsivity')
        plt.savefig('Figures/NuRadioMC_' + "impulsivity"+f"_{datetime.datetime.now().strftime('%y-%m-%d_%H%M')}"+"."+Format, format=Format, bbox_inches="tight")
        plt.close()
    return avg_snrs,impulsivities,max_a_norm,coh_snrs

if False:
    # Open and read the JSON file
    with open('deep_cr_search/data/station11_2022_burn_sample_dict.json', 'r') as file:
        burn_sample = json.load(file) #150276 total events, avg 184 evts/run, abt 1.8-4secs per evt
    
    list_of_root_files = ['/pnfs/iihe/rno-g/data/handcarry/station11/run' + str(run_nr) for run_nr in burn_sample.keys()]
    list_of_root_files=np.random.choice(list_of_root_files, size=2)

    calc_vars(list_of_root_files, burn_sample, station_id=11, channel_ids=[0,1,2,3], debug=True)