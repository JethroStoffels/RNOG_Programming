from deep_cr_search.tools.make_variables import Event
import json
import numpy as np
import datetime
import sys
import time
from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
import matplotlib.pyplot as plt

def calc_vars(list_of_root_files, burn_sample, station_id=11, channel_ids=[0,1,2,3], debug=False):
    StartTime=time.time()
    
    det = detector.Detector(json_filename = "/user/jstoffels/software/DeepCR/NuRadioMC/NuRadioReco/detector/RNO_G/RNO_season_2021.json")
    det.update(datetime.datetime(2022, 10, 1))
    
    avg_snrs=np.array([])
    impulsivities={ch_id:np.array([]) for ch_id in channel_ids} 
    max_a_norm=np.array([])
    coh_snrs=np.array([])
    
    for run_nr, event_ids in burn_sample.items():
        if not (run_nr in [root_file[-4:] for root_file in list_of_root_files]):
            # print(run_nr, end='\r')
            continue
        for event_id in event_ids[:10]: #Take first 10 events of the run for debugging purposes
            event=Event(station=station_id,run=run_nr,data_path = "/pnfs/iihe/rno-g/data/handcarry/")
            event.LoadWaveform(event_id)
            if event==None:
                print('None event')
                continue
            avg_snrs = np.append(avg_snrs, event.avg_ch_SNR())
            event.CoherentSum(ref_channel=0)
            coh_snrs = np.append(coh_snrs, event.coherentSNR(event.sum_chan))
            max_a_norm = np.append(max_a_norm, event.MaxA())
            for ch_id in channel_ids:
                impulsivities[ch_id] = np.append(impulsivities[ch_id], event.impulsive_value(event.wf[ch_id],event.t[ch_id]))
    
    CalcTime=time.time()-StartTime
    print("Computing time: " + str(int(np.floor(CalcTime/3600))) + "hr" + str(int(60*((CalcTime/3600)%1))) + "min for",len(max_a_norm), 'events (',np.round(CalcTime/len(max_a_norm),2),'s/evt )')
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
        plt.savefig('Figures/Bryan_' + "avg_snr"+f"_{datetime.datetime.now().strftime('%y-%m-%d_%H%M')}"+"."+Format, format=Format, bbox_inches="tight")
        plt.close()
        
        plt.figure()
        plt.hist(coh_snrs)
        plt.xlabel('coh_snr')
        plt.savefig('Figures/Bryan_' + "coh_snr"+f"_{datetime.datetime.now().strftime('%y-%m-%d_%H%M')}"+"."+Format, format=Format, bbox_inches="tight")
        plt.close()
        
        plt.figure()
        plt.hist(max_a_norm)
        plt.xlabel('max_a_norm')
        plt.savefig('Figures/Bryan_' + "max_a_norm"+f"_{datetime.datetime.now().strftime('%y-%m-%d_%H%M')}"+"."+Format, format=Format, bbox_inches="tight")
        plt.close()
        
        plt.figure()
        for ch_id in channel_ids:
            plt.hist(impulsivities[ch_id],label='Ch'+str(ch_id))
        plt.xlabel('impulsivity')
        plt.savefig('Figures/Bryan_' + "impulsivity"+f"_{datetime.datetime.now().strftime('%y-%m-%d_%H%M')}"+"."+Format, format=Format, bbox_inches="tight")
        plt.close()

    return avg_snrs,impulsivities,max_a_norm,coh_snrs

if False:
    # Open and read the JSON file
    with open('deep_cr_search/data/station11_2022_burn_sample_dict.json', 'r') as file:
        burn_sample = json.load(file) #150276 total events, avg 184 evts/run, abt 1.8-4secs per evt
    
    list_of_root_files = ['/pnfs/iihe/rno-g/data/handcarry/station11/run' + str(run_nr) for run_nr in burn_sample.keys()]
    list_of_root_files=np.random.choice(list_of_root_files, size=1)

    calc_vars(list_of_root_files, burn_sample, station_id=11, channel_ids=[0,1,2,3], debug=True)