from NuRadioReco.modules.RNO_G.stationDeepCRVariables import stationDeepCRVariables
from NuRadioReco.modules.channelSignalReconstructor import channelSignalReconstructor
from NuRadioReco.framework.parameters import channelParameters as chp, stationParametersRNOG as stpRNOG, stationParameters as stp
import NuRadioReco
import matplotlib.pyplot as plt
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
import pandas as pd
import numpy as np
from NuRadioReco.utilities import units
# from NuRadioReco.detector import detector
import datetime
import sys
import time
import json

def calc_vars(list_of_root_files, burn_sample, station_id=11, channel_ids=[0,1,2,3], debug=False):
    StartTime=time.time()
    
    # det = detector.Detector(json_filename = "/user/jstoffels/software/DeepCR/NuRadioMC/NuRadioReco/detector/RNO_G/RNO_season_2021.json")
    # det.update(datetime.datetime(2022, 10, 1))

    # """ read in data """
    
    readRNOGData = NuRadioReco.modules.io.RNO_G.readRNOGDataMattak.readRNOGData(load_run_table=False)
    readRNOGData.begin(list_of_root_files,apply_baseline_correction='none')
    
    N=0
    burn_sample_ev_nr=burn_sample.copy()
    for run_nr, event_ids in burn_sample.items():
        if not (run_nr in [root_file[-4:] for root_file in list_of_root_files]):
            continue
        # for event_id in event_ids[:100]: #Take first 10 events of the run for debugging purposes
        for i, event_id in enumerate(event_ids[:100]):
            N+=1
            event=readRNOGData.get_event_by_index(int(event_id),run_nr=int(run_nr))
            if event==None:
                print('None event')
                continue
            print('Run',event.get_run_number(), ", event", event.get_id(), ', index', event_id)
            burn_sample_ev_nr[run_nr][i]=event.get_run_number()
            # print(burn_sample_ev_nr[run_nr][i])
            del event
    
    CalcTime=time.time()-StartTime
    print("Computing time: " + str(int(np.floor(CalcTime/3600))) + "hr" + str(int(60*((CalcTime/3600)%1))) + "min for",N, 'events (',np.round(CalcTime/N,2),'s/evt )')
    return 

if True:
    # Open and read the JSON file
    with open('deep_cr_search/data/station11_2022_burn_sample_dict.json', 'r') as file:
        burn_sample = json.load(file) #150276 total events, avg 184 evts/run, abt 1.8-4secs per evt
    
    list_of_root_files = ['/pnfs/iihe/rno-g/data/handcarry/station11/run' + str(run_nr) for run_nr in burn_sample.keys()]
    # list_of_root_files=np.random.choice(list_of_root_files, size=2)
    list_of_root_files=[list_of_root_files[0]]

    calc_vars(list_of_root_files, burn_sample, station_id=11, channel_ids=[0,1,2,3], debug=True)