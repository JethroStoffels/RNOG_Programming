from NuRadioReco.modules.RNO_G.stationLDAVariables import stationLDAVariables

import NuRadioReco
import matplotlib.pyplot as plt
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
import pandas as pd
import numpy as np
from NuRadioReco.utilities import units
from NuRadioReco.modules import channelBandPassFilter
from NuRadioReco.detector import detector
import datetime
from NuRadioReco.modules import sphericalWaveFitter
from NuRadioReco.modules import channelAddCableDelay

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
sphericalWaveFitter = NuRadioReco.modules.sphericalWaveFitter.sphericalWaveFitter()
channelAddCableDelay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()

#use_channels = [0,1,2,3]
#sphericalWaveFitter.begin(channel_ids = use_channels)

det = detector.Detector(json_filename = "/user/jstoffels/software/DeepCR/NuRadioMC/NuRadioReco/detector/RNO_G/RNO_season_2021.json")
det.update(datetime.datetime(2022, 10, 1))

station_id = 21
# pulser_id = 3  #Helper string C
# rel_pulser_position = det.get_relative_position(station_id, pulser_id, mode = 'device')

plots = True
""" read in data """
list_of_root_files = ['/pnfs/iihe/rno-g/data/handcarry/station21/run588']

stationLDAVariables = NuRadioReco.modules.RNO_G.stationLDAVariables.stationLDAVariables()
stationLDAVariables.begin()

readRNOGData = NuRadioReco.modules.io.RNO_G.readRNOGDataMattak.readRNOGData()
readRNOGData.begin(list_of_root_files)

for i_event, event in enumerate(readRNOGData.run()):
    stationLDAVariables.run(event,event.get_station(station_id),det)
    break
