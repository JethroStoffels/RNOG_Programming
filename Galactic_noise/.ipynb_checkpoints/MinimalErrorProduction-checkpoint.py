from NuRadioReco.utilities import units
from NuRadioReco.framework import event, station, channel
from NuRadioReco.detector import detector
from datetime import datetime
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import numpy as np
import NuRadioReco.examples.cr_efficiency_analysis.helper_cr_eff as hcr
import NuRadioReco.modules.channelGenericNoiseAdder as ChannelGenericNoiseAdder

StNr,ChNrs,RunNr,EventNr,UnixTime=23,[13,16],1,80,1658120517.2110965

Date = datetime.utcfromtimestamp(UnixTime)

kwargs = {"source": "rnog_mongo",'select_stations':StNr}
GNDetector = detector.Detector(**kwargs)

GNDetector.update(Date)
    
GNStation=station.Station(StNr)
GNStation.set_station_time(Date)
GNEvent=event.Event(RunNr,EventNr)

for ChNr in ChNrs:
    GNChannel=channel.Channel(ChNr)
    GNChannel.set_trace(trace=np.zeros(2048), sampling_rate=GNDetector.get_sampling_frequency(StNr, ChNr))
    GNStation.add_channel(GNChannel) 
    


Noise_min_freq,Noise_max_freq,Noise_df=10 * units.MHz,1100 * units.MHz,100*units.MHz
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()
channelGenericNoiseAdder.run(GNEvent, GNStation, GNDetector, amplitude=hcr.calculate_thermal_noise_Vrms(T_noise=266, T_noise_max_freq=Noise_max_freq, T_noise_min_freq=Noise_min_freq),min_freq=Noise_min_freq, max_freq=Noise_max_freq,type='rayleigh')

hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
hardwareResponseIncorporator.run(GNEvent, GNStation, GNDetector, sim_to_data=True)

for channel in  GNStation.iter_channels():
    print(channel.get_trace())