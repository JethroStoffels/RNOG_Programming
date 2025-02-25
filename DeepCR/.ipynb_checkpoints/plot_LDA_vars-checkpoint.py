import matplotlib.pyplot as plt
from NuRadioReco.framework.parameters import channelParameters as chp, stationParametersRNOG as stpRNOG, stationParameters as stp
from NuRadioReco.utilities import units
import NuRadioReco.modules.io.eventReader
import datetime
import numpy as np

Format='pdf'
Vars=['max_a','coh_snr','avg_snr','avg_imp']
LDA_vars={var:np.array([]) for var in Vars}
station_id = 11
channel_ids=[0,1,2,3]

event_reader = NuRadioReco.modules.io.eventReader.eventReader()
event_reader.begin(['TestVarFile.nur'])
# iterate over events
for event in event_reader.run():
    station = event.get_station(11)
    station.add_parameter_type(stpRNOG)
    LDA_vars['max_a'] = np.append(LDA_vars['max_a'], station[stp.channels_max_amplitude_norm])
    LDA_vars['coh_snr'] = np.append(LDA_vars['coh_snr'], station[stpRNOG.coherent_snr])
    p2p_snrs=[station.get_channel(ch_id)[chp.SNR]['peak_2_peak_amplitude_split_noise_rms'] for ch_id in channel_ids]
    LDA_vars['avg_snr'] = np.append(LDA_vars['avg_snr'], np.mean(p2p_snrs))
    avg_imp=0
    for ch_id in channel_ids:
        avg_imp += station.get_channel(ch_id)[chp.impulsivity]
    LDA_vars['avg_imp'] = np.append(LDA_vars['avg_imp'], np.mean(avg_imp/len(channel_ids)))
    
NVars=len(Vars)-1
fig, axs = plt.subplots(nrows=NVars, ncols=NVars, figsize=(20,20))
for ax_idx, ax in enumerate(axs.reshape(-1)):
    ax.tick_params(axis='both', which='major', labelsize=20)
    i,j=int(ax_idx%NVars),int(np.floor(ax_idx/NVars))+1
    if i>=j:
        ax.set_visible(False)
        continue
    ax.hist2d(LDA_vars[Vars[i]], LDA_vars[Vars[j]], bins=(20, 20), cmap='viridis',edgecolor='face')
    ax.set_xlabel(Vars[i],fontsize=20)
    ax.set_ylabel(Vars[j],fontsize=20)
    ax.grid(False)
fig.tight_layout()
plt.savefig('./Figures/2dhist_test_'+f"_{datetime.datetime.now().strftime('%y-%m-%d_%H%M')}"+"."+Format, format=Format, bbox_inches="tight")
plt.show()