import burnSampleStatsTest as nu_module
import bryan_scripts_vars as bryan
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime

# Open and read the JSON file
with open('../deep_cr_search/data/station11_2022_burn_sample_dict.json', 'r') as file:
    burn_sample = json.load(file) #150276 total events, avg 184 evts/run, abt 1.8-4secs per evt
    
list_of_root_files = ['/pnfs/iihe/rno-g/data/handcarry/station11/run' + str(run_nr) for run_nr in burn_sample.keys()]
list_of_root_files=np.random.choice(list_of_root_files, size=26)

avg_snrs_nu,impulsivities_nu,max_a_norm_nu,coh_snrs_nu=nu_module.calc_vars(list_of_root_files, burn_sample, station_id=11, channel_ids=[0,1,2,3], debug=False)
print(30*'=')
avg_snrs_bryan,impulsivities_bryan,max_a_norm_bryan,coh_snrs_bryan=bryan.calc_vars(list_of_root_files, burn_sample, station_id=11, channel_ids=[0,1,2,3], debug=False)

def hist_comparison(nu_vals,bryan_vals,title, Format='png'):
    f, axs = plt.subplots(1, 1, figsize=(8,5))
    # style_nu = {'facecolor': 'none', 'edgecolor': 'C0', 'linewidth': 3,linestyle:'solid'}
    # style_bryan = {'facecolor': 'none', 'edgecolor': 'g', 'linewidth': 3,linestyle:'dashed'}
    counts, bins = np.histogram(np.hstack((nu_vals,bryan_vals)), bins='fd')
    xlim=(0.9*np.min(bins),1.1*np.max(bins))
    ylim=(0,0.55*np.max(counts))
    colors=['#ced9ec','#66b0c7','#34419d'] #https://lospec.com/palette-list/retro-blue
    axs.add_patch(mpl.patches.Rectangle((xlim[0],ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0],color=colors[0],alpha=1))
    plt.hist(nu_vals,bins,label='NuRadio module', histtype='step', color=colors[2],linewidth=3)
    plt.hist(bryan_vals,bins, label='Original code', histtype='step', linestyle='--',color=colors[1],linewidth=3)
    plt.title(title + ' for ' + str(len(nu_vals)) + ' events')
    leg=plt.legend(fontsize=15)
    for legobj in leg.legend_handles:
        legobj.set_linewidth(3)
    # plt.xlabel('avg_snr')
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.savefig('../Figures/Comparison_' + title+f"_{datetime.datetime.now().strftime('%y-%m-%d_%H%M')}"+"."+Format, format=Format, bbox_inches="tight")
    plt.close()

hist_comparison(avg_snrs_nu,avg_snrs_bryan,'avg_snr')
hist_comparison(max_a_norm_nu,max_a_norm_bryan,'max_a_norm')
hist_comparison(coh_snrs_nu,coh_snrs_bryan,'coh_snr')
for ch_id in [0,1,2,3]:
    hist_comparison(impulsivities_nu[ch_id],impulsivities_bryan[ch_id],'impulsivity ch' + str(ch_id))