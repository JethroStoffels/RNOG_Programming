from grafana_client import GrafanaApi
import matplotlib.pyplot as plt
import datetime
import numpy as np
import sys
from scipy.signal import argrelextrema

[StNr]= sys.argv[1:]

if StNr =='21':
    Querystr="SELECT rcv_time as \"time\", (msg_payload->'power_monitor'->>'BAT_I_mA')::float/1000., (msg_payload->'power_monitor'->>'BAT_V')::float AS \"V\" FROM inbox WHERE source_id = %s AND (msg_payload->'power_monitor'->>'BAT_I_mA')::float IS NOT NULL AND (msg_payload->'power_monitor'->>'BAT_V')::float IS NOT NULL"%(StNr)
else:
    Querystr="SELECT rcv_time as \"time\", (msg_payload->'currents'->>'batt')::float/1000., (msg_payload->'voltages'->>'batt')::float/1000. AS \"V\" FROM inbox WHERE source_id = %s AND (msg_payload->'currents'->>'batt')::float IS NOT NULL AND (msg_payload->'voltages'->>'batt')::float IS NOT NULL"%(StNr)
        

G = GrafanaApi.from_url("https://rno-g:Kangerlussuaq@rno-g.uchicago.edu/grafana/")
result = G.datasource.smartquery(G.datasource.get_datasource_by_name('PostgreSQL'),Querystr, {'time_from': 'now-1h'})

Filter=np.array([datetime.datetime.now() - datetime.timedelta(days=14)<datetime.datetime.utcfromtimestamp(t*1e-3)<datetime.datetime.now() for t in result['results']['test']['frames'][0]['data']['values'][0]])

UnixTime=np.array([t*1e-3 for t in result['results']['test']['frames'][0]['data']['values'][0]])[Filter]
time = np.array([datetime.datetime.utcfromtimestamp(t*1e-3) for t in result['results']['test']['frames'][0]['data']['values'][0]])[Filter] #unix time converted to seconds (it defaults to millisecond unix time for some reason, wtf)

var1 = np.array(result['results']['test']['frames'][0]['data']['values'][1])[Filter]

var2 = np.array(result['results']['test']['frames'][0]['data']['values'][2])[Filter]

VlocminIdx=argrelextrema(var2, np.less_equal,order=500)

# coef=np.polyfit(np.array(result['results']['test']['frames'][0]['data']['values'][0])[Filter],var2, 3)
coef=np.polyfit(UnixTime[VlocminIdx],var2[VlocminIdx], 2)
xfit=np.linspace(np.min(UnixTime),np.max(UnixTime),50)
#xfitmin=np.linspace(np.min(df["Time"]),5*np.max(df["Time"]),10**3)
p=np.poly1d(coef)
yfit=p(xfit)#-p(0)*np.ones(len(xfit))
coefext=coef

if StNr in ['21','22']:
    VTresh=11.5
else:
    VTresh=23
        

coefext[-1]-=VTresh
pextract=np.poly1d(coefext)
roots=np.roots(pextract)
yextfit=pextract(xfit)

# plt.plot(time,var1*var2)
# plt.xlabel('Time (UTC)')
# plt.ylabel('Battery power [W]')
# plt.xticks(rotation=45, ha='right')
# # plt.xlim(datetime.datetime.now() - datetime.timedelta(days=1),datetime.datetime.now())
# plt.savefig(f"Power_St"+str(StNr)+f"_{datetime.datetime.now().strftime('%y-%m-%d_%H%M')}.jpg", bbox_inches="tight")
# plt.close()

plt.plot(time,var2,color='#d76caa')
plt.plot([datetime.datetime.utcfromtimestamp(t) for t in xfit],yfit,color="#8fb8e2")
# plt.plot([datetime.datetime.utcfromtimestamp(t) for t in xfit],yextfit*1e-3)
plt.scatter(time[VlocminIdx],var2[VlocminIdx],color="#7565ad")
plt.title("Battery voltage Station %s"%(StNr),fontsize=20)
# plt.axhline(y=0, color='k', linestyle='--')
plt.axhline(y=VTresh, color='k', linestyle='--')
plt.xlabel('Time (UTC)')
plt.ylabel('Voltage (V)')
plt.xticks(rotation=45, ha='right')
plt.figtext(0.175, 0.15, "Time at "+str(VTresh)+"V: " +f"{datetime.datetime.utcfromtimestamp(np.max(np.abs(roots))).strftime('%b-%d %H:%M')}" , fontsize=10,bbox=dict(edgecolor='black', facecolor='none', alpha=0, pad=10.0))
# plt.xlim(datetime.datetime.now() - datetime.timedelta(days=1),datetime.datetime.now())
plt.ylim(np.min([VTresh,np.min(var2)])-1,np.max(var2)+1)
plt.savefig(f"Voltage_St"+str(StNr)+f"_{datetime.datetime.now().strftime('%y-%m-%d_%H%M')}.jpg", bbox_inches="tight")
plt.close()

# G = GrafanaApi.from_url("https://rno-g:Kangerlussuaq@rno-g.uchicago.edu/grafana/")
# result = G.datasource.smartquery(G.datasource.get_datasource_by_name('PostgreSQL'),"SELECT rcv_time as \"time\", (msg_payload->'currents'->>'batt')::float FROM inbox WHERE source_id = 12 AND (msg_payload->'currents'->>'batt')::float IS NOT NULL LIMIT 10", {'time_from': 'now-1h'})