def RotPlot(FilePaths,TimeFormat="LST",NBins=4*24,StdCut=(-1,-1),HardCut=-0.0035,ZeroAvg=False,SavePath="",Format="pdf"):
    """ Plots the curves for multiple files.
    Parameters:
    DataFileId= Name of the file where the data is stored.
    SimFileId= Name of the file where the simulated results are stored.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as scst
    import pandas as pd
    import matplotlib as mpl
    import datetime

    #Color palette: https://lospec.com/palette-list/curiosities
    Colors=['#46425e',
'#15788c',
'#00b9be',
'#ffeecc',
'#ffb0a3',
'#ff6973']
    ylim=(0,10)
    ymin,ymax=10,0
    LineStyles=['dotted','solid','dashed']
    f, axs = plt.subplots(1, 1, figsize=(15,5.56))
    plt.rcParams['legend.title_fontsize'] = 15
    axs.add_patch(mpl.patches.Rectangle((0,ylim[0]), 24, ylim[1]-ylim[0],color=Colors[3],alpha=0.5))
    if ZeroAvg:
        RefMed=np.median(pd.read_pickle(FilePaths[list(FilePaths.keys())[0]][0])["VRMS"])
        plt.figtext(0.141, 0.8, "Set to median: %.2f mV" % (1e3*RefMed), fontsize=18,bbox=dict(edgecolor='black', facecolor='none', alpha=0.2, pad=10.0))
    for ChNr, ChFilePath in FilePaths.items():
        ChIdx=int((ChNr-13)/3)
        # plt.plot([], [],  label="Channel number")
        # plt.plot([], [], color=Colors[ChIdx],linestyle=LineStyles[1], label="Ch"+str(ChNr))
        for FPIdx,FilePath in enumerate(ChFilePath):
            #Import data
            
            GNData=pd.read_pickle(FilePath)
            BinEdges=np.linspace(0,24,NBins+1,True)
            GNData['Bin'] = pd.cut(GNData[TimeFormat], BinEdges)
    
            if HardCut>0:
                GNData=HardCut(GNData,Threshold)
            
            if np.all(np.array(StdCut)>0):
                GNData=StdCut(GNData,StdCut)
            
            #Compute relevant statistics
            GNBinStat=GNData.groupby(["Bin"],observed=False).agg({"VRMS":['median',(lambda x: np.std(x)/np.sqrt(len(x))) ]})
            GNPlotFilter=GNBinStat[('VRMS', 'median')].isnull()
            GNVRMSMed=GNBinStat[('VRMS', 'median')].drop(GNBinStat[('VRMS', 'median')][GNPlotFilter].index)
            GNVRMSStd=GNBinStat[('VRMS', '<lambda_0>')].drop(GNBinStat[('VRMS', 'median')][GNPlotFilter].index).fillna(0)
            ymin,ymax=np.min([ymin,np.min(GNVRMSMed)]),np.max([ymax,np.max(GNVRMSMed)])
            MidBins=np.array([(BinEdges[i] + BinEdges[i+1])/2 for i in range(0,len(BinEdges)-1)])[np.logical_not(GNPlotFilter)]
        
        #Count the amount of entries in the transit curve
        # NEntries=GN1.shape[0]
        
            if ZeroAvg: 
                GNVRMSMed-=(np.median(GNData["VRMS"])-RefMed)
            plt.plot(MidBins,1000*GNVRMSMed,zorder=2,color=Colors[[1,2,5][ChIdx]],linestyle=LineStyles[FPIdx]) #.split('-')[-2]

    ylim=(1e3*0.995*ymin,1e3*1.005*ymax)
    plt.grid(color='grey', linestyle='-', linewidth=1,alpha=0.5)
    plt.title("Rotated antenna transit curves St23",fontsize=25)
    plt.xlabel(TimeFormat + " Time (hrs)",fontsize=20)#20)
    plt.ylabel("V_RMS (mV)",fontsize=20)#20)
    plt.xticks(np.arange(0, 25, 1.0),fontsize=25)#15)
    plt.yticks(fontsize=25)#15)
    plt.xlim(0,24)
    #plt.legend(loc="lower left",fontsize=15)
    ax=plt.gca()
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    plt.plot([], [],  label="Channel number")
    for ChNr, ChFilePath in FilePaths.items():
        ChIdx=int((ChNr-13)/3)
        plt.plot([], [], color=Colors[[1,2,5][ChIdx]],linestyle=LineStyles[1], label="Ch"+str(ChNr))
    # plt.plot([], [], color=Colors[0],linestyle=LineStyles[1], label="Ch13")
    # plt.plot([], [], color=Colors[1],linestyle=LineStyles[1], label="Ch16")
    # plt.plot([], [], color=Colors[2],linestyle=LineStyles[1], label="Ch19")
    plt.plot([], [],  label="Rotation")
    axs.plot([], [], color='k',linestyle=LineStyles[0], label="Phi - deg")
    axs.plot([], [], color='k',linestyle=LineStyles[1], label="Phi + 0 deg")
    axs.plot([], [], color='k',linestyle=LineStyles[2], label="Phi + 30 deg")
    
    
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
    #      fancybox=True, shadow=True, ncol=2,fontsize=17,title='Channel Number'+10*' '+'Rotation')
    lines = plt.gca().get_lines() 
    print(len(lines))
    legend1 = plt.legend([lines[i] for i in np.arange(len(lines))[-4-len(list(FilePaths.keys())):-4]], FilePaths.keys(), loc=(0.75,0.0125),title='Channel',fontsize=14,framealpha=0.6)
    legend2 = plt.legend([lines[i] for i in np.arange(len(lines))[-3:]], [r"$\phi$ - 30$^\circ$", r"$\phi$ + 0$^\circ$", r"$\phi$ + 30$^\circ$"], loc=(0.865,0.0125),title='Rotation',fontsize=14,framealpha=0.6)
    axs.add_artist(legend1)
    axs.add_artist(legend2)
    plt.ylim(*ylim)
    plt.savefig("./RotatedAETcurves-"+f"_{datetime.datetime.now().strftime('%y-%m-%d_%H%M')}"+"."+Format, format=Format, bbox_inches="tight")
    plt.show()
    return
    
    

#Options:
# # First results:
# FilePaths=[['GNSim-St23Ch13-544514-m30-Tot.pkl','GNSim-St23Ch13-544558-p0-Tot.pkl','GNSim-St23Ch13-544572-p30-Tot.pkl'],
#            ['GNSim-St23Ch16-544514-m30-Tot.pkl','GNSim-St23Ch16-544558-p0-Tot.pkl','GNSim-St23Ch16-544572-p30-Tot.pkl'],
#            ['GNSim-St23Ch19-544514-m30-Tot.pkl','GNSim-St23Ch19-544558-p0-Tot.pkl','GNSim-St23Ch19-544572-p30-Tot.pkl'],
#           ]
# FilePaths=[['/pnfs/iihe/rno-g/store/user/jstoffels/Jobs/GalacticNoise/Sim/V_RotAEPhiRot/St23/' + path for path in pathList] for pathList in FilePaths]
# ylim=(1.7,2)

# #Second Results:
# #A lot of these jobs are messed up due to a pnfs malfunction on the server side
# FilePaths=[['GNSim-St23Ch13-740554-Tot.pkl','GNSim-St23Ch13-740555-Tot.pkl','GNSim-St23Ch13-740557-Tot.pkl'],
#            ['GNSim-St23Ch16-740554-Tot.pkl','GNSim-St23Ch16-740555-Tot.pkl','GNSim-St23Ch16-740557-Tot.pkl'],
#            ['GNSim-St23Ch19-740554-Tot.pkl','GNSim-St23Ch19-740555-Tot.pkl','GNSim-St23Ch19-740557-Tot.pkl'],
#           ]
# FilePaths=[['/pnfs/iihe/rno-g/store/user/jstoffels/Jobs/GalacticNoise/Sim/V_RotAEPhiRot-2025-01-21/St23/' + path for path in pathList] for pathList in FilePaths]
# ylim=(3.125,3.35)

#Third results:
FilePaths={13:['GNSim-St23Ch13-870427-Tot.pkl','GNSim-St23Ch13-852152-Tot.pkl','GNSim-St23Ch13-870428-Tot.pkl'],
           # 16:['GNSim-St23Ch16-870427-Tot.pkl','GNSim-St23Ch16-852152-Tot.pkl','GNSim-St23Ch16-870428-Tot.pkl'],
           # 19:['GNSim-St23Ch19-870427-Tot.pkl','GNSim-St23Ch19-852152-Tot.pkl','GNSim-St23Ch19-870428-Tot.pkl'],
          }

# FilePaths={16:['GNSim-St23Ch16-870427-Tot.pkl','GNSim-St23Ch16-852152-Tot.pkl','GNSim-St23Ch16-870428-Tot.pkl'],
#           }

prefix='/pnfs/iihe/rno-g/store/user/jstoffels/Jobs/GalacticNoise/Sim/V_RotFit_2025-02-13/St23/'
# ylim=(1.7,2)
ylim=(2.8,3.8)
# ylim=(0,10)
for ChNr, ChFilePaths in FilePaths.items():
    FilePaths[ChNr]=[prefix + filepath for filepath in FilePaths[ChNr]]
RotPlot(FilePaths,TimeFormat="LST",NBins=4*24,StdCut=(-1,-1),HardCut=-0.0035,ZeroAvg=False,Format="png")