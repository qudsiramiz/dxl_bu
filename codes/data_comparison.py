import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# Set the font style to Times New Roman
font = {'family': 'serif', 'weight': 'normal', 'size': 10}
plt.rc('font', **font)
plt.rc('text', usetex=True)

df_dsco = pd.read_csv("../data/DSCOVR_H1_FC_25808.csv")
df_ace = pd.read_csv("../data/AC_H0_SWE_21105.csv")

df_dsco.index = pd.to_datetime(df_dsco["EPOCH_yyyy-mm-ddThh:mm:ss.sssZ"])
df_ace.index = pd.to_datetime(df_ace["EPOCH_yyyy-mm-ddThh:mm:ss.sssZ"])
df_dsco.drop(["EPOCH_yyyy-mm-ddThh:mm:ss.sssZ"], axis=1, inplace=True)
df_ace.drop(["EPOCH_yyyy-mm-ddThh:mm:ss.sssZ"], axis=1, inplace=True)

df_dsco.rename(columns={"ION_N_#/cc_(w/o_error_bars)": "np_dsco"}, inplace=True)
df_ace.rename(columns={"H_DENSITY_#/cc": "np_ace"}, inplace=True)

# Replace the fill values with NaN
df_dsco.replace(to_replace=[-999.0, -999.00, -999.000, -1.e+31], value=np.nan, inplace=True)
df_ace.replace(to_replace=[-999.0, -999.00, -999.000, -1.e+31], value=np.nan, inplace=True)

df_dsco_eph = pd.read_csv("../data/DSCOVR_ORBIT_PRE_121439.csv")
df_ace_eph = pd.read_csv("../data/AC_OR_SSC_120581.csv")

df_dsco_eph.index = pd.to_datetime(df_dsco_eph["EPOCH_yyyy-mm-ddThh:mm:ss.sssZ"])
df_ace_eph.index = pd.to_datetime(df_ace_eph["EPOCH__yyyy-mm-ddThh:mm:ss.sssZ"])
df_dsco_eph.drop(["EPOCH_yyyy-mm-ddThh:mm:ss.sssZ"], axis=1, inplace=True)
df_ace_eph.drop([df_ace_eph.keys()[0]], axis=1, inplace=True)

# Radius of the Earth in km
r_e = 6371.0

df_dsco_eph["r"] = np.sqrt(df_dsco_eph.X_GSE_km**2 + df_dsco_eph.Y_GSE_km**2 + df_dsco_eph.Z_GSE_km**2)/r_e
df_ace_eph["r"] = np.sqrt(df_ace_eph[df_ace_eph.keys()[0]]**2 +
                          df_ace_eph[df_ace_eph.keys()[1]]**2 + df_ace_eph[df_ace_eph.keys()[2]]**2)/r_e

# Define the plot parameters
# cmap = plt.cm.viridis
# pad = 0.02
# clabelpad = 10
labelsize = 22
ticklabelsize = 20
# cticklabelsize = 15
# clabelsize = 15
ticklength = 6
tickwidth = 1.0
# mticklength = 4
# cticklength = 5
# mcticklength = 4
# labelrotation = 0
xlabelsize = 20
ylabelsize = 20
alpha = 0.3
bar_color = 'k'

ms = 2
lw = 2
ncols = 2


# Change to datetime object
date1 = "2019-01-04 10:00:00"
date2 = "2019-01-04 23:00:00"
date_lim1 = datetime.datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
date_lim2 = datetime.datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")

fig = plt.figure(num=None, figsize=(10, 13), dpi=200, facecolor='w', edgecolor='gray')
fig.subplots_adjust(left=0.01, right=0.95, top=0.95, bottom=0.01, wspace=0.02, hspace=0.)
fig.suptitle(f'ACE and DSCOVR proton density comparison', fontsize=24)

gs = fig.add_gridspec(6, 1)
axs1 = fig.add_subplot(gs[0, 0])

axs1.plot(df_dsco.index, df_dsco["np_dsco"], color='r', linestyle='-', linewidth=lw, label='DSCOVR')
axs1.plot(df_ace.index, df_ace["np_ace"], color='b', linestyle='--', linewidth=lw, label='ACE')
axs1.set_xlim(date_lim1, date_lim2)
axs1.set_ylim(5, 65)
axs1.set_yscale('linear')

axs2 = fig.add_subplot(gs[2, 0], sharex=axs1)
axs2.plot(df_dsco_eph.index, df_dsco_eph["r"], color='r', linestyle='-', linewidth=lw, label='DSCOVR')
axs2.plot(df_ace_eph.index, df_ace_eph["r"], color='b', linestyle='--', linewidth=lw, label='ACE')
axs2.set_ylabel(r'Spacecraft position [$R_\oplus$]', fontsize=labelsize)
axs2.set_xlabel('Time [MM:DD HH]', fontsize=labelsize)

fig_name = f"/media/cephadrius/endless/bu_research/dxl/figures/ace_dscovr_data_comparison.png"
plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.05, format='png', dpi=300)
plt.close("all")
#plt.show()
