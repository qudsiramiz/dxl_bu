# Code to read a net cdf file
import datetime
import glob
import os
import sched
import time

import geopack.geopack as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from scipy.io import netcdf

s = sched.scheduler(time.time, time.sleep)


def plot_figures_dsco_54days(sc):
#for xx in range(1):
    """
    Download and plot the data from DSCOVR for the last 54 days starting from the present time.
    """

    print(f"Code execution for DSCOVR 54 days data started at at (UTC):" +
          f"{datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}")

    # Set the font style to Times New Roman
    font = {'family': 'serif', 'weight': 'normal', 'size': 10}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)

    # Set up the time to run the job
    s.enter(3600, 1, plot_figures_dsco_54days, (sc,))

    # start = time.time()
    # Find the present time in unix time
    t_unix = datetime.datetime(1970, 1, 1)
    time_now = (datetime.datetime.utcnow() - t_unix).total_seconds()

    # Set the start and end time (in unix time, 54 days interval)
    time_start = time_now - (60 * 60 * 24 * 54)
    time_end = time_now

    # Change the directory to the data directory
    os.chdir("/media/cephadrius/endless/bu_research/dxl/data/dscovr_data/")

    # Delete all the files in the directory
    os.system("rm -rf *")

    # Find the date corresponding to the time
    #date_start = datetime.datetime.utcfromtimestamp(time_start).strftime('%Y-%m-%d')
    #date_end = datetime.datetime.utcfromtimestamp(time_end).strftime('%Y-%m-%d')

    time_data = time_start
    while time_data < time_end:
        year = datetime.datetime.utcfromtimestamp(time_data).strftime('%Y')
        month = datetime.datetime.utcfromtimestamp(time_data).strftime('%m')
        day = datetime.datetime.utcfromtimestamp(time_data).strftime('%d')
        print(f"Downloading data for {year}-{str(month).zfill(2)}-{str(day).zfill(2)}")
        url = f"https://www.ngdc.noaa.gov/dscovr/data/{year}/{str(month).zfill(2)}/"
        mag_file_name = f'oe_m1m_dscovr_s{year}{str(month).zfill(2)}{str(day).zfill(2)}*.gz'
        os.system(f"""wget -q -r -np -nc -nH -nd -A {mag_file_name} {url}""")
        plas_file_name = f'oe_f1m_dscovr_s{year}{str(month).zfill(2)}{str(day).zfill(2)}*.gz'
        os.system(f"""wget -q -r -np -nc -nH -nd -A {plas_file_name} {url}""")
        time_data += 60 * 60 * 24

    print("Downloading complete\n")

    # Unzip the files and delete the zipped files
    print("Unzipping the files\n")
    os.system("gunzip oe_*.gz")

    os.chdir("/home/cephadrius/Desktop/git/dxl_bu/codes")

    plas_file_list = np.sort(
        glob.glob("/media/cephadrius/endless/bu_research/dxl/data/dscovr_data/oe_f1m_*.nc"))
    mag_file_list = np.sort(
        glob.glob("/media/cephadrius/endless/bu_research/dxl/data/dscovr_data/oe_m1m_*.nc"))

    df_mag_list = [None] * len(mag_file_list)
    df_plas_list = [None] * len(plas_file_list)

    count = 0
    for mag_file, plas_file in zip(mag_file_list, plas_file_list):

        mag_data = netcdf.netcdf_file(mag_file, 'r')
        plas_data = netcdf.netcdf_file(plas_file, 'r')

        mag_data_time = mag_data.variables['time'][:]
        plas_data_time = plas_data.variables['time'][:]

        mag_data_bx_gsm = mag_data.variables['bx_gsm'][:].byteswap().newbyteorder()
        mag_data_by_gsm = mag_data.variables['by_gsm'][:].byteswap().newbyteorder()
        mag_data_bz_gsm = mag_data.variables['bz_gsm'][:].byteswap().newbyteorder()
        mag_data_bt_gsm = mag_data.variables['bt'][:].byteswap().newbyteorder()

        plas_data_np = plas_data.variables['proton_density'][:].byteswap().newbyteorder()
        plas_data_vp = plas_data.variables['proton_speed'][:].byteswap().newbyteorder()

        mag_data_time_utc = np.array([datetime.datetime.utcfromtimestamp(t/1.e3)
                                                                            for t in mag_data_time])
        plas_data_time_utc = np.array([datetime.datetime.utcfromtimestamp(t/1.e3)
                                                                           for t in plas_data_time])

        df_mag_list[count] = pd.DataFrame({'bx_gsm': mag_data_bx_gsm, 'by_gsm': mag_data_by_gsm, 'bz_gsm': mag_data_bz_gsm, 'bt': mag_data_bt_gsm}, index=mag_data_time_utc)
        df_plas_list[count] = pd.DataFrame({'np': plas_data_np, 'vp': plas_data_vp}, index=plas_data_time_utc)
        #df_mag_list[count] = pd.DataFrame([mag_data_bx_gsm, mag_data_by_gsm, mag_data_bz_gsm], index=mag_data_time_utc, columns=['bx_gsm', 'by_gsm', 'bz_gsm'])
#
        #df_plas_list[count] = pd.DataFrame([plas_data_np, plas_data_vp], index=plas_data_time_utc, columns=['np', 'vp'])
#
        count += 1

    df_dscovr_mag = pd.concat(df_mag_list, axis=0)
    df_dscovr_plas = pd.concat(df_plas_list, axis=0)

    df_dscovr = pd.concat([df_dscovr_mag, df_dscovr_plas], axis=1)

    # Repldscovr data gaps with NaN
    df_dscovr.replace([-999.9, -99999.0, -100000], np.nan, inplace=True)

    # Save the flux data to the dataframe
    df_dscovr['flux'] = df_dscovr.np * df_dscovr.vp * 1e-3

    # Save the magnitude of magnetic field data to the dataframe
    df_dscovr['bm'] = np.sqrt(df_dscovr.bx_gsm**2 + df_dscovr.by_gsm**2 + df_dscovr.bz_gsm**2)

    # Compute the IMF clock angle and save it to dataframe
    df_dscovr['theta_c'] = np.arctan2(df_dscovr.by_gsm, df_dscovr.bz_gsm)

    # Compute the dynamic pressure of solar wind
    df_dscovr['p_dyn'] = 1.6726e-6 * 1.15 * df_dscovr.np * df_dscovr.vp**2

    # Find index in dataframe which is closest to the value of solar wind assuming 35 minutes
    # propagation time
    indx_min = min(range(len(df_dscovr.index)),
        key=lambda i: abs(df_dscovr.index[i] - df_dscovr.index[-1] + datetime.timedelta(minutes=35)))

    # Set a new series corresponding to the index for the closest value of solar wind
    df_param = df_dscovr.iloc[indx_min]

    # Define the parameter for computing the total magnetic field from Tsyganenko model (T04)
    # NOTE: For the Tsyganenko model, the elemnts of 'param' are solar wind dynamic pressure, DST,
    # y-component of IMF, z-component of IMF, and 6 zeros.
    param = [df_param.p_dyn, 1, df_param.by_gsm, df_param.bz_gsm, 0, 0, 0, 0, 0, 0]

    # Compute the unix time and the dipole tilt angle
    t_unix = datetime.datetime(1970, 1, 1)
    time_dipole = (df_dscovr.index[indx_min] - t_unix).total_seconds()
    ps = gp.recalc(time_dipole)

    # Compute the dipole tilt angle correction, indegrees, to be applied to the cusp locations
    # NOTE: The correctionvalue computed  here is based on the values given in Newell et al. (2006),
    # doi:10.1029/2006JA011731, 2006
    dipole_correction = - 0.046 * ps * 180 / np.pi

    # Compute the location of cusp based on different coupling equations
    df_dscovr['lambda_phi'] = - 3.65e-2 * (df_dscovr.vp * df_dscovr.bt\
                            * np.sin(df_dscovr.theta_c/2.)**4)**(2/3) + 77.2 + dipole_correction
    df_dscovr['lambda_wav'] = - 2.27e-3 * (df_dscovr.vp * df_dscovr.bt\
                            * np.sin(df_dscovr.theta_c/2.)**4) + 78.5 + dipole_correction
    df_dscovr['lambda_vas'] = - 2.14e-4 * df_dscovr.p_dyn**(1/6) * df_dscovr.vp**(4/3) * df_dscovr.bt\
                            * np.sin(df_dscovr.theta_c)**4 + 78.3 + dipole_correction
    df_dscovr['lambda_ekl'] = - 1.90e-3 * df_dscovr.vp * df_dscovr.bt\
                            * np.sin(df_dscovr.theta_c/2.)**2 + 78.9 + dipole_correction

    # Make a copy of the dataframe at original cadence
    df_dscovr_hc = df_dscovr.copy()

    # Compute 1 hr rolling average for each of the parameters and save it to the dataframe
    df_dscovr = df_dscovr.rolling("1H", center=True).median()


    # Make a copy of the dataframe at original cadence
    df_dscovr_hc = df_dscovr.copy()

    # Compute 1 hr rolling average for each of the parameters and save it to the dataframe
    df_dscovr = df_dscovr.rolling("1H", center=True).median()

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

    try:
        plt.close('all')
    except Exception:
        pass

    t1 = df_dscovr.index.max() - datetime.timedelta(minutes=30)
    t2 = df_dscovr.index.max() - datetime.timedelta(minutes=40)

    fig = plt.figure(num=None, figsize=(10, 13), dpi=200, facecolor='w', edgecolor='gray')
    fig.subplots_adjust(left=0.01, right=0.95, top=0.95, bottom=0.01, wspace=0.02, hspace=0.)
    fig.suptitle(f'54 days DSCOVR Real Time Data', fontsize=24)

    # Magnetic field plot
    gs = fig.add_gridspec(6, 1)
    axs1 = fig.add_subplot(gs[0, 0])
    im1a = axs1.plot(df_dscovr.index, df_dscovr.bx_gsm, 'r-', lw=lw, ms=ms, label=r'$B_x$')
    im1b = axs1.plot(df_dscovr.index, df_dscovr.by_gsm, 'b-', lw=lw, ms=ms, label=r'$B_y$')
    im1c = axs1.plot(df_dscovr.index, df_dscovr.bz_gsm, 'g-', lw=lw, ms=ms, label=r'$B_z$')
    im1d = axs1.plot(df_dscovr.index, df_dscovr.bm, 'k-.', lw=lw, ms=ms, label=r'$|\vec{B}|$')
    im1e = axs1.plot(df_dscovr.index, -df_dscovr.bm, 'k-.', lw=lw, ms=ms)
    axs1.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_dscovr.bm.isnull().all():
        axs1.set_ylim([-1, 1])
    else:
        axs1.set_ylim(-1.1 * np.nanmax(df_dscovr.bm), 1.1 * np.nanmax(df_dscovr.bm))

    axs1.set_xlim(df_dscovr.index.min(), df_dscovr.index.max())
    axs1.set_ylabel(r'B [nT]', fontsize=20 )
    lgnd1 = axs1.legend(fontsize=labelsize, loc='best', ncol=ncols)
    lgnd1.legendHandles[0]._sizes = [labelsize]

    #axs1.text(0.98, 0.95, f'{model_type}', horizontalalignment='right', verticalalignment='center',
    #          transform=axs1.transAxes, fontsize=18)

    # Density plot
    axs2 = fig.add_subplot(gs[1, 0], sharex=axs1)
    im2 = axs2.plot(df_dscovr.index, df_dscovr.np, 'r-', lw=lw, ms=ms, label=r'$n_p$')
    axs2.plot(df_dscovr_hc.index, df_dscovr_hc.np, color='r', lw=1, alpha=alpha)
    axs2.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_dscovr.np.isnull().all():
        axs2.set_ylim([0, 1])
    else:
        axs2.set_ylim(0.9 * np.nanmin(df_dscovr.np), 1.1 * np.nanmax(df_dscovr.np))

    lgnd2 = axs2.legend(fontsize=labelsize, loc='best', ncol=ncols)
    lgnd2.legendHandles[0]._sizes = [labelsize]
    axs2.set_ylabel(r'$n_p [1/\rm{cm^{3}}]$', fontsize=ylabelsize)

    # Speed plot
    axs3 = fig.add_subplot(gs[2, 0], sharex=axs1)
    im3 = axs3.plot(df_dscovr.index, df_dscovr.vp, 'b-', lw=lw, ms=ms, label=r'$V_p$')
    axs3.plot(df_dscovr_hc.index, df_dscovr_hc.vp, color='b', lw=1, alpha=alpha)
    axs3.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_dscovr.vp.isnull().all():
        axs3.set_ylim([0, 1])
    else:
        axs3.set_ylim(0.9 * np.nanmin(df_dscovr.vp), 1.1 * np.nanmax(df_dscovr.vp))

    lgnd3 = axs3.legend(fontsize=labelsize, loc='best', ncol=ncols)
    lgnd3.legendHandles[0]._sizes = [labelsize]
    axs3.set_ylabel(r'$V_p [\rm{km/sec}]$', fontsize=ylabelsize)

    # Flux plot
    axs4 = fig.add_subplot(gs[3, 0], sharex=axs1)
    im4 = axs4.plot(df_dscovr.index, df_dscovr.flux, 'g-', lw=lw, ms=ms, label=r'flux')
    im4a = axs4.axhline(y=2.9, xmin=0, xmax=1, color='r', ls='-', lw=lw, ms=ms, label=r'cut-off')
    axs4.plot(df_dscovr_hc.index, df_dscovr_hc.flux, color='g', lw=1, alpha=alpha)
    axs4.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_dscovr.flux.isnull().all():
        axs4.set_ylim([0, 1])
    else:
        axs4.set_ylim(np.nanmin([0.9 * np.nanmin(df_dscovr.flux), 2.4]),
                      np.nanmax([1.1 * np.nanmax(df_dscovr.flux), 3.3]))

    lgnd4 = axs4.legend(fontsize=labelsize, loc='best', ncol=ncols)
    lgnd4.legendHandles[0]._sizes = [labelsize]
    axs4.set_ylabel(r'~~~~Flux\\ $10^8 [\rm{1/(sec\, cm^2)}]$', fontsize=ylabelsize)

    # Cusp latitude plot
    axs5 = fig.add_subplot(gs[4:, 0], sharex=axs1)

    min_lambda = np.nanmin([
                            np.nanmin(df_dscovr.lambda_phi),
                            np.nanmin(df_dscovr.lambda_wav),
                            np.nanmin(df_dscovr.lambda_vas),
                            np.nanmin(df_dscovr.lambda_ekl),
    ])
    max_lambda = np.nanmax([
                            np.nanmax(df_dscovr.lambda_phi),
                            np.nanmax(df_dscovr.lambda_wav),
                            np.nanmax(df_dscovr.lambda_vas),
                            np.nanmax(df_dscovr.lambda_ekl),
    ])

    im5a = axs5.plot(df_dscovr.index, df_dscovr.lambda_phi, 'r-', lw=lw, ms=ms, label=r'$d\phi/dt$')
    #im5b = axs5.plot(df_dscovr.index, df_dscovr.lambda_wav, 'b-', lw=lw, ms=ms, label=r'WAV')
    #im5c = axs5.plot(df_dscovr.index, df_dscovr.lambda_vas, 'g-', lw=lw, ms=ms, label=r'Vas')
    #im5d = axs5.plot(df_dscovr.index, df_dscovr.lambda_ekl, 'm-', lw=lw, ms=ms, label=r'EKL')
    axs5.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if (df_dscovr.lambda_phi.isnull().all() and
        df_dscovr.lambda_wav.isnull().all() and
        df_dscovr.lambda_vas.isnull().all() and
        df_dscovr.lambda_ekl.isnull().all()):
        axs5.set_ylim([-1, 1])
    else:
        axs5.set_ylim(0.97 * min_lambda, 1.03 * max_lambda)

    lgnd5 = axs5.legend(fontsize=labelsize, loc='best', ncol=4)
    lgnd5.legendHandles[0]._sizes = [labelsize]

    count = 0
    for count in range(len(df_dscovr.index)):
        if(~np.isnan(df_dscovr.index.year[count]) or count>=len(df_dscovr.index)):
            break
        else:
            count = count + 1

    axs5.set_ylabel(r'$\lambda[^\circ]$', fontsize=ylabelsize)

    # Set axis tick-parameters
    axs1.tick_params(which='both', direction='in', left=True, labelleft=True, top=True,
                     labeltop=False, right=True, labelright=False, bottom=True, labelbottom=False,
                     width=tickwidth, length=ticklength, labelsize=ticklabelsize, labelrotation=0)

    axs2.tick_params(which='both', direction='in', left=True, labelleft=False, top=True,
                     labeltop=False, right=True, labelright=True, bottom=True, labelbottom=False,
                     width=tickwidth, length=ticklength, labelsize=ticklabelsize, labelrotation=0)
    axs2.yaxis.set_label_position("right")

    axs3.tick_params(which='both', direction='in', left=True, labelleft=True, top=True,
                     labeltop=False, right=True, labelright=False, bottom=True, labelbottom=False,
                     width=tickwidth, length=ticklength, labelsize=ticklabelsize, labelrotation=0)

    axs4.tick_params(which='both', direction='in', left=True, labelleft=False, top=True,
                     labeltop=False, right=True, labelright=True, bottom=True, labelbottom=False,
                     width=tickwidth, length=ticklength, labelsize=ticklabelsize, labelrotation=0)
    axs4.yaxis.set_label_position("right")

    axs5.tick_params(which='both', axis='y', direction='in', left=True, labelleft=True, top=True,
                     labeltop=False, right=True, labelright=False, bottom=True, labelbottom=True,
                     width=tickwidth, length=ticklength, labelsize=ticklabelsize, labelrotation=0)
    axs5.yaxis.set_label_position("left")


    date_form = DateFormatter('%m-%d')
    axs5.set_xlabel(
    f'Date and Time starting on {int(df_dscovr.index.year[count])}-{int(df_dscovr.index.month[count])}-{int(df_dscovr.index.day[count])} (UTC) [MM-DD]',
            fontsize=xlabelsize
        )
    axs5.tick_params(which='both', axis='x', direction='in', left=True, labelleft=True, top=True,
                 labeltop=False, right=True, labelright=False, bottom=True, labelbottom=True,
                 width=tickwidth, length=ticklength, labelsize=ticklabelsize, labelrotation=0)



    axs5.xaxis.set_major_formatter(date_form)
    figure_time = f"{datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}"

    axs3.text(-0.1, 0.5, f'Figure plotted on {figure_time[0:10]} at {figure_time[11:]} UTC',
            ha='right', va='center', transform=axs3.transAxes, fontsize=20, rotation='vertical')

    fig_name_git = f"../figures/sw_dscovr_parameters_54days.png"
    fig_name = f"/home/cephadrius/Dropbox/DXL-Figure/sw_dscovr_parameters_54days.png"
    fig_name_gdr = f"/home/cephadrius/google-drive/Studies/Research/bu_research/dxl/figures/sw_dscovr_parameters_54days.png"

    plt.savefig(fig_name_git, bbox_inches='tight', pad_inches=0.05, format='png', dpi=300)
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.05, format='png', dpi=300)
    plt.savefig(fig_name_gdr, bbox_inches='tight', pad_inches=0.05, format='png', dpi=300)
    plt.close("all")
    print("Figure saved for DSCOVR 54 days at (UTC):" +
          f"{datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Waiting for about 60 minutes before running the code again.\n")
    # print(f'It took {round(time.time() - start, 3)} seconds')
    #return df

s.enter(0, 1, plot_figures_dsco_54days, (s,))
s.run()
