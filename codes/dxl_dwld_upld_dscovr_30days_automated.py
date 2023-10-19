# Code to read a net cdf file
import datetime
import glob
import os
import sched
import time
import re

from matplotlib import colors

import geopack.geopack as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from scipy.io import netcdf

s = sched.scheduler(time.time, time.sleep)


def plot_figures_dsco_30days(number_of_days=30):
    # for xx in range(1):
    #    number_of_days=35
    """
    Download and plot the data from DSCOVR for the last 30 days starting from the present time.
    """

    print(
        f"Code execution for DSCOVR 30 days data started at at (UTC):"
        + f"{datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Set the font style to Times New Roman
    font = {"family": "serif", "weight": "normal", "size": 10}
    plt.rc("font", **font)
    plt.rc("text", usetex=True)

    # Set up the time to run the job
    # s.enter(3600, 1, plot_figures_dsco_30days, (sc,))

    # start = time.time()
    # Find the present time in unix time
    t_unix = datetime.datetime(1970, 1, 1)
    time_now = (datetime.datetime.utcnow() - t_unix).total_seconds()

    # Set the start and end time (in unix time, 30 days interval)
    time_start = time_now - (60 * 60 * 24 * number_of_days)
    time_end = time_now

    # Change the directory to the data directory
    os.chdir("/mnt/cephadrius/bu_research/dxl/data/dscovr_data/")

    # Delete all the files in the directory
    # os.system("rm -rf *")

    # Find the date corresponding to the time
    # date_start = datetime.datetime.utcfromtimestamp(time_start).strftime('%Y-%m-%d')
    # date_end = datetime.datetime.utcfromtimestamp(time_end).strftime('%Y-%m-%d')

    time_mag_data = time_start

    while time_mag_data < time_end:
        year = datetime.datetime.utcfromtimestamp(time_mag_data).strftime("%Y")
        month = datetime.datetime.utcfromtimestamp(time_mag_data).strftime("%m")
        day = datetime.datetime.utcfromtimestamp(time_mag_data).strftime("%d")
        url = f"https://www.ngdc.noaa.gov/dscovr/data/{year}/{str(month).zfill(2)}/"
        mag_file_name = (
            f"oe_m1m_dscovr_s{year}{str(month).zfill(2)}{str(day).zfill(2)}*.gz"
        )

        # Check if the mag file exists in the directory, if not then download it
        mag_pattern = re.compile(f"{mag_file_name[:-4]}")
        mag_file_list_all = os.listdir(os.getcwd())

        mag_file_exist_list = list(filter(mag_pattern.match, mag_file_list_all))
        if len(mag_file_exist_list) == 0:
            print(
                f"Downloading mag data for {year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
            )
            os.system(f"""wget -q -r -np -nc -nH -nd -A {mag_file_name} {url}""")
            time_mag_data += 60 * 60 * 24
        else:
            print(
                f"Mag data for {year}-{str(month).zfill(2)}-{str(day).zfill(2)} already exists"
            )
            time_mag_data += 60 * 60 * 24

        # for mag_files in os.listdir(os.getcwd()):
        #    if pattern.match(mag_files):
        #        print(f"{mag_file_name[:-3]} exists")
        #        time_mag_data += 60 * 60 * 24
        #        continue
        #    else:
        #        print(f"Downloading mag data for {year}-{str(month).zfill(2)}-{str(day).zfill(2)}")
        #        os.system(f"""wget -q -r -np -nc -nH -nd -A {mag_file_name} {url}""")
        #        time_mag_data += 60 * 60 * 24
        #        continue

    time_plasma_data = time_start
    while time_plasma_data < time_end:
        year = datetime.datetime.utcfromtimestamp(time_plasma_data).strftime("%Y")
        month = datetime.datetime.utcfromtimestamp(time_plasma_data).strftime("%m")
        day = datetime.datetime.utcfromtimestamp(time_plasma_data).strftime("%d")
        url = f"https://www.ngdc.noaa.gov/dscovr/data/{year}/{str(month).zfill(2)}/"
        plas_file_name = (
            f"oe_f1m_dscovr_s{year}{str(month).zfill(2)}{str(day).zfill(2)}*.gz"
        )

        # Check if the plasma file exists in the directory, if not then download it
        plas_pattern = re.compile(f"{plas_file_name[:-4]}")
        plas_file_list_all = os.listdir(os.getcwd())

        plas_file_exist_list = list(filter(plas_pattern.match, plas_file_list_all))
        if len(plas_file_exist_list) == 0:
            print(
                f"Downloading plasma data for {year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
            )
            os.system(f"""wget -q -r -np -nc -nH -nd -A {plas_file_name} {url}""")
            time_plasma_data += 60 * 60 * 24
        else:
            print(
                f"Plasma data for {year}-{str(month).zfill(2)}-{str(day).zfill(2)} already exists"
            )
            time_plasma_data += 60 * 60 * 24

        # if not os.path.exists(plas_file_name[:-3]):
        #    print(f"Downloading plas data for {year}-{str(month).zfill(2)}-{str(day).zfill(2)}")
        #    os.system(f"""wget -q -r -np -nc -nH -nd -A {plas_file_name} {url}""")
        #    time_plasma_data += 60 * 60 * 24
        # else:
        #    print(f"File {plas_file_name[:-3]} already exists")
        #    time_plasma_data += (60 * 60 * 24)
        #    continue

    print("Downloading complete\n")

    # Unzip the files and delete the zipped files
    print("Unzipping the files\n")
    try:
        os.system("gunzip oe_*.gz")
        print("Unzipping complete\n")
    except:
        print("No files to unzip\n")

    os.chdir("/home/cephadrius/Desktop/git/dxl_bu/codes")

    plas_file_list = np.sort(
        glob.glob("/mnt/cephadrius/bu_research/dxl/data/dscovr_data/oe_f1m_*.nc")
    )[-number_of_days:]
    mag_file_list = np.sort(
        glob.glob("/mnt/cephadrius/bu_research/dxl/data/dscovr_data/oe_m1m_*.nc")
    )[-number_of_days:]

    df_mag_list = [None] * len(mag_file_list)
    df_plas_list = [None] * len(plas_file_list)

    count = 0
    for mag_file, plas_file in zip(mag_file_list, plas_file_list):
        mag_data = netcdf.netcdf_file(mag_file, "r")
        plas_data = netcdf.netcdf_file(plas_file, "r")

        mag_data_time = mag_data.variables["time"][:]
        plas_data_time = plas_data.variables["time"][:]

        mag_data_bx_gsm = mag_data.variables["bx_gsm"][:].byteswap().newbyteorder()
        mag_data_by_gsm = mag_data.variables["by_gsm"][:].byteswap().newbyteorder()
        mag_data_bz_gsm = mag_data.variables["bz_gsm"][:].byteswap().newbyteorder()
        mag_data_bt_gsm = mag_data.variables["bt"][:].byteswap().newbyteorder()

        plas_data_np = (
            plas_data.variables["proton_density"][:].byteswap().newbyteorder()
        )
        plas_data_vp = plas_data.variables["proton_speed"][:].byteswap().newbyteorder()
        plas_data_tp = (
            plas_data.variables["proton_temperature"][:].byteswap().newbyteorder()
        )

        mag_data_time_utc = np.array(
            [datetime.datetime.utcfromtimestamp(t / 1.0e3) for t in mag_data_time]
        )
        plas_data_time_utc = np.array(
            [datetime.datetime.utcfromtimestamp(t / 1.0e3) for t in plas_data_time]
        )

        df_mag_list[count] = pd.DataFrame(
            {
                "bx_gsm": mag_data_bx_gsm,
                "by_gsm": mag_data_by_gsm,
                "bz_gsm": mag_data_bz_gsm,
                "bt": mag_data_bt_gsm,
            },
            index=mag_data_time_utc,
        )
        df_plas_list[count] = pd.DataFrame(
            {"np": plas_data_np, "vp": plas_data_vp, "tp": plas_data_tp},
            index=plas_data_time_utc,
        )
        # df_mag_list[count] = pd.DataFrame([mag_data_bx_gsm, mag_data_by_gsm, mag_data_bz_gsm], index=mag_data_time_utc, columns=['bx_gsm', 'by_gsm', 'bz_gsm'])
        #
        # df_plas_list[count] = pd.DataFrame([plas_data_np, plas_data_vp], index=plas_data_time_utc, columns=['np', 'vp'])
        #
        count += 1

    df_dscovr_mag = pd.concat(df_mag_list, axis=0)
    df_dscovr_plas = pd.concat(df_plas_list, axis=0)

    df_dscovr_mag_plas = pd.concat([df_dscovr_mag, df_dscovr_plas], axis=1)

    # Replace dscovr data gaps with NaN
    df_dscovr_mag_plas.replace([-999.9, -99999.0, -100000], np.nan, inplace=True)

    # Get rid of rows with same time stamp as index while keeping the latest value
    df_dscovr_hc = df_dscovr_mag_plas[~df_dscovr_mag_plas.index.duplicated(keep="last")]

    # Save the flux data to the dataframe
    df_dscovr_hc["flux"] = df_dscovr_hc.np * df_dscovr_hc.vp * 1e-3

    # Save the magnitude of magnetic field data to the dataframe
    df_dscovr_hc["bm"] = np.sqrt(
        df_dscovr_hc.bx_gsm**2 + df_dscovr_hc.by_gsm**2 + df_dscovr_hc.bz_gsm**2
    )

    # Compute the IMF clock angle and save it to dataframe
    df_dscovr_hc["theta_c"] = np.arctan2(df_dscovr_hc.by_gsm, df_dscovr_hc.bz_gsm)

    # Compute the dynamic pressure of solar wind
    df_dscovr_hc["p_dyn"] = 1.6726e-6 * 1.15 * df_dscovr_hc.np * df_dscovr_hc.vp**2

    # Compute 1 hr rolling average for each of the parameters and save it to the dataframe
    df_dscovr = df_dscovr_hc.rolling("1H", center=True).median()
    df_dscovr_all = df_dscovr.copy()

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
    bar_color = "k"

    ms = 2
    lw = 2
    ncols = 2

    try:
        plt.close("all")
    except Exception:
        pass

    t_start = df_dscovr.index.min()
    t_end = df_dscovr.index.min() + datetime.timedelta(minutes=1440 * 30)

    while t_end <= df_dscovr_hc.index[-1]:
        # df_dscovr = df_dscovr_hc[(df_dscovr_hc.index >= t_start) & (df_dscovr_hc.index <= t_end)]
        df_dscovr = df_dscovr_all.loc[t_start:t_end]

        t1 = datetime.datetime.utcfromtimestamp(t_start.timestamp()).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        t2 = datetime.datetime.utcfromtimestamp(t_end.timestamp()).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        fig_name = f"/mnt/cephadrius/bu_research/dxl/figures/historical/dscovr/30days/sw_dscovr_parameters_30days_{t1}_{t2}.png"

        # Check if the figure already exists
        if os.path.isfile(fig_name):
            print(f"Figure {fig_name[-71:]} already exists. Skipping...")
            t_start = t_start + datetime.timedelta(days=0.25)
            t_end = t_end + datetime.timedelta(days=0.25)
            continue

        fig = plt.figure(
            num=None, figsize=(10, 15), dpi=200, facecolor="w", edgecolor="gray"
        )
        fig.subplots_adjust(
            left=0.01, right=0.95, top=0.95, bottom=0.01, wspace=0.02, hspace=0.0
        )
        fig.suptitle(f"30 days DSCOVR Real Time Data", fontsize=24)

        # Magnetic field plot (x-component)
        gs = fig.add_gridspec(7, 1)
        axs1 = fig.add_subplot(gs[0:1, 0])
        im1a = axs1.plot(
            df_dscovr_hc.index.values,
            df_dscovr_hc.bx_gsm.values,
            "r-",
            alpha=0.3,
            lw=lw,
            ms=ms,
            label=r"$B_x$",
        )
        im1b = axs1.plot(
            df_dscovr.index.values,
            df_dscovr.bx_gsm.values,
            "r-",
            lw=lw,
            ms=ms,
            label=r"$B_x$",
        )

        # im1d = axs1.plot(df_dscovr.index, df_dscovr.bm, 'k-.', lw=lw, ms=ms, label=r'$|\vec{B}|$')
        # im1e = axs1.plot(df_dscovr.index, -df_dscovr.bm, 'k-.', lw=lw, ms=ms)

        if df_dscovr.bm.isnull().all():
            axs1.set_ylim([-1, 1])
        else:
            axs1.set_ylim(
                0.9 * np.nanmin(df_dscovr_all.bx_gsm),
                1.1 * np.nanmax(df_dscovr_all.bx_gsm),
            )

        axs1.set_xlim(df_dscovr.index.min(), df_dscovr.index.max())
        axs1.set_ylabel(r"$B_{\rm x}$ (GSM) [nT]", fontsize=20)
        # lgnd1 = axs1.legend(fontsize=labelsize, loc='best', ncol=ncols)
        # lgnd1.legendHandles[0]._sizes = [labelsize]

        # Magnetic field plot (y-component)
        axs2 = fig.add_subplot(gs[1:2, 0])
        im2a = axs2.plot(
            df_dscovr_hc.index.values,
            df_dscovr_hc.by_gsm.values,
            "g-",
            alpha=0.3,
            lw=lw,
            ms=ms,
            label=r"$B_y$",
        )
        im2b = axs2.plot(
            df_dscovr.index.values,
            df_dscovr.by_gsm.values,
            "g-",
            lw=lw,
            ms=ms,
            label=r"$B_y$",
        )

        if df_dscovr.bm.isnull().all():
            axs2.set_ylim([-1, 1])
        else:
            axs2.set_ylim(
                0.9 * np.nanmin(df_dscovr_all.by_gsm),
                1.1 * np.nanmax(df_dscovr_all.by_gsm),
            )

        axs2.set_xlim(df_dscovr.index.min(), df_dscovr.index.max())
        axs2.set_ylabel(r"$B_{\rm y}$ (GSM) [nT]", fontsize=20)
        # lgnd2 = axs2.legend(fontsize=labelsize, loc='best', ncol=ncols)
        # lgnd2.legendHandles[0]._sizes = [labelsize]

        # Magnetic field plot (z-component)
        axs3 = fig.add_subplot(gs[2:3, 0])
        im3a = axs3.plot(
            df_dscovr_hc.index.values,
            df_dscovr_hc.bz_gsm.values,
            "b-",
            alpha=0.3,
            lw=lw,
            ms=ms,
            label=r"$B_z$",
        )
        im3b = axs3.plot(
            df_dscovr.index.values,
            df_dscovr.bz_gsm.values,
            "b-",
            lw=lw,
            ms=ms,
            label=r"$B_z$",
        )

        if df_dscovr.bm.isnull().all():
            axs3.set_ylim([-1, 1])
        else:
            axs3.set_ylim(
                0.9 * np.nanmin(df_dscovr_all.bz_gsm),
                1.1 * np.nanmax(df_dscovr_all.bz_gsm),
            )

        axs3.set_xlim(df_dscovr.index.min(), df_dscovr.index.max())
        axs3.set_ylabel(r"$B_{\rm z}$ (GSM) [nT]", fontsize=20)
        # lgnd3 = axs3.legend(fontsize=labelsize, loc='best', ncol=ncols)
        # lgnd3.legendHandles[0]._sizes = [labelsize]

        # axs1.text(0.98, 0.95, f'{model_type}', horizontalalignment='right', verticalalignment='center',
        #          transform=axs1.transAxes, fontsize=18)

        # Magnetic field plot (z-component)
        axs4 = fig.add_subplot(gs[3:4, 0])
        im4a = axs4.plot(
            df_dscovr_hc.index.values,
            df_dscovr_hc.bm.values,
            "k-",
            alpha=0.3,
            lw=lw,
            ms=ms,
            label=r"|$\vec{B}$|",
        )
        im4b = axs4.plot(
            df_dscovr.index, df_dscovr.bm, "k-", lw=lw, ms=ms, label=r"$B_z$"
        )

        if df_dscovr.bm.isnull().all():
            axs4.set_ylim([-1, 1])
        else:
            axs4.set_ylim(
                0.8 * np.nanmin(df_dscovr_all.bm), 1.2 * np.nanmax(df_dscovr_all.bm)
            )

        axs4.set_xlim(df_dscovr.index.min(), df_dscovr.index.max())
        axs4.set_ylabel(r"B [nT]", fontsize=20)

        axs4.set_yscale("log")

        # Density plot
        axs5 = fig.add_subplot(gs[4:5, 0], sharex=axs1)
        im5 = axs5.plot(
            df_dscovr.index.values,
            df_dscovr.np.values,
            "r-",
            lw=lw,
            ms=ms,
            label=r"$n_p$",
        )
        axs5.plot(
            df_dscovr_hc.index.values,
            df_dscovr_hc.np.values,
            color="r",
            lw=1,
            alpha=alpha,
        )

        if df_dscovr.np.isnull().all():
            axs5.set_ylim([0, 1])
        else:
            axs5.set_ylim(
                0.9 * np.nanmin(df_dscovr_all.np), 1.1 * np.nanmax(df_dscovr_all.np)
            )

        axs5.set_yscale("log")
        # lgnd4 = axs5.legend(fontsize=labelsize, loc='best', ncol=ncols)
        # lgnd4.legendHandles[0]._sizes = [labelsize]
        axs5.set_ylabel(r"$n_{\rm p} [1/\rm{cm^{3}}]$", fontsize=ylabelsize)

        axs5a = axs5.twinx()
        im5a = axs5a.plot(
            df_dscovr.index.values,
            df_dscovr.flux.values,
            "g-",
            lw=lw,
            ms=ms,
            label=r"flux",
        )
        axs5a.plot(
            df_dscovr_hc.index.values,
            df_dscovr_hc.flux.values,
            color="g",
            lw=1,
            alpha=alpha,
        )

        if df_dscovr.flux.isnull().all():
            axs5a.set_ylim([0, 1])
        else:
            axs5a.set_ylim(
                np.nanmin([0.9 * np.nanmin(df_dscovr_all.flux), 2.4]),
                np.nanmax([1.1 * np.nanmax(df_dscovr_all.flux), 3.3]),
            )

        # lgnd4a = axs5a.legend(fontsize=labelsize, loc='best', ncol=ncols)
        # lgnd4a.legendHandles[0]._sizes = [labelsize]
        axs5a.set_ylabel(
            r"~~~~Flux\\ $10^8 [\rm{1/(sec\, cm^2)}]$", fontsize=ylabelsize
        )

        # Speed plot
        axs6 = fig.add_subplot(gs[5:6, 0], sharex=axs1)
        im6 = axs6.plot(
            df_dscovr.index.values,
            df_dscovr.vp.values,
            "b-",
            lw=lw,
            ms=ms,
            label=r"$V_p$",
        )
        axs6.plot(
            df_dscovr_hc.index.values,
            df_dscovr_hc.vp.values,
            color="b",
            lw=1,
            alpha=alpha,
        )

        if df_dscovr.vp.isnull().all():
            axs6.set_ylim([0, 1])
        else:
            axs6.set_ylim(
                0.9 * np.nanmin(df_dscovr_all.vp), 1.1 * np.nanmax(df_dscovr_all.vp)
            )

        # lgnd5 = axs6.legend(fontsize=labelsize, loc='best', ncol=ncols)
        # lgnd5.legendHandles[0]._sizes = [labelsize]
        axs6.set_ylabel(r"$V_p [\rm{km/sec}]$", fontsize=ylabelsize)

        # Flux plot
        # axs6a = fig.add_subplot(gs[5:6, 0], sharex=axs1)
        axs6a = axs6.twinx()
        im6a = axs6a.plot(
            df_dscovr.index.values,
            df_dscovr.tp.values / 1.0e5,
            "g-",
            lw=lw,
            ms=ms,
            label=r"flux",
        )
        axs6a.plot(
            df_dscovr_hc.index.values,
            df_dscovr_hc.tp.values,
            color="g",
            lw=1,
            alpha=alpha,
        )

        if df_dscovr.tp.isnull().all():
            axs6a.set_ylim([0, 1])
        else:
            axs6a.set_ylim(
                0.9 * np.nanmin(df_dscovr_all.tp) / 1.0e5,
                1.1 * np.nanmax(df_dscovr_all.tp) / 1.0e5,
            )

        axs6a.set_yscale("log")

        # lgnd6 = axs6a.legend(fontsize=labelsize, loc='best', ncol=ncols)
        # lgnd6.legendHandles[0]._sizes = [labelsize]
        axs6a.set_ylabel(r"$T_{\rm p} [10^5~\rm{K}]$", fontsize=ylabelsize)

        # Set axis tick-parameters
        axs1.tick_params(
            which="both",
            direction="in",
            left=True,
            labelleft=True,
            top=True,
            labeltop=False,
            right=True,
            labelright=False,
            bottom=True,
            labelbottom=False,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=0,
            colors="r",
        )
        axs1.spines["left"].set_color("r")
        axs1.spines["left"].set_linewidth(2)
        axs1.yaxis.label.set_color("r")
        axs1.yaxis.set_label_position("left")

        axs2.tick_params(
            which="both",
            direction="in",
            left=True,
            labelleft=False,
            top=True,
            labeltop=False,
            right=True,
            labelright=True,
            bottom=True,
            labelbottom=False,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=0,
            colors="g",
        )
        axs2.spines["right"].set_color("g")
        axs2.spines["right"].set_linewidth(2)
        axs2.yaxis.label.set_color("g")
        axs2.yaxis.set_label_position("right")

        axs3.tick_params(
            which="both",
            direction="in",
            left=True,
            labelleft=True,
            top=True,
            labeltop=False,
            right=True,
            labelright=False,
            bottom=True,
            labelbottom=False,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=0,
            colors="b",
        )
        axs3.spines["left"].set_color("b")
        axs3.spines["left"].set_linewidth(2)
        axs3.yaxis.label.set_color("b")
        axs3.yaxis.set_label_position("left")

        axs4.tick_params(
            which="both",
            direction="in",
            left=True,
            labelleft=False,
            top=True,
            labeltop=False,
            right=True,
            labelright=True,
            bottom=True,
            labelbottom=False,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=0,
            colors="k",
        )
        axs4.spines["left"].set_color("k")
        axs4.spines["left"].set_linewidth(2)
        axs4.yaxis.label.set_color("k")
        axs4.yaxis.set_label_position("right")

        axs5.tick_params(
            which="both",
            direction="in",
            left=False,
            labelleft=False,
            top=True,
            labeltop=False,
            right=True,
            labelright=True,
            bottom=True,
            labelbottom=False,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=0,
            colors="r",
        )
        axs5.spines["right"].set_color("r")
        axs5.spines["right"].set_linewidth(2)
        axs5.yaxis.label.set_color("r")
        axs5.yaxis.set_label_position("right")

        axs5a.tick_params(
            which="both",
            direction="in",
            left=True,
            labelleft=True,
            top=True,
            labeltop=False,
            right=False,
            labelright=False,
            bottom=True,
            labelbottom=False,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=0,
            colors="g",
        )
        axs5a.spines["left"].set_color("g")
        axs5a.spines["left"].set_linewidth(2)
        axs5a.yaxis.label.set_color("g")
        axs5a.yaxis.set_label_position("left")

        axs6.tick_params(
            which="both",
            axis="y",
            direction="in",
            left=True,
            labelleft=True,
            top=True,
            labeltop=False,
            right=False,
            labelright=False,
            bottom=True,
            labelbottom=False,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=0,
            colors="b",
        )
        axs6.spines["left"].set_color("b")
        axs6.spines["left"].set_linewidth(2)
        axs6.yaxis.label.set_color("b")
        axs6.yaxis.set_label_position("left")

        axs6a.tick_params(
            which="both",
            axis="y",
            direction="in",
            left=False,
            labelleft=False,
            top=True,
            labeltop=False,
            right=True,
            labelright=True,
            bottom=True,
            labelbottom=True,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=0,
            colors="g",
        )
        axs6a.spines["right"].set_color("g")
        axs6a.spines["right"].set_linewidth(2)
        axs6a.yaxis.label.set_color("g")
        axs6a.yaxis.set_label_position("right")

        axs6.tick_params(
            which="both",
            axis="x",
            direction="in",
            left=False,
            labelleft=False,
            top=True,
            labeltop=False,
            right=False,
            labelright=False,
            bottom=True,
            labelbottom=True,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=0,
            colors="k",
        )
        date_form = DateFormatter("%m-%d")
        # axs6.xaxis.label.set_color('k')
        axs6.set_xlabel(
            f"Date and Time starting on {int(df_dscovr.index.year[count])}-{str(int(df_dscovr.index.month[count])).zfill(2)}-{str(int(df_dscovr.index.day[count])).zfill(2)} (UTC) [MM-DD]",
            fontsize=xlabelsize,
        )
        axs6.xaxis.set_major_formatter(date_form)

        figure_time = f"{datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}"

        plt.savefig(
            fig_name, bbox_inches="tight", pad_inches=0.05, format="png", dpi=300
        )
        plt.close("all")
        print(f"Figure saved for DSCOVR 30 days starting from {t_start} to {t_end}\n")

        t_start = t_start + datetime.timedelta(days=0.25)
        t_end = t_end + datetime.timedelta(days=0.25)

    # print(f'It took {round(time.time() - start, 3)} seconds')
    # return df


if __name__ == "__main__":
    plot_figures_dsco_30days()
