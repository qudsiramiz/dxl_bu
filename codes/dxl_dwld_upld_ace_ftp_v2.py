#!/home/cephadrius/.cache/pypoetry/virtualenvs/dxl-bu-jVBu7xQf-py3.10/bin/python
# -*- coding: utf-8 -*-
import csv
import datetime
import sched
import time
from ftplib import FTP

import geopack.geopack as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

s = sched.scheduler(time.time, time.sleep)


def plot_figures_ace_ftp_v2(*args):
    # for xx in range(1,2):
    #    plot_duration=1
    """
    Download ACE data from FTP server, and make plots corresponding to the plot duration in days.

    Parameters
    ----------
    plot_duration : int, optional
        Plot duration in days. The default is 1 day

    Returns
    -------
    Plot figures.
    """
    # Set up the time to run the job
    # s.enter(60, 1, plot_figures_ace_7days, (sc,))

    plot_duration = args[0]
    # start = time.time()
    print(
        f"Code execution for ACE {plot_duration} days data started at at (UTC):"
        + f"{datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Set the font style to Times New Roman
    font = {"family": "serif", "weight": "normal", "size": 10}
    plt.rc("font", **font)
    plt.rc("text", usetex=True)

    fl_path = f"/home/cephadrius/Desktop/git/dxl_bu/data/ace_ftp_data/"

    ftp = FTP("ftp.swpc.noaa.gov")
    ftp.login()
    ftp.cwd("/pub/lists/ace/")
    swepam_files = np.sort(ftp.nlst("*_ace_swepam_1m*"))
    mag_files = np.sort(ftp.nlst("*_ace_mag_1m*"))

    # Make list of swepam and mag files
    swepam_files_list = []
    mag_files_list = []

    for fl in swepam_files[-plot_duration:]:
        try:
            # Download the ACE data from website
            ftp.retrbinary("RETR " + fl, open(f"{fl_path}{fl}", "wb").write)
            swepam_files_list.append(f"{fl_path}{fl}")
        except Exception as e:
            pass

    for fl in mag_files[-plot_duration:]:
        try:
            # Download the ACE data from website
            ftp.retrbinary("RETR " + fl, open(f"{fl_path}{fl}", "wb").write)
            mag_files_list.append(f"{fl_path}{fl}")
        except Exception as e:
            pass

    # List of keys for the two files
    ace_key_list_mag = [
        "year",
        "month",
        "date",
        "utctime",
        "julian_day",
        "doy",
        "s",
        "bx_gsm",
        "by_gsm",
        "bz_gsm",
        "bt",
        "lat_gsm",
        "lon_gsm",
    ]
    ace_key_list_swp = [
        "year",
        "month",
        "date",
        "utctime",
        "julian_day",
        "doy",
        "s",
        "np",
        "vp",
        "Tp",
    ]

    # swp and mag dataframe list
    swp_df_list = [None] * len(swepam_files_list)
    mag_df_list = [None] * len(mag_files_list)

    count = 0
    # Read the sweap and mag data from the file lists
    for swp_fl, mag_fl in zip(swepam_files_list, mag_files_list):
        # Read data from sweap and magnetometer in a dataframe
        mag_df_list[count] = pd.read_csv(
            mag_fl,
            sep=r"\s{1,}",
            skiprows=20,
            names=ace_key_list_mag,
            engine="python",
            dtype={"month": "string", "date": "string", "utctime": "string"},
        )
        swp_df_list[count] = pd.read_csv(
            swp_fl,
            sep=r"\s{1,}",
            skiprows=18,
            names=ace_key_list_swp,
            engine="python",
            dtype={"month": "string", "date": "string", "utctime": "string"},
        )
        count += 1

    # Concatenate all swp and mag dataframes
    df_ace_mag = pd.concat(mag_df_list, ignore_index=True)
    df_ace_swp = pd.concat(swp_df_list, ignore_index=True)

    # Replace data gaps with NaN
    df_ace_mag.replace([-999.9, -100000], np.nan, inplace=True)
    df_ace_swp.replace([-9999.9, -100000], np.nan, inplace=True)

    # Set the indices of two dataframes to datetime objects/timestamps
    df_ace_mag.index = np.array(
        [
            datetime.datetime.strptime(
                f"{df_ace_mag.year[i]}{df_ace_mag.month[i]}{df_ace_mag.date[i]}{df_ace_mag.utctime[i]}",
                "%Y%m%d%H%M",
            )
            for i in range(len(df_ace_mag.index))
        ]
    )

    df_ace_swp.index = np.array(
        [
            datetime.datetime.strptime(
                f"{df_ace_swp.year[i]}{df_ace_swp.month[i]}{df_ace_swp.date[i]}{df_ace_swp.utctime[i]}",
                "%Y%m%d%H%M",
            )
            for i in range(len(df_ace_swp.index))
        ]
    )

    # Combine the two dataframes in one single dataframe along the column/index
    df_ace_all = pd.concat([df_ace_mag, df_ace_swp], axis=1)

    # Set the minimum time to 25 hours or 7 days before the current time
    min_time = datetime.datetime.utcfromtimestamp(time.time()) - datetime.timedelta(
        days=plot_duration
    )

    # Select the dataframe with the minimum time
    df_ace = df_ace_all.loc[df_ace_all.index >= min_time]

    # Remove the duplicate columns
    df_ace = df_ace.loc[:, ~df_ace.columns.duplicated()]

    # Save the flux data to the dataframe
    df_ace["flux"] = df_ace.np * df_ace.vp * 1e-3

    # Save the magnitude of magnetic field data to the dataframe
    df_ace["bm"] = np.sqrt(df_ace.bx_gsm**2 + df_ace.by_gsm**2 + df_ace.bz_gsm**2)

    # Compute the IMF clock angle and save it to dataframe
    df_ace["theta_c"] = np.arctan2(df_ace.by_gsm, df_ace.bz_gsm)

    # Compute the dynamic pressure of solar wind
    df_ace["p_dyn"] = 1.6726e-6 * 1.15 * df_ace.np * df_ace.vp**2

    # Find index in dataframe which is closest to the value of solar wind assuming 35 minutes
    # propagation time
    indx_min = min(
        range(len(df_ace.index)),
        key=lambda i: abs(
            df_ace.index[i] - df_ace.index[-1] + datetime.timedelta(minutes=35)
        ),
    )

    # Set a new series corresponding to the index for the closest value of solar wind
    df_param = df_ace.iloc[indx_min]

    # Define the parameter for computing the total magnetic field from Tsyganenko model (T04)
    # NOTE: For the Tsyganenko model, the elemnts of 'param' are solar wind dynamic pressure, DST,
    # y-component of IMF, z-component of IMF, and 6 zeros.
    param = [df_param.p_dyn, 1, df_param.by_gsm, df_param.bz_gsm, 0, 0, 0, 0, 0, 0]

    # Compute the unix time and the dipole tilt angle
    t_unix = datetime.datetime(1970, 1, 1)
    time_dipole = (df_ace.index[indx_min] - t_unix).total_seconds()
    ps = gp.recalc(time_dipole)

    # Compute the dipole tilt angle correction, indegrees, to be applied to the cusp locations
    # NOTE: The correctionvalue computed  here is based on the values given in Newell et al. (2006),
    # doi:10.1029/2006JA011731, 2006
    dipole_correction = -0.046 * ps * 180 / np.pi

    # Compute the location of cusp based on different coupling equations
    df_ace["lambda_phi"] = (
        -3.65e-2
        * (df_ace.vp * df_ace.bt * np.sin(df_ace.theta_c / 2.0) ** 4) ** (2 / 3)
        + 77.2
        + dipole_correction
    )
    df_ace["lambda_wav"] = (
        -2.27e-3 * (df_ace.vp * df_ace.bt * np.sin(df_ace.theta_c / 2.0) ** 4)
        + 78.5
        + dipole_correction
    )
    df_ace["lambda_vas"] = (
        -2.14e-4
        * df_ace.p_dyn ** (1 / 6)
        * df_ace.vp ** (4 / 3)
        * df_ace.bt
        * np.sin(df_ace.theta_c) ** 4
        + 78.3
        + dipole_correction
    )
    df_ace["lambda_ekl"] = (
        -1.90e-3 * df_ace.vp * df_ace.bt * np.sin(df_ace.theta_c / 2.0) ** 2
        + 78.9
        + dipole_correction
    )

    # Make a copy of the dataframe at original cadence
    df_ace_hc = df_ace.copy()

    # Compute 1 hr rolling average for each of the parameters and save it to the dataframe
    df_ace = df_ace.rolling("1H", center=True).median()

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

    t1 = df_ace.index.max() - datetime.timedelta(minutes=30)
    t2 = df_ace.index.max() - datetime.timedelta(minutes=40)

    fig = plt.figure(
        num=None, figsize=(10, 13), dpi=200, facecolor="w", edgecolor="gray"
    )
    fig.subplots_adjust(
        left=0.01, right=0.95, top=0.95, bottom=0.01, wspace=0.02, hspace=0.0
    )
    fig.suptitle(f"{plot_duration} day ACE Real Time Data", fontsize=24)

    # Magnetic field plot
    gs = fig.add_gridspec(6, 1)
    axs1 = fig.add_subplot(gs[0, 0])
    im1a = axs1.plot(df_ace.index, df_ace.bx_gsm, "r-", lw=lw, ms=ms, label=r"$B_x$")
    im1b = axs1.plot(df_ace.index, df_ace.by_gsm, "b-", lw=lw, ms=ms, label=r"$B_y$")
    im1c = axs1.plot(df_ace.index, df_ace.bz_gsm, "g-", lw=lw, ms=ms, label=r"$B_z$")
    im1d = axs1.plot(df_ace.index, df_ace.bm, "k-.", lw=lw, ms=ms, label=r"$|\vec{B}|$")
    im1e = axs1.plot(df_ace.index, -df_ace.bm, "k-.", lw=lw, ms=ms)
    axs1.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_ace.bm.isnull().all():
        axs1.set_ylim([-1, 1])
    else:
        axs1.set_ylim(-1.1 * np.nanmax(df_ace.bm), 1.1 * np.nanmax(df_ace.bm))

    axs1.set_xlim(df_ace.index.min(), df_ace.index.max())
    axs1.set_ylabel(r"B [nT]", fontsize=20)
    lgnd1 = axs1.legend(fontsize=labelsize, loc="best", ncol=ncols)
    lgnd1.legendHandles[0]._sizes = [labelsize]

    # axs1.text(0.98, 0.95, f'{model_type}', horizontalalignment='right', verticalalignment='center',
    #          transform=axs1.transAxes, fontsize=18)

    # Density plot
    axs2 = fig.add_subplot(gs[1, 0], sharex=axs1)
    im2 = axs2.plot(df_ace.index, df_ace.np, "r-", lw=lw, ms=ms, label=r"$n_p$")
    axs2.plot(df_ace_hc.index, df_ace_hc.np, color="r", lw=1, alpha=alpha)
    axs2.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_ace.np.isnull().all():
        axs2.set_ylim([0, 1])
    else:
        axs2.set_ylim(0.9 * np.nanmin(df_ace.np), 1.1 * np.nanmax(df_ace.np))

    lgnd2 = axs2.legend(fontsize=labelsize, loc="best", ncol=ncols)
    lgnd2.legendHandles[0]._sizes = [labelsize]
    axs2.set_ylabel(r"$n_p [1/\rm{cm^{3}}]$", fontsize=ylabelsize)

    # Speed plot
    axs3 = fig.add_subplot(gs[2, 0], sharex=axs1)
    im3 = axs3.plot(df_ace.index, df_ace.vp, "b-", lw=lw, ms=ms, label=r"$V_p$")
    axs3.plot(df_ace_hc.index, df_ace_hc.vp, color="b", lw=1, alpha=alpha)
    axs3.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_ace.vp.isnull().all():
        axs3.set_ylim([0, 1])
    else:
        axs3.set_ylim(0.9 * np.nanmin(df_ace.vp), 1.1 * np.nanmax(df_ace.vp))

    lgnd3 = axs3.legend(fontsize=labelsize, loc="best", ncol=ncols)
    lgnd3.legendHandles[0]._sizes = [labelsize]
    axs3.set_ylabel(r"$V_p [\rm{km/sec}]$", fontsize=ylabelsize)

    # Flux plot
    axs4 = fig.add_subplot(gs[3, 0], sharex=axs1)
    im4 = axs4.plot(df_ace.index, df_ace.flux, "g-", lw=lw, ms=ms, label=r"flux")
    im4a = axs4.axhline(
        y=2.9, xmin=0, xmax=1, color="r", ls="-", lw=lw, ms=ms, label=r"cut-off"
    )
    axs4.plot(df_ace_hc.index, df_ace_hc.flux, color="g", lw=1, alpha=alpha)
    axs4.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_ace.flux.isnull().all():
        axs4.set_ylim([0, 1])
    else:
        axs4.set_ylim(
            np.nanmin([0.9 * np.nanmin(df_ace.flux), 2.4]),
            np.nanmax([1.1 * np.nanmax(df_ace.flux), 3.3]),
        )

    lgnd4 = axs4.legend(fontsize=labelsize, loc="best", ncol=ncols)
    lgnd4.legendHandles[0]._sizes = [labelsize]
    axs4.set_ylabel(r"~~~~Flux\\ $10^8 [\rm{1/(sec\, cm^2)}]$", fontsize=ylabelsize)

    # Cusp latitude plot
    axs5 = fig.add_subplot(gs[4:, 0], sharex=axs1)

    min_lambda = np.nanmin(
        [
            np.nanmin(df_ace.lambda_phi),
            np.nanmin(df_ace.lambda_wav),
            np.nanmin(df_ace.lambda_vas),
            np.nanmin(df_ace.lambda_ekl),
        ]
    )
    max_lambda = np.nanmax(
        [
            np.nanmax(df_ace.lambda_phi),
            np.nanmax(df_ace.lambda_wav),
            np.nanmax(df_ace.lambda_vas),
            np.nanmax(df_ace.lambda_ekl),
        ]
    )

    im5a = axs5.plot(
        df_ace.index, df_ace.lambda_phi, "r-", lw=lw, ms=ms, label=r"$d\phi/dt$"
    )
    im5b = axs5.plot(df_ace.index, df_ace.lambda_wav, "b-", lw=lw, ms=ms, label=r"WAV")
    im5c = axs5.plot(df_ace.index, df_ace.lambda_vas, "g-", lw=lw, ms=ms, label=r"Vas")
    im5d = axs5.plot(df_ace.index, df_ace.lambda_ekl, "m-", lw=lw, ms=ms, label=r"EKL")
    axs5.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if (
        df_ace.lambda_phi.isnull().all()
        and df_ace.lambda_wav.isnull().all()
        and df_ace.lambda_vas.isnull().all()
        and df_ace.lambda_ekl.isnull().all()
    ):
        axs5.set_ylim([-1, 1])
    else:
        axs5.set_ylim(0.97 * min_lambda, 1.03 * max_lambda)

    lgnd5 = axs5.legend(fontsize=labelsize, loc="best", ncol=4)
    lgnd5.legendHandles[0]._sizes = [labelsize]

    count = 0
    for count in range(len(df_ace.index)):
        if ~np.isnan(df_ace.year[count]) or count >= len(df_ace.index):
            break
        else:
            count = count + 1

    axs5.set_ylabel(r"$\lambda[^\circ]$", fontsize=ylabelsize)

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
    )

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
    )
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
    )

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
    )
    axs4.yaxis.set_label_position("right")

    axs5.tick_params(
        which="both",
        axis="y",
        direction="in",
        left=True,
        labelleft=True,
        top=True,
        labeltop=False,
        right=True,
        labelright=False,
        bottom=True,
        labelbottom=True,
        width=tickwidth,
        length=ticklength,
        labelsize=ticklabelsize,
        labelrotation=0,
    )
    axs5.yaxis.set_label_position("left")

    if plot_duration == 1:
        date_form = DateFormatter("%H:%M")
        axs5.set_xlabel(
            f"Time on {int(df_ace.year[count])}-{int(df_ace.month[count])}-{int(df_ace.date[count])} (UTC) [HH:MM]",
            fontsize=xlabelsize,
        )
        axs5.tick_params(
            which="both",
            axis="x",
            direction="in",
            left=True,
            labelleft=True,
            top=True,
            labeltop=False,
            right=True,
            labelright=False,
            bottom=True,
            labelbottom=True,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=0,
        )
    else:
        date_form = DateFormatter("%d, %H:%M")
        axs5.set_xlabel(
            f"Date and Time starting on {int(df_ace.year[count])}-{int(df_ace.month[count])}-{int(df_ace.date[count])} (UTC) [DD, HH:MM]",
            fontsize=xlabelsize,
        )
        axs5.tick_params(
            which="both",
            axis="x",
            direction="in",
            left=True,
            labelleft=True,
            top=True,
            labeltop=False,
            right=True,
            labelright=False,
            bottom=True,
            labelbottom=True,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=30,
        )

    axs5.xaxis.set_major_formatter(date_form)
    figure_time = f"{datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}"

    axs3.text(
        -0.1,
        0.5,
        f"Figure plotted on {figure_time[0:10]} at {figure_time[11:]} UTC",
        ha="right",
        va="center",
        transform=axs3.transAxes,
        fontsize=20,
        rotation="vertical",
    )

    fig_name_git = f"../figures/sw_ace_parameters_{plot_duration}day.png"
    fig_name = (
        f"/home/cephadrius/Dropbox/DXL-Figure/sw_ace_parameters_{plot_duration}day.png"
    )
    fig_name_gdr = f"/home/cephadrius/google-drive/Studies/Research/bu_research/dxl/figures/sw_ace_parameters_{plot_duration}day.png"

    plt.savefig(
        fig_name_git, bbox_inches="tight", pad_inches=0.05, format="png", dpi=300
    )
    plt.savefig(fig_name, bbox_inches="tight", pad_inches=0.05, format="png", dpi=300)
    plt.savefig(
        fig_name_gdr, bbox_inches="tight", pad_inches=0.05, format="png", dpi=300
    )
    plt.close("all")
    print(
        "Figure saved for ACE at (UTC):"
        + f"{datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    # print(f'It took {round(time.time() - start, 3)} seconds')
    # return df


# s.enter(0, 1, plot_figures_ace, (s,))
# s.run()
