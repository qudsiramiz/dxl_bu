import datetime
import itertools
import multiprocessing as mp
import time

import geopack.geopack as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from dateutil import parser
from mayavi import mlab
from mpl_toolkits.mplot3d import Axes3D

#from func_t01 import func_t01
start = time.time()
import sched

from matplotlib.patches import Circle, Wedge

font = {'family': 'serif', 'weight': 'normal', 'size': 10}
plt.rc('font', **font)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{color}')

today_date = datetime.datetime.today().strftime('%Y-%m-%d')

s = sched.scheduler(time.time, time.sleep)

def dual_half_circle(center=(0,0), radius=1, angle=90, ax=None, colors=('w','k','k'),
                     **kwargs):
    """
    Add two half circles to the axes *ax* (or the current axes) with the 
    specified facecolors *colors* rotated at *angle* (in degrees).
    """
    if ax is None:
        ax = plt.gca()
    theta1, theta2 = angle, angle + 180
    #w1 = Wedge(center, radius, theta1, theta2, fc=colors[0], **kwargs)
    #w2 = Wedge(center, radius, theta2, theta1, fc=colors[1], **kwargs)
    
    w1 = Wedge(center, radius, theta1, theta2, fc=colors[1], **kwargs)
    w2 = Wedge(center, radius, theta2, theta1, fc=colors[0], **kwargs)
   
    cr = Circle(center, radius, fc=colors[2], fill=False, **kwargs)
    for wedge in [w1, w2, cr]:
        ax.add_artist(wedge)
    return [w1, w2, cr]

def setup_fig(xlim=(15,-30),ylim=(-15, 15), xlabel='X GSM [Re]', ylabel='Z GSM [Re]'):

    fig = plt.figure(figsize=(6, 4))
    ax  = fig.add_subplot(111)
    ax.axvline(0,ls=':', color='k')
    ax.axhline(0,ls=':', color='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_aspect('equal')
    w1,w2,cr = dual_half_circle(ax=ax)

    return ax
# ax = setup_fig(xlim=(-10,10),ylim=(-10,10))

def trace_lines(*args):
    """
    Trace lines from the field line to the surface of the Earth.
    """
    dscovr_url_mag = "https://services.swpc.noaa.gov/products/solar-wind/mag-2-hour.json"
    dscovr_url_plas = "https://services.swpc.noaa.gov/products/solar-wind/plasma-2-hour.json"
    dscovr_url_eph = "https://services.swpc.noaa.gov/products/solar-wind/ephemerides.json"

    dscovr_key_list_mag = ["time_tag", "bx_gsm", "by_gsm", "bz_gsm", "lon_gsm", "lat_gsm", "bt"]
    dscovr_key_list_plas = ["time_tag", "np", "vp", "Tp"]
    dscovr_key_list_eph = ["time_tag", "x_gse", "y_gse", "z_gse", "vx_gse", "vy_gse", "vz_gse",
                            "x_gsm", "y_gsm", "z_gsm", "vx_gsm", "vy_gsm", "vz_gsm"]

    df_dsco_mag = pd.read_json(dscovr_url_mag, orient='columns')
    df_dsco_plas = pd.read_json(dscovr_url_plas, orient='columns')
    df_dsco_eph = pd.read_json(dscovr_url_eph, orient='columns')

    # Drop the first row of the dataframe to get rid of all strings
    df_dsco_mag.drop([0], inplace=True)
    df_dsco_plas.drop([0], inplace=True)
    df_dsco_eph.drop([0], inplace=True)

    # Set column names to the list of keys
    df_dsco_mag.columns = dscovr_key_list_mag
    df_dsco_plas.columns = dscovr_key_list_plas
    df_dsco_eph.columns = dscovr_key_list_eph

    # Set the index to the time_tag column and convert it to a datetime object
    df_dsco_mag.index = pd.to_datetime(df_dsco_mag.time_tag)
    df_dsco_plas.index = pd.to_datetime(df_dsco_plas.time_tag)
    df_dsco_eph.index = pd.to_datetime(df_dsco_eph.time_tag)

    # Drop the time_tag column
    df_dsco_mag.drop(["time_tag"], axis=1, inplace=True)
    df_dsco_plas.drop(["time_tag"], axis=1, inplace=True)
    df_dsco_eph.drop(["time_tag"], axis=1, inplace=True)

    df_dsco_eph = df_dsco_eph[(df_dsco_eph.index >= 
                               np.nanmin([df_dsco_mag.index.min(), df_dsco_plas.index.min()])) & 
                              (df_dsco_eph.index <= 
                               np.nanmax([df_dsco_mag.index.max(), df_dsco_plas.index.max()]))]

    df_dsco = pd.concat([df_dsco_mag, df_dsco_plas, df_dsco_eph], axis=1)

    # for key in df_dsco.keys():
    #     df_dsco[key] = pd.to_numeric(df_dsco[key])
    df_dsco = df_dsco.apply(pd.to_numeric)
    # Save the flux data to the dataframe
    df_dsco['flux'] = df_dsco.np * df_dsco.vp * 1e-3

    # Save the magnitude of magnetic field data to the dataframe
    df_dsco['bm'] = np.sqrt(df_dsco.bx_gsm**2 + df_dsco.by_gsm**2 + df_dsco.bz_gsm**2)

    # Compute the IMF clock angle and save it to dataframe
    df_dsco['theta_c'] = np.arctan2(df_dsco.by_gsm, df_dsco.bz_gsm)

    # Compute the dynamic pressure of solar wind
    df_dsco['p_dyn'] = 1.6726e-6 * 1.15 * df_dsco.np * df_dsco.vp**2

    # Find index in dataframe which is closest to the value of solar wind assuming 35 minutes
    # propagation time
    indx_min = min(range(len(df_dsco.index)),
        key=lambda i: abs(df_dsco.index[i] - df_dsco.index[-1] + datetime.timedelta(minutes=35)))

    # Set a new series corresponding to the index for the closest value of solar wind
    df_param = df_dsco.iloc[indx_min]

    # Define the parameter for computing the total magnetic field from Tsyganenko model (T04)
    # NOTE: For the Tsyganenko model, the elemnts of 'param' are solar wind dynamic pressure, DST,
    # y-component of IMF, z-component of IMF, and 6 zeros.
    param = [df_param.p_dyn, 1, df_param.by_gsm, df_param.bz_gsm, 0, 0, 0, 0, 0, 0]

    # Compute the unix time and the dipole tilt angle
    t_unix = datetime.datetime(1970, 1, 1)
    time_dipole = (df_dsco.index[indx_min] - t_unix).total_seconds()
    ps = gp.recalc(time_dipole)

    theta = args[0][0]
    phi = args[0][1]
    x_gsm = np.sin(theta) * np.cos(phi)
    y_gsm = np.sin(theta) * np.sin(phi)
    z_gsm = np.cos(theta)
    x,y,z,xx1,yy1,zz1 = gp.trace(x_gsm,y_gsm,z_gsm, dir=-1, rlim=30, r0=.99999, parmod=param,
                              exname='t96',inname='igrf',maxloop=10000)
    x,y,z,xx2,yy2,zz2 = gp.trace(x_gsm,y_gsm,z_gsm, dir=1, rlim=30, r0=.99999, parmod=param,
                              exname='t96',inname='igrf',maxloop=10000)
    return xx1, yy1, zz1, xx2, yy2, zz2, ps

def line_trace(sd):
#for xxx in range(1):    
    """
    Create a line trace of the field lines using real time data.
    """

    # Set up the time to run the job
    s.enter(1000, 1, line_trace, (sd,))

    start = time.time()
    print(f"Code execution started at (UTC):" +
          f"{datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}")

    theta_arr = np.linspace(0, 2 * np.pi, 60)
    phi_arr = np.array([0])  # np.linspace(-11 * np.pi/180, 11*np.pi/180, 2)

    try:
        plt.close("all")
    except:
        pass

    p = mp.Pool(30)
    input = ((i, j) for i,j in itertools.product(theta_arr, phi_arr))
    res = p.map(trace_lines, input)
    p.close()
    p.join()

    ax=setup_fig()
    for r in res:
        xx1 = r[0]
        yy1 = r[1]
        zz1 = r[2]
        xx2 = r[3]
        yy2 = r[4]
        zz2 = r[5]
        ps = r[6]
        ax.plot(xx1,zz1)
        ax.plot(xx2,zz2)


    figure_time = f"{datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}"

    ax.text(0.01, 0.01, f'Figure plotted on {figure_time[0:10]} at {figure_time[11:]} UTC',
            ha='left', va='bottom', transform=ax.transAxes, fontsize=12)
    ax.text(0.99, 0.99, f'Real-time T-96 model', ha='right', va='top', transform=ax.transAxes, 
            fontsize=12)
    ax.text(0.01, 0.99, f'Dipole Tilt: {np.round(np.rad2deg(ps), 2)}$^\circ$', 
            ha='left', va='top', transform=ax.transAxes,
            fontsize=12)
    fig_name = f"../figures/Earths_magnetic_field.png"

    plt.tight_layout()
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.05, format='png', dpi=300)

    print(f"Code execution finished at (UTC):" +
          f"{datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"Figure saved at (UTC):{figure_time}\n")
    print(f"Waiting for a preset amount of time before running the code again.\n")
    

s.enter(0, 1, line_trace, (s,))
s.run()
