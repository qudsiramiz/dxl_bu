import datetime
import sched
import time

import geopack.geopack as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

from dxl_dwld_upld_ace import plot_figures_ace
from dxl_dwld_upld_dscovr import plot_figures_dsco
from dxl_dwld_upld_dscovr_1day import plot_figures_dsco_1day
from dxl_dwld_upld_dscovr_7days import plot_figures_dsco_7days
from dxl_dwld_upld_ace_ftp import plot_figures_ace_ftp
from dxl_dwld_upld_ace_ftp_v2 import plot_figures_ace_ftp_v2

s_exec = sched.scheduler(time.time, time.sleep)


def execute_all_files(sc):
    """
    Execute all files used for plotting figures.

    Parameters
    ----------
    sc : sched.scheduler
        Scheduler object.

    Returns
    -------
    None.
    """
    s_exec.enter(60, 1, execute_all_files, (sc,))

    print(f"Code execution started at (UTC):" +
          f"{datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}\n")

    s_ace = sched.scheduler(time.time, time.sleep)
    s_dscovr = sched.scheduler(time.time, time.sleep)

    s_dscovr1 = sched.scheduler(time.time, time.sleep)
    s_ace1 = sched.scheduler(time.time, time.sleep)

    s_dscovr7 = sched.scheduler(time.time, time.sleep)
    s_ace7 = sched.scheduler(time.time, time.sleep)

    #s_ace27 = sched.scheduler(time.time, time.sleep)

    try:
        s_dscovr.enter(0, 1, plot_figures_dsco)
        s_dscovr.run()
    except Exception as e:
        print(e)
        pass

    try:
        s_ace.enter(0, 1, plot_figures_ace)
        s_ace.run()
    except Exception as e:
        print(e)
        pass

    try:
        s_dscovr1.enter(0, 1, plot_figures_dsco_1day)
        s_dscovr1.run()
    except Exception as e:
        print(e)
        pass

    try:
        s_ace1.enter(0, 1, plot_figures_ace_ftp_v2, [1])
        s_ace1.run()
    except Exception as e:
        print(e)
        pass

    try:
        s_dscovr7.enter(0, 1, plot_figures_dsco_7days)
        s_dscovr7.run()
    except Exception as e:
        print(e)
        pass

    try:
        s_ace7.enter(0, 1, plot_figures_ace_ftp_v2, [7])
        s_ace7.run()
    except Exception as e:
        print(e)
        pass

    # try:
        #s_ace27.enter(0, 1, plot_figures_ace_ftp_v2, [27])
        #s_ace27.run()
    # except Exception as e:
        #print(e)

s_exec.enter(0, 1, execute_all_files, (s_exec,))
s_exec.run()
