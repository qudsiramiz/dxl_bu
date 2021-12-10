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

s = sched.scheduler(time.time, time.sleep)


def execute_all_files(sc):

    s.enter(60, 1, plot_figures_dsco, (sc,))

    print(f"Code execution started at at (UTC):" +
          f"{datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}")

    s_ace = sched.scheduler(time.time, time.sleep)
    s_dscovr = sched.scheduler(time.time, time.sleep)

    s_ace.enter(0, 1, plot_figures_ace, (s_ace,))
    s_ace.run()
    s_dscovr.enter(0, 1, plot_figures_dsco, (s_dscovr,))
    s_dscovr.run()

s.enter(0, 1, execute_all_files, (s,))
s.run()
