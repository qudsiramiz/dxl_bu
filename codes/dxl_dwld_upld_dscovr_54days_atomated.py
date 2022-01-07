import datetime
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from scipy.io import netcdf

# Find the present time in unix time
t_unix = datetime.datetime(1970, 1, 1)
time_now = (datetime.datetime.utcnow() - t_unix).total_seconds()
time_start = time_now - (60 * 60 * 24 * 14)
time_end = time_now
os.chdir("/media/cephadrius/endless/bu_research/dxl/data/dscovr_data/")
# Find the date corresponding to the time
date_start = datetime.datetime.utcfromtimestamp(time_start).strftime('%Y-%m-%d')
date_end = datetime.datetime.utcfromtimestamp(time_end).strftime('%Y-%m-%d')

time_data = time_start
while time_data < time_end:
    year = datetime.datetime.utcfromtimestamp(time_data).strftime('%Y')
    month = datetime.datetime.utcfromtimestamp(time_data).strftime('%m')
    day = datetime.datetime.utcfromtimestamp(time_data).strftime('%d')
    print(day)
    url = f"https://www.ngdc.noaa.gov/dscovr/data/{year}/{str(month).zfill(2)}/"
    file_name = f'oe_f1m_dscovr_s{year}{str(month).zfill(2)}{str(day).zfill(2)}*.gz'
    os.system(f"""wget -r -np -nc -nH -nd -A {file_name} {url}""")
    time_data += 60 * 60 * 24

print("Downloading complete\n")

# Unzip the files and delete the zipped files
print("Unzipping the files\n")
os.system("gunzip oe_f1m_*.gz")

os.chdir("/home/cephadrius/Desktop/git/dxl_bu/codes")
