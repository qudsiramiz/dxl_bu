from ftplib import FTP_TLS
from spacepy.pycdf import CDF as cdf
import os.path
import numpy as np
import pandas as pd
year = '2021'
"""
mfi_ftp = FTP_TLS('spdf.gsfc.nasa.gov')
mfi_ftp.login()
mfi_ftp.cwd(f"pub/data/wind/mfi/mfi_h2/{year}")
mfi_file_list = np.sort(mfi_ftp.nlst())
mfi_file_list_27 = mfi_file_list[-27:]
for fl in mfi_file_list_27:
    if os.path.isfile(f"../data/wind_data/mfi/hres/{fl}") == False:
        try:
            mfi_ftp.retrbinary('RETR ' + fl, open(f'../data/wind_data/mfi/hres/{fl}', 'wb').write)
        except:
            print(f'{fl} failed')
mfi_ftp.quit()

swe_ftp = FTP_TLS('spdf.gsfc.nasa.gov')
swe_ftp.login()
swe_ftp.cwd(f"pub/data/wind/swe/swe_faraday/{year}")
swe_file_list = np.sort(swe_ftp.nlst())
swe_file_list_27 = swe_file_list[-27:]
for fl in swe_file_list_27:
    if os.path.isfile(f"../data/wind_data/swe/{fl}") == False:
        try:
            swe_ftp.retrbinary('RETR ' + fl, open(f'../data/wind_data/swe/{fl}', 'wb').write)
        except:
            print(f'{fl} failed')
swe_ftp.quit()

for fl in swe_file_list_27:
    swe_file_name = f"../data/wind_data/swe/{fl}"
    swe_cdf = cdf(swe_file_name)
    df = pd.DataFrame()
    df['time'] = swe_cdf['Epoch'][:]


omni_ftp = FTP_TLS('spdf.gsfc.nasa.gov')
omni_ftp.login()
omni_ftp.cwd(f"pub/data/wind/omni/omni_h2/{year}")
"""
df_omni = pd.read_csv('https://cdaweb.gsfc.nasa.gov/pub/data/omni/high_res_omni/omni_min2021.asc', sep="\s+")
#key_list = ["year", "doy", "hour", "minute", "second", "id_imf", "id_sw", "n_imf", "n_sw", "", 
omni_key_list = []
for i in range(46):
    omni_key_list.append("")

omni_key_list[0] = "year"
omni_key_list[1] = "doy"
omni_key_list[2] = "hour"
omni_key_list[3] = "minute"
omni_key_list[13] = "bm"
omni_key_list[14] = "bx_gsm"
omni_key_list[17] = "by_gsm"
omni_key_list[18] = "bz_gsm"
omni_key_list[21] = "vp"
omni_key_list[25] = "np"
omni_key_list[26] = "Tp"

df_omni.drop([0], inplace=True)

df_omni.columns = omni_key_list
df_omni.index = df_omni["doy"]

df_omni = df_omni.apply(pd.to_numeric, errors='coerce')

df_omni["flux"] = df_omni.np * df_omni.vp * 1e-3