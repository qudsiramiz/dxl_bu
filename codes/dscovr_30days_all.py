import os
import time

from dxl_dwld_upld_dscovr_30days_automated import plot_figures_dsco_30days
from dxl_mp4_dscovr import make_gifs
time_code_start = time.time()
number_of_days = 365
plot_figures_dsco_30days(number_of_days=number_of_days)
make_gifs(number_of_days=number_of_days)
#os.system("python dxl_dwld_upld_dscovr_30days_automated.py")
#os.system("python dxl_mp4_dscovr.py")
print(f"Time taken: {round(time.time() - time_code_start, 2)} seconds")