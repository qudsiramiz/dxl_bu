import os
import time

time_code_start = time.time()

os.system("python dxl_dwld_upld_dscovr_30days_automated.py")
os.system("python dxl_mp4_dscovr.py")
print(f"Time taken: {round(time.time() - time_code_start, 2)} seconds")