import glob as glob
import numpy as np
import imageio as iio
from pygifsicle import optimize
import sched
import datetime
import time

s = sched.scheduler(time.time, time.sleep)

def gif_maker(file_list, gif_name, mode="I", skip_rate=10, duration=0.05):
    """
    Make a gif from a list of images.
    
    Parameters
    ----------
    file_list : list
        List of image files.
    gif_name : str
        Name of the gif file.
    mode : str, optional
        Mode of the gif. The default is "I".
    skip_rate : int, optional
        Skip rate of the gif. The default is 10.
    duration : float, optional
        Duration for which each image is displayed in gif. The default is 0.05.

    Raises
    ------
    ValueError
        If the skip_rate is not an integer.
    ValueError
        If the duration is not a float.
    ValueError
        If the file_list is empty.
    ValueError
        If gif_name is empty.

    Returns
    -------
    None.
    """
    if file_list is None:
        raise ValueError("file_list is None")
    if gif_name is None:
        raise ValueError("gif_name is None. Please provide the name of the gif")

    if len(file_list) == 0:
        raise ValueError("file_list is empty")
    if len(file_list) >= 1501:
        # Check if the skip_rate is an integer
        if skip_rate != int(skip_rate):
            raise ValueError("skip_rate must be an integer")
        file_list = file_list[-1500::skip_rate]
    if duration != float(duration):
        raise ValueError("duration must be a float")

    count = 0
    with iio.get_writer(gif_name, mode=mode, duration=duration) as writer:
        for fl in file_list:
            image = iio.imread(fl)
            writer.append_data(image)
            count += 1
            print(f"{count} images added to the gif")
    writer.close()
    #optimize(gif_name)

    print(f"{gif_name} is created\n")

#gif_path = "/home/cephadrius/Dropbox/DXL-Figure/gifs/"
gif_path = "/home/cephadrius/google-drive/Studies/Research/bu_research/dxl/figures/gifs/"
def make_gifs(sc):

    s.enter(600, 1, make_gifs, (sc,))

    print(f"Code execution started at (UTC):" +
          f"{datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}\n")
    file_list_dict = {}
    #file_list_dict["file_list_2hr"] = np.sort(glob.glob("/media/cephadrius/endless/bu_research/dxl/figures/historical/dscovr/2hr/sw_dsco_*.png"))[-1500::60]
#
    #file_list_dict["file_list_1day"] = np.sort(glob.glob("/media/cephadrius/endless/bu_research/dxl/figures/historical/dscovr/1day/sw_dsco_*.png"))[-1500::60]
#
    #file_list_dict["file_list_7days"] = np.sort(glob.glob("/media/cephadrius/endless/bu_research/dxl/figures/historical/dscovr/7days/sw_dsco_*.png"))[-1500::60]
#
    file_list_dict["trace"] = np.sort(glob.glob("/media/cephadrius/endless/bu_research/dxl/figures/historical/line_trace/*.png"))[-354:]

    skip_rate_list = [1, 1, 1, 1]
    for i,key in enumerate(list(file_list_dict.keys())):
        gif_name = f"{gif_path}{key}_300.gif"
        gif_maker(file_list_dict[key], gif_name, mode="I", skip_rate=skip_rate_list[i], duration=0.04)

s.enter(0, 1, make_gifs, (s,))
s.run()