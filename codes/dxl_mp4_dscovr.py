import glob as glob
import numpy as np
import imageio as iio
from pygifsicle import optimize
import sched
import datetime
import time

s = sched.scheduler(time.time, time.sleep)

def gif_maker(file_list, vid_name, mode="I", skip_rate=10, vid_type="mp4", duration=0.05, fps=25):
    """
    Make a gif from a list of images.
    
    Parameters
    ----------
    file_list : list
        List of image files.
    vid_name : str
        Name of the gif file.
    mode : str, optional
        Mode of the gif. The default is "I".
    skip_rate : int, optional
        Skip rate of the gif. The default is 10.
    vid_type : str, optional
        Type of the video. The default is "mp4".
    duration : float, optional
        Duration for which each image is displayed in gif. The default is 0.05.
    fps : int, optional
        Frames per second for mp4 video. The default is 25.

    Raises
    ------
    ValueError
        If the skip_rate is not an integer.
    ValueError
        If the duration is not a float.
    ValueError
        If the file_list is empty.
    ValueError
        If vid_name is empty.

    Returns
    -------
    None.
    """
    if file_list is None:
        raise ValueError("file_list is None")
    if vid_name is None:
        raise ValueError("vid_name is None. Please provide the name of the gif/video")
    if len(file_list) == 0:
        raise ValueError("file_list is empty")
    #if len(file_list) >= 1501:
    #    # Check if the skip_rate is an integer
    #    if skip_rate != int(skip_rate):
    #        raise ValueError("skip_rate must be an integer")
    #    file_list = file_list[-1500::skip_rate]
    if vid_type == "gif":
        if duration != float(duration):
            raise ValueError("duration must be a float")
    if vid_type == "mp4":
        if fps != int(fps):
            raise ValueError("Frame rate (fps) must be an integer")

    count = 0
    if vid_type == "gif":
        with iio.get_writer(vid_name, mode=mode, duration=duration) as writer:
            for file in file_list:
                count += 1
                print(f"Processing image {count} of {len(file_list)}")
                image = iio.imread(file)
                writer.append_data(image)
    elif vid_type == "mp4":
        with iio.get_writer(vid_name, mode=mode, fps=fps) as writer:
            for filename in file_list:
                count += 1
                print(f"Processing image {count} of {len(file_list)}")
                img = iio.imread(filename)
                writer.append_data(img)
    writer.close()
    #with iio.get_writer(vid_name, mode=mode, fps=fps) as writer:
    #    for fl in file_list:
    #        image = iio.imread(fl)
    #        writer.append_data(image)
    #        count += 1
    #        print(f"{count} images added to the video file")
    
    #optimize(vid_name)

    print(f"{vid_name} is created\n")


def make_gifs(number_of_days=30):
#for xxx in range(1, 2):
    number_of_days = (number_of_days - 30) * 4
    #s.enter(6000, 1, make_gifs, (sc,))

    vid_type = "mp4"  # "gif" or "mp4"
    if vid_type == "gif":
        #gif_path = "/home/cephadrius/Dropbox/DXL-Figure/gifs/"
        #gif_path = "/home/cephadrius/google-drive/Studies/Research/bu_research/dxl/figures/gifs/"
        gif_path = "/media/cephadrius/endless/bu_research/dxl/figures/gifs/"
    elif vid_type == "mp4":
        #gif_path = "/home/cephadrius/Dropbox/DXL-Figure/vids/"
        #gif_path = "/home/cephadrius/google-drive/Studies/Research/bu_research/dxl/figures/vids/"
        gif_path = "/home/cephadrius/Desktop/git/qudsiramiz.github.io/images/moving_pictures/"
        print(f"Code execution started at (UTC):" +
              f"{datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}\n")

    file_list_dict = {}
    #file_list_dict["file_list_2hr"] = np.sort(glob.glob("/media/cephadrius/endless/bu_research/dxl/figures/historical/dscovr/2hr/sw_dsco_*.png"))[-1500::60]

    #file_list_dict["file_list_1day"] = np.sort(glob.glob("/media/cephadrius/endless/bu_research/dxl/figures/historical/dscovr/1day/sw_dsco_*.png"))[-4500::20]

    #file_list_dict["file_list_7days"] = np.sort(glob.glob("/media/cephadrius/endless/bu_research/dxl/figures/historical/dscovr/7days/sw_dsco_*.png"))[-2000::1]

    #file_list_dict["trace"] = np.sort(glob.glob("/media/cephadrius/endless/bu_research/dxl/figures/historical/line_trace/*.png"))[-4000:]

    file_list_dict["file_list_30days"] = np.sort(glob.glob("/media/cephadrius/endless/bu_research/dxl/figures/historical/dscovr/30days/sw_dsco*.png"))[-number_of_days:]

    skip_rate_list = [1, 1, 1, 1]
    for i,key in enumerate(list(file_list_dict.keys())):
        #vid_name = f"{gif_path}{key}.{vid_type}"
        vid_name = f"{gif_path}DSCOVR_30days_hourly_averaged.mp4"
        try:
            gif_maker(file_list_dict[key], vid_name, mode="I", skip_rate=skip_rate_list[i], vid_type=vid_type, fps=25, duration=0.05)
        except ValueError as e:
            print(e)
            pass

#s.enter(0, 1, make_gifs, (s,))
#s.run()