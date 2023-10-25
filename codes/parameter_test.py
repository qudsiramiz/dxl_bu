import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyspedas as spd
import pytplot as ptt
import geopack.geopack as gp
import datetime

# Define a 3 months long period centered on winter solstice of 2014.
start_date = "2014-11-01 00:00:00"
end_date = "2015-02-01 00:00:00"


def get_sw_params(omni_level="hro", time_clip=True, trange=None, verbose=False):
    r"""
    Get the solar wind parameters from the OMNI database.

    Parameters
    ----------
    omni_level : str
        The omni data level to use. Options are "hro" and "hro2". Default is "hro".
    time_clip : bool
        If True, the data will be clipped to the time range specified by trange. Default is True.
    trange : list or an array of length 2
        The time range to use. Should in the format [start, end], where start and end times should
        be a string in the format "YYYY-MM-DD HH:MM:SS".
    verbose : bool
        If True, print out a few messages and the solar wind parameters. Default is False.

    Raises
    ------
    ValueError: If the probe is not one of the options.
    ValueError: If the trange is not in the correct format.

    Returns
    -------
    sw_params : dict
        The solar wind parameters.
    """

    if trange is None:
        raise ValueError(
            "trange must be specified as a list of start and end times in the format"
            "'YYYY-MM-DD HH:MM:SS'."
        )

    # Check if trange is either a list or an array of length 2
    if not isinstance(trange, (list, np.ndarray)) or len(trange) != 2:
        raise ValueError(
            "trange must be specified as a list or array of length 2 in the format"
            + "'YYYY-MM-DD HH:MM:SS'"
        )

    # Download the OMNI data (default level of "hro_1min") for the specified timerange.
    omni_varnames = [
        "BX_GSE",
        "BY_GSM",
        "BZ_GSM",
        "proton_density",
        "Vx",
        "Vy",
        "Vz",
        "T",
    ]
    omni_vars = spd.omni.data(
        trange=trange, varnames=omni_varnames, level=omni_level, time_clip=time_clip
    )

    omni_time = ptt.get_data(omni_vars[0])[0]
    # print(f"omni_time: {omni_time}")
    omni_bx_gse = ptt.get_data(omni_vars[0])[1]
    omni_by_gsm = ptt.get_data(omni_vars[1])[1]
    omni_bz_gsm = ptt.get_data(omni_vars[2])[1]
    omni_np = ptt.get_data(omni_vars[3])[1]
    omni_vx = ptt.get_data(omni_vars[4])[1]
    omni_vy = ptt.get_data(omni_vars[5])[1]
    omni_vz = ptt.get_data(omni_vars[6])[1]
    omni_t_p = ptt.get_data(omni_vars[7])[1]

    # Convert omni_time to datetime objects from unix time
    omni_time_datetime = [datetime.datetime.utcfromtimestamp(t) for t in omni_time]
    # Get trange in datetime format
    omni_trange_time_object = [pd.to_datetime(trange[0]), pd.to_datetime(trange[1])]
    # Add utc as timezone to omni_trange_time_object
    omni_trange_time_object = [
        t.replace(tzinfo=datetime.timezone.utc) for t in omni_trange_time_object
    ]

    # Create the dataframe for OMNI data using omni_time_datetime as the index
    omni_df = pd.DataFrame(
        {
            "time": omni_time,
            "bx_gsm": omni_bx_gse,
            "by_gsm": omni_by_gsm,
            "bz_gsm": omni_bz_gsm,
            "vx": omni_vx,
            "vy": omni_vy,
            "vz": omni_vz,
            "np": omni_np,
            "t_p": omni_t_p,
        },
        index=omni_time_datetime,
    )

    # Add UTC as time zone to the index of omni_df
    omni_df.index = omni_df.index.tz_localize("UTC")

    # Get the solar wind clock angle
    omni_df["theta_c"] = np.arctan2(omni_df["by_gsm"], omni_df["bz_gsm"]) * 180 / np.pi

    # Convert "theta_c" to be in the range of 0 to 180 degrees
    omni_df["theta_c"] = np.where(
        omni_df["theta_c"] < 0, omni_df["theta_c"] + 180, omni_df["theta_c"]
    )
    # Get the solar wind flux in the units of 1/s * cm^2
    omni_df["flux"] = (
        omni_df["np"]
        * np.sqrt(omni_df["vx"] ** 2 + omni_df["vy"] ** 2 + omni_df["vz"] ** 2)
        * 1e-3
    )  # in the units of 1e8 1/s * cm^2

    # For each column, interpolate the missing values using the previous and next values
    # omni_df = omni_df.interpolate(method="time")
    return omni_df


start_date_1day = "2014-12-01 01:30:00"
end_date_1day = "2014-12-01 01:40:00"

omni_df = get_sw_params(
    omni_level="hro",
    time_clip=True,
    trange=[start_date_1day, end_date_1day],
    verbose=False,
)

# Find the times when the solar wind flux is greater than 3e8 1/s * cm^2 for at least 0.5 hours
flux_threshold = 3  # in units of 1e8 1/s * cm^2
flux_threshold_time = 0.5  # in hours

# Create a boolean mask of the times when the flux is greater than the threshold
omni_df["above_threshold_fx"] = omni_df["flux"] >= flux_threshold

# Identify consecutive True values in the boolean mask
consecutive_periods_fx = (omni_df["above_threshold_fx"] == False).cumsum()

# Calculate the duration of each consecutive period
period_durations_fx = omni_df.groupby(consecutive_periods_fx)[
    "above_threshold_fx"
].transform(
    "size"
)  # in minutes

# Filter for periods longer than 30 minutes
long_periods_fx = period_durations_fx >= flux_threshold_time * 60  # in minutes

# Count the number of times the condition was satisfied for over 30 minutes
count_long_periods_fx = long_periods_fx.sum()

# Find the times when clock angle is more than 90 degrees for at least 0.5 hours
clock_angle_threshold = 90  # in degrees
clock_angle_threshold_time = 0.5  # in hours

# Create a boolean mask of the times when the clock angle is greater than the threshold
omni_df["above_threshold_ca"] = omni_df["theta_c"] >= clock_angle_threshold

# Identoffy consecutive True values in the boolean mask
consecutive_periods_ca = (omni_df["above_threshold_ca"] == False).cumsum()

# Calculate the duration of each consecutive period
period_durations_ca = omni_df.groupby(consecutive_periods_ca)[
    "above_threshold_ca"
].transform(
    "size"
)  # in minutes

# Filter for periods longer than 30 minutes
long_periods_ca = period_durations_ca >= clock_angle_threshold_time * 60  # in minutes

# Count the number of times the condition was satisfied for over 30 minutes
count_long_periods_ca = long_periods_ca.sum()

# Plot the solar wind parameters
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
axs[0, 0].plot(
    omni_df.index.values, omni_df["bx_gsm"].values, color="b", label="Bx", linewidth=2
)
axs[0, 0].plot(
    omni_df.index.values, omni_df["by_gsm"].values, color="r", label="By", linewidth=2
)
axs[0, 0].plot(
    omni_df.index.values, omni_df["bz_gsm"].values, color="g", label="Bz", linewidth=2
)
axs[0, 0].plot(
    omni_df.index.values,
    np.sqrt(
        omni_df["bx_gsm"] ** 2 + omni_df["by_gsm"] ** 2 + omni_df["bz_gsm"] ** 2
    ).values,
    color="k",
    label="|B|",
    linewidth=2,
)
axs[0, 0].set_ylabel("B (nT)")
axs[0, 0].legend(loc="upper right")
axs[0, 0].grid()
axs[0, 0].set_title("Magnetic Field")

axs[0, 0].set_xlim([omni_df.index.values[0], omni_df.index.values[-1]])

axs[0, 1].plot(
    omni_df.index.values, omni_df["np"].values, color="b", label="Np", linewidth=2
)
axs[0, 1].set_ylabel("Np (cm$^{-3}$)")
axs[0, 1].legend(loc="upper right")
axs[0, 1].grid()
axs[0, 1].set_title("Proton Density")

axs[0, 1].set_xlim([omni_df.index.values[0], omni_df.index.values[-1]])

axs[1, 0].plot(
    omni_df.index.values, omni_df["vx"].values, color="b", label="Vx", linewidth=2
)
axs[1, 0].plot(
    omni_df.index.values, omni_df["vy"].values, color="r", label="Vy", linewidth=2
)
axs[1, 0].plot(
    omni_df.index.values, omni_df["vz"].values, color="g", label="Vz", linewidth=2
)
axs[1, 0].plot(
    omni_df.index.values,
    np.sqrt(omni_df["vx"] ** 2 + omni_df["vy"] ** 2 + omni_df["vz"] ** 2).values,
    color="k",
    label="|V|",
    linewidth=2,
)
axs[1, 0].set_ylabel("V (km/s)")
axs[1, 0].legend(loc="upper right")
axs[1, 0].grid()
axs[1, 0].set_title("Velocity")

axs[1, 0].set_xlim([omni_df.index.values[0], omni_df.index.values[-1]])

axs[1, 1].plot(
    omni_df.index.values, omni_df["t_p"].values, color="b", label="Tp", linewidth=2
)
axs[1, 1].set_ylabel("Tp (K)")
axs[1, 1].legend(loc="upper right")
axs[1, 1].grid()
axs[1, 1].set_title("Proton Temperature")

axs[1, 1].set_xlim([omni_df.index.values[0], omni_df.index.values[-1]])

# Plot a horizontal bar of the length of period_durations for each period
start_ind_ca = 0
while start_ind_ca < len(period_durations_ca):
    # Get the index of the next closest value where period_durations changes value
    try:
        next_ind_ca = np.where(
            period_durations_ca[start_ind_ca:] != period_durations_ca[start_ind_ca]
        )[0][0]
    except Exception:
        pass
    axs[2, 0].axvspan(
        period_durations_ca.index[start_ind_ca],
        period_durations_ca.index[next_ind_ca - 1],
        color="r",
        alpha=0.2,
    )
    start_ind_ca = start_ind_ca + next_ind_ca
    print(start_ind_ca)
# On the plot, print the fraction of time the clock angle was above the threshold
axs[2, 0].text(
    0.05,
    0.95,
    f"Above Threshold: {sum(omni_df.above_threshold_ca)/len(omni_df):.2f}",
    horizontalalignment="left",
    verticalalignment="top",
    transform=axs[2, 0].transAxes,
)
axs[2, 0].plot(
    omni_df.index.values,
    omni_df["theta_c"].values,
    color="b",
    label="Theta_c",
    linewidth=2,
)
# Make a horizontal line at the clock_angle_threshold
axs[2, 0].axhline(
    clock_angle_threshold,
    color="g",
    linestyle="--",
    label=f"Clock Angle Threshold ({clock_angle_threshold})",
    linewidth=2,
)
axs[2, 0].set_ylabel("Theta_c (deg)")
axs[2, 0].legend(loc="upper right")
axs[2, 0].grid()
axs[2, 0].set_title("Clock Angle")

axs[2, 0].set_xlim([omni_df.index.values[0], omni_df.index.values[-1]])

axs[2, 1].plot(omni_df.index.values, omni_df["flux"].values, "bo")
# Make a horizontal line at the flux_threshold
axs[2, 1].axhline(
    flux_threshold,
    color="r",
    linestyle="--",
    # label=f"Flux Threshold ({flux_threshold})",
    linewidth=2,
)
axs[2, 1].set_ylabel("Flux ($10^8$ 1/s * cm$^2$)")
axs[2, 1].legend(loc="upper right")
axs[2, 1].grid()
axs[2, 1].set_title("Flux")

axs[2, 1].set_xlim([omni_df.index.values[0], omni_df.index.values[-1]])

# Plot a horizontal bar of the length of period_durations for each period
start_ind_fx = 0
while start_ind_fx < len(period_durations_fx):
    # Get the index of the next closest value where period_durations chnages value
    try:
        next_ind_fx = np.where(
            period_durations_fx[start_ind_fx:] != period_durations_fx[start_ind_fx]
        )[0][0]
        axs[2, 1].axvspan(
            period_durations_fx.index[start_ind_fx],
            period_durations_fx.index[next_ind_fx - 1],
            color="r",
            alpha=0.2,
        )
        start_ind_fx = start_ind_fx + next_ind_fx

    except Exception:
        break

# On the plot, print the fraction of time the flux was above the threshold
axs[2, 1].text(
    0.05,
    0.95,
    f"Above Threshold: {sum(omni_df.above_threshold_fx)/len(omni_df):.2f}",
    horizontalalignment="left",
    verticalalignment="top",
    transform=axs[2, 1].transAxes,
)

# axs[2, 1].bar(
#     omni_df.index.values,
#     period_durations,
#     width=0.01,
#     color="r",
#     alpha=0.2,
# )
plt.tight_layout()
# Save the figure
plt.savefig("../figures/omni_params.png", dpi=300, bbox_inches="tight")
