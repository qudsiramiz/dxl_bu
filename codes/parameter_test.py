import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyspedas as spd
import pytplot as ptt
import geopack.geopack as gp
import datetime

# Activate the latex text rendering
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams.update({"font.size": 18})

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

    # For each parameter, do the 30 mnute rolling mean centered on the time
    for key in omni_df.keys():
        omni_df[key + "_rolling"] = omni_df[key].rolling("30T", center=True).mean()
    # For each column, interpolate the missing values using the previous and next values
    omni_df = omni_df.interpolate(method="time")
    return omni_df


start_date_1day = "2004-11-01 00:00:00"
end_date_1day = "2005-02-01 00:00:00"

omni_df = get_sw_params(
    omni_level="hro",
    time_clip=True,
    trange=[start_date_1day, end_date_1day],
    verbose=False,
)

# Find the times when the solar wind flux is greater than 3e8 1/s * cm^2 for at least 0.5 hours
flux_threshold = 2.8  # in units of 1e8 1/s * cm^2
flux_threshold_time = 30  # in minutes

# Find the times when clock angle is more than 90 degrees for at least 0.5 hours
clock_angle_threshold = 75  # in degrees
clock_angle_threshold_time = 30  # in minutes

# Create a boolean mask of the times when the flux is greater than the threshold
omni_df["above_threshold_fx"] = omni_df["flux_rolling"] >= flux_threshold

# Identify consecutive True values in the boolean mask
consecutive_periods_fx = (
    omni_df["above_threshold_fx"] != omni_df["above_threshold_fx"].shift()
).cumsum()
# Find the
# Calculate the duration of each consecutive period
period_durations_fx = omni_df.groupby(consecutive_periods_fx)[
    "above_threshold_fx"
].transform(
    "size"
)  # in minutes

# Filter for periods longer than 30 minutes and where the flux is greater than the threshold
long_periods_fx = (period_durations_fx >= flux_threshold_time) & (
    omni_df["flux_rolling"] >= flux_threshold
)


# Create a boolean mask of the times when the clock angle is greater than the threshold
omni_df["above_threshold_ca"] = omni_df["theta_c_rolling"] >= clock_angle_threshold

# Identoffy consecutive True values in the boolean mask
consecutive_periods_ca = (
    omni_df["above_threshold_ca"] != omni_df["above_threshold_ca"].shift()
).cumsum()

# Calculate the duration of each consecutive period
period_durations_ca = omni_df.groupby(consecutive_periods_ca)[
    "above_threshold_ca"
].transform(
    "size"
)  # in minutes

# Filter for periods longer than 30 minutes and where the clock angle is greater than the threshold
long_periods_ca = (period_durations_ca >= clock_angle_threshold_time) & (
    omni_df["theta_c_rolling"] >= clock_angle_threshold
)
# Count the number of times the condition was satisfied for over 30 minutes
count_long_periods_ca = long_periods_ca.sum()


long_periods_fx_ca = (
    # (period_durations_fx >= flux_threshold_time)
    (omni_df["flux"] >= flux_threshold)
    # & (period_durations_ca >= clock_angle_threshold_time)
    & (omni_df["theta_c"] >= clock_angle_threshold)
)

long_periods_ca_fx = (
    # (period_durations_ca >= clock_angle_threshold_time)
    (omni_df["theta_c"] >= clock_angle_threshold)
    # & (period_durations_fx >= flux_threshold_time)
    & (omni_df["flux"] >= flux_threshold)
)

# Plot the solar wind parameters
fig, axs = plt.subplots(6, 1, figsize=(15, 16), sharex=True)
fig.subplots_adjust(hspace=0.0, wspace=0.0)

axs[0].plot(
    omni_df.index.values,
    omni_df["bx_gsm_rolling"].values,
    color="b",
    label="Bx",
    linewidth=2,
)
axs[0].plot(
    omni_df.index.values,
    omni_df["by_gsm_rolling"].values,
    color="r",
    label="By",
    linewidth=2,
)
axs[0].plot(
    omni_df.index.values,
    omni_df["bz_gsm_rolling"].values,
    color="g",
    label="Bz",
    linewidth=2,
)
axs[0].plot(
    omni_df.index.values,
    np.sqrt(
        omni_df["bx_gsm_rolling"] ** 2
        + omni_df["by_gsm_rolling"] ** 2
        + omni_df["bz_gsm_rolling"] ** 2
    ).values,
    color="k",
    label="$|B|$",
    linewidth=2,
)
axs[0].set_ylabel("B [nT]")
# Hide the x-axis labels and ticks
axs[0].tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=False)
axs[0].legend(loc="upper right")
axs[0].grid()
# axs[0].set_title("Magnetic Field")

axs[0].set_xlim([omni_df.index.values[0], omni_df.index.values[-1]])

axs[1].plot(
    omni_df.index.values,
    omni_df["vx_rolling"].values,
    color="b",
    label="Vx",
    linewidth=2,
)
axs[1].plot(
    omni_df.index.values,
    omni_df["vy_rolling"].values,
    color="r",
    label="Vy",
    linewidth=2,
)
axs[1].plot(
    omni_df.index.values,
    omni_df["vz_rolling"].values,
    color="g",
    label="Vz",
    linewidth=2,
)
axs[1].plot(
    omni_df.index.values,
    np.sqrt(
        omni_df["vx_rolling"] ** 2
        + omni_df["vy_rolling"] ** 2
        + omni_df["vz_rolling"] ** 2
    ).values,
    color="k",
    label="$|V|$",
    linewidth=2,
)
axs[1].set_ylabel("V [km/s]")
axs[1].legend(loc="upper right")
axs[1].grid()
# Hide the x-axis labels
axs[1].tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=False)
# axs[1].set_title("Velocity")

axs[1].set_xlim([omni_df.index.values[0], omni_df.index.values[-1]])

# On the plot, print the fraction of time the clock angle was above the threshold
above_threshold_percent_ca = sum(omni_df.above_threshold_ca) * 100 / len(omni_df)
for_atleast_30_min_ca = sum(long_periods_ca) * 100 / len(omni_df)
axs[2].text(
    0.02,
    0.95,
    f"Above Threshold: {above_threshold_percent_ca:.2f}%\nFor at least 30 min: {for_atleast_30_min_ca:.2f}%",
    horizontalalignment="left",
    verticalalignment="top",
    transform=axs[2].transAxes,
    bbox=dict(facecolor="white", alpha=0.95),
)
axs[2].plot(
    omni_df.index.values,
    omni_df["theta_c_rolling"].values,
    color="b",
    label="$\\theta_c$",
    linewidth=2,
)
# Make a horizontal line at the clock_angle_threshold
axs[2].axhline(
    clock_angle_threshold,
    color="r",
    linestyle="--",
    label=f"Clock Angle Threshold ({clock_angle_threshold})",
    linewidth=5,
)
axs[2].set_ylabel("$\\theta_{\\rm c} = {\\rm tan}^{-1}(B_{\\rm y}/B_{\\rm z})$ [deg]")
# axs[2].legend(loc="upper right")
axs[2].grid()
# Hide the x-axis labels
axs[2].tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=False)
# axs[2].set_title("Clock Angle")

axs[2].set_xlim([omni_df.index.values[0], omni_df.index.values[-1]])

axs[3].plot(
    omni_df.index.values,
    omni_df["np_rolling"].values,
    color="b",
    label="Np",
    linewidth=2,
)
axs[3].set_ylabel("Np [cm$^{-3}$]")
# axs[3].legend(loc="upper right")
axs[3].grid()
# Hide the x-axis labels
axs[3].tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=False)
# axs[3].set_title("Proton Density")
axs[3].set_yscale("log")

axs[3].set_xlim([omni_df.index.values[0], omni_df.index.values[-1]])

axs[4].plot(
    omni_df.index.values,
    omni_df["t_p_rolling"].values,
    color="b",
    label="Tp",
    linewidth=2,
)
axs[4].set_ylabel("$T_{\\rm p}$ [K]")
# axs[4].legend(loc="upper right")
axs[4].grid()
# Hide the x-axis labels
axs[4].tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=False)
# axs[4].set_title("Proton Temperature")
axs[4].set_yscale("log")

axs[4].set_xlim([omni_df.index.values[0], omni_df.index.values[-1]])


axs[5].plot(omni_df.index.values, omni_df["flux_rolling"].values, "b-")
# Make a horizontal line at the flux_threshold
axs[5].axhline(
    flux_threshold,
    color="r",
    linestyle="--",
    # label=f"Flux Threshold ({flux_threshold})",
    linewidth=5,
)
axs[5].set_ylabel("Flux [$10^8$ s$^{-1}$ cm$^2$)")
# axs[5].legend(loc="upper right")
axs[5].grid()
axs[5].set_xlabel("Time")
# axs[5].set_title("Flux")
axs[5].set_yscale("log")

axs[5].set_xlim([omni_df.index.values[0], omni_df.index.values[-1]])

# On the plot, print the fraction of time the flux was above the threshold
above_threshold_percent_fx = sum(omni_df.above_threshold_fx) * 100 / len(omni_df)
for_atleast_30_min_fx = sum(long_periods_fx) * 100 / len(omni_df)
axs[5].text(
    0.02,
    0.95,
    f"Above Threshold: {above_threshold_percent_fx:.2f}%\nFor at least 30 min: {for_atleast_30_min_fx:.2f}%",
    horizontalalignment="left",
    verticalalignment="top",
    transform=axs[5].transAxes,
    bbox=dict(facecolor="white", alpha=0.95),
)

axs[5].text(
    1.00,
    0.95,
    f"Both Above Threshold: {sum(long_periods_fx_ca) * 100 / len(omni_df):.2f}%",
    horizontalalignment="right",
    verticalalignment="top",
    transform=axs[5].transAxes,
    bbox=dict(facecolor="white", alpha=0.95),
)

# plt.tight_layout()
# Save the figure
plt.savefig(
    f"../figures/omni_params_{start_date[0:10]}_{end_date[0:10]}_rolling.png",
    dpi=300,
    bbox_inches="tight",
)
