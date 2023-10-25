import pandas as pd
import numpy as np

# Sample DataFrame with a datetime index and a column named "flux" and frequency of 1 minute
data = {
    "datetime": pd.date_range("2021-01-01", periods=100, freq="1T"),
    "flux": np.random.randint(0, 100, 100),
}
df = pd.DataFrame(data)
df.set_index("datetime", inplace=True)

# Set your flux_threshold
flux_threshold = 30

# Create a boolean mask to check if flux > flux_threshold
df["above_threshold"] = df["flux"] > flux_threshold

# Identify consecutive periods where the condition is True
consecutive_periods = (df["above_threshold"] != df["above_threshold"].shift()).cumsum()

# Calculate the duration of each consecutive period
period_durations = df.groupby(consecutive_periods)["above_threshold"].transform(
    "size"
)  # Assuming 6T is 6 minutes

# Filter for periods longer than 30 minutes
long_periods = period_durations >= 10

# Count the number of times the condition was satisfied for over 30 minutes
count_long_periods = long_periods.sum()

print(
    f"Number of times 'flux' was above {flux_threshold} for at least 30 minutes: {count_long_periods}"
)
