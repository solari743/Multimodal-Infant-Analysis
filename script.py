# ===============================================================
# CONTINUOUS + COMPRESSED SLEEP COV PLOTS
# ===============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime, re, h5py, numpy as np
from scipy.signal import butter, filtfilt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D



# Low Risk Data

filename  = r'Low Risk 02.h5'
targetID  = '16162'
sleep_file = r'Sleep profile [EDF BASED].txt'

# High Risk Data

# filename  = r'Patient-Data/DATASET - HIGH RISK 02/High_Risk02 [RAW MOVEMENT]/highrisk02_20240820-141937.h5'
# targetID  = 'XI-016162'
# sleep_file = r'Patient-Data/DATASET - HIGH RISK 02/High_Risk02 [CHILD TEMPLATE]/Sleep profile - HighRisk.txt'



# ===============================================================
# 2. PARSE SLEEP START TIME
# ===============================================================
file_date = None
start_clock_time = None

with open(sleep_file, 'r', errors='ignore') as f:
    for line in f:
        if "Start Time" in line:
            match = re.search(
                r'(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})\s*(AM|PM)',
                line
            )
            if match:
                month, day, year, hour, minute, second, meridiem = match.groups()
                hour = int(hour)
                if meridiem.upper() == 'PM' and hour != 12:
                    hour += 12
                if meridiem.upper() == 'AM' and hour == 12:
                    hour = 0
                file_date = datetime.date(int(year), int(month), int(day))
                start_clock_time = datetime.time(hour, int(minute), int(second))
                break

if file_date is None:
    name_match = re.search(r'(\d{8})', sleep_file)
    if name_match:
        file_date = datetime.datetime.strptime(name_match.group(1), '%Y%m%d').date()

if file_date is None:
    raise ValueError("Recording date not found.")

sleep_start_dt = datetime.datetime.combine(
    file_date,
    start_clock_time or datetime.time(0, 0, 0)
)

# ===============================================================
# 3. LOAD SLEEP STATES
# ===============================================================
sleep_data = []

with open(sleep_file, 'r', errors='ignore') as f:
    for line in f:
        if ';' not in line:
            continue
        time_part, state = line.strip().split(';')
        try:
            time_obj = datetime.datetime.strptime(
                time_part.strip(), '%H:%M:%S,%f'
            ).time()
            full_time = datetime.datetime.combine(file_date, time_obj)
            sleep_data.append((full_time, state.strip()))
        except ValueError:
            continue

sleep_df = (
    pd.DataFrame(sleep_data, columns=['timestamp','state'])
      .set_index('timestamp')
      .sort_index()
)

if sleep_df.empty:
    raise ValueError("No sleep data found.")

sleep_df['state_norm'] = sleep_df['state'].str.lower().str.strip()


# ===============================================================
# 4. LOAD & CLEAN ACCELEROMETER DATA
# ===============================================================
with h5py.File(filename,'r') as f:
    base = f"Sensors/{targetID}"
    acc_data = np.array(f[f'{base}/Accelerometer'][:], dtype=np.float64)
    time_raw = np.array(f[f'{base}/Time'][:], dtype=np.float64)
    time_dt  = np.array([
        datetime.datetime.fromtimestamp(t * 1e-6)
        for t in time_raw
    ])

acc_df = pd.DataFrame(
    acc_data,
    columns=['ax','ay','az'],
    index=pd.to_datetime(time_dt)
)

acc_df['mag'] = np.sqrt(
    acc_df['ax']**2 +
    acc_df['ay']**2 +
    acc_df['az']**2
)

acc_df['mag'] = acc_df['mag'].clip(
    lower=acc_df['mag'].quantile(0.01),
    upper=acc_df['mag'].quantile(0.99)
)

acc_df['mag'] -= acc_df['mag'].rolling(
    window=250, center=True, min_periods=1
).mean()

def butter_lowpass_filter(data, cutoff=3, fs=50, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)

acc_df['mag'] = butter_lowpass_filter(
    acc_df['mag'].interpolate(),
    cutoff=3,
    fs=50
)

acc_df.dropna(inplace=True)


# ===============================================================
# 5. COMPUTE 30s CoV
# ===============================================================
def cov_func(x):
    mean_x = x.mean()
    return np.nan if mean_x == 0 else x.std() / mean_x

cov_30s = (
    acc_df['mag']
    .resample('30s')
    .apply(cov_func)
    .dropna()
)


# ===============================================================
# 6. STATE DEFINITIONS
# ===============================================================

# States shown in FIRST (continuous) plot ONLY
plot1_states = [
    'wake',
    'nrem',
    'rem',
    'transitional',
    'movement',
    'a'          # abstract → artifact-like, kept separate
]

plot1_colors = {
    'wake': '#ff8c00',          # orange
    'nrem': '#1f77b4',          # blue
    'rem': '#2ca02c',           # green
    'transitional': '#9467bd',  # purple
    'movement': '#8c564b',      # brown
    'a': '#7f7f7f'              # gray (abstract)
}

plot1_df = sleep_df[sleep_df['state_norm'].isin(plot1_states)]


# States used for SECOND (compressed) plot — UNCHANGED
sleep_states = ['nrem', 'rem', 'transitional']
sleep_colors = {
    'nrem':'#1f77b4',
    'rem':'#2ca02c',
    'transitional':'#9467bd'
}

sleep_only_df = sleep_df[sleep_df['state_norm'].isin(sleep_states)]

print(
    sleep_df['state']
    .str.lower()
    .value_counts()
)

print("\n=== START TIME CHECK ===")
print("Accel start:", acc_df.index.min())
print("Accel end:  ", acc_df.index.max())
print("CoV start:  ", cov_30s.index.min())
print("CoV end:    ", cov_30s.index.max())
print("Sleep start:", sleep_df.index.min())
print("Sleep end:  ", sleep_df.index.max())

offset = sleep_df.index.min() - cov_30s.index.min()
print("Sleep start - CoV start offset:", offset)

# ===============================================================
# 7. FIRST PLOT — FULL TIMELINE (WITH WAKE) — INTERVAL-BASED SPANS
# ===============================================================
fig, ax = plt.subplots(figsize=(15, 7))

# Plot CoV
ax.plot(
    cov_30s.index, cov_30s.values,
    color='magenta', lw=1.5,
    solid_capstyle="butt", solid_joinstyle="round"
)

# --- infer epoch width (should be ~30s); fallback to 30s safely
epoch = cov_30s.index.to_series().diff().median()
if pd.isna(epoch) or epoch <= pd.Timedelta(0):
    epoch = pd.Timedelta(seconds=30)

# Optional: use half-epoch padding so boundaries look clean and symmetric
half = epoch / 2

for state, color in plot1_colors.items():
    state_df = plot1_df[plot1_df['state_norm'] == state].copy()

    # SAME grouping logic you had (detect breaks)
    state_df['gap'] = (
        state_df.index.to_series().diff() > pd.Timedelta('40s')
    ).cumsum()

    for _, blk in state_df.groupby('gap'):
        # INTERVAL-BASED edges:
        # Each timestamp t represents an interval [t - half, t + half) (centered)
        # so the block covers from first_left to last_right.
        left  = blk.index.min() - half
        right = blk.index.max() + half  # includes the full final epoch

        ax.axvspan(
            left,
            right,
            color=color,
            alpha=0.25,
            linewidth=0  # prevents hairline seams
        )

ax.margins(x=0)

# Legend: patches for states + line for CoV
legend1 = (
    [Patch(facecolor=color, edgecolor="none", alpha=0.25, label=state.upper())
     for state, color in plot1_colors.items()] +
    [Line2D([0], [0], color='magenta', lw=1.5, label='CoV')]
)
ax.legend(handles=legend1, loc='upper right')

ax.set_title("Continuous Movement Variability (CoV) Across Sleep and Wake - Low Risk 02")
ax.set_xlabel("Time")
ax.set_ylabel("Coefficient of Variation")
ax.grid(True)

ax.set_xlim(cov_30s.index.min(), cov_30s.index.max())
ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

plt.tight_layout()
plt.show()


# # ===============================================================
# # 7. FIRST PLOT — FULL TIMELINE (WITH WAKE)
# # ===============================================================
# fig, ax = plt.subplots(figsize=(15,7))
#
# # CoV line — also help avoid tiny rendering seams
# ax.plot(
#     cov_30s.index, cov_30s.values,
#     color='magenta', lw=1.5,
#     solid_capstyle="butt", solid_joinstyle="round"
# )
#
# # infer your sampling interval (e.g., 30s) from plot1_df's index
# dt = plot1_df.index.to_series().diff().median()
# if pd.isna(dt) or dt <= pd.Timedelta(0):
#     dt = pd.Timedelta(seconds=30)  # safe fallback
#
# half = dt / 2  # optional: makes boundaries look nicer on transitions
#
# for state, color in plot1_colors.items():
#     state_df = plot1_df[plot1_df['state_norm'] == state].copy()
#     state_df['gap'] = (state_df.index.to_series().diff() > pd.Timedelta('40s')).cumsum()
#
#     for _, blk in state_df.groupby('gap'):
#         start = blk.index.min() - half
#         end   = blk.index.max() + half + dt  # <-- key: cover full last epoch
#         ax.axvspan(start, end, color=color, alpha=0.25, linewidth=0)
#
# # remove x padding so labels reach edges cleanly
# ax.margins(x=0)
#
# ax.set_xlim(cov_30s.index.min(), cov_30s.index.max())
# ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
#
# ax.set_title("Continuous Movement Variability (CoV) Across Sleep and Wake - Low Risk 02")
# ax.set_xlabel("Time")
# ax.set_ylabel("Coefficient of Variation")
# ax.grid(True)
#
# legend1 = (
#     [Patch(facecolor=color, edgecolor='none', alpha=0.25, label=state.upper())
#      for state, color in plot1_colors.items()] +
#     [Line2D([0], [0], color='magenta', lw=1.5, label='CoV')]
# )
#
# ax.legend(handles=legend1, loc='upper right', frameon=True)
#
# plt.tight_layout()
# plt.show()

# ===============================================================
# 8. SECOND PLOT — COMPRESSED SLEEP-ONLY
# ===============================================================
fig2, ax2 = plt.subplots(figsize=(15,7))

sleep_labels = sleep_only_df['state'].reindex(
    cov_30s.index,
    method='nearest',
    tolerance=pd.Timedelta('40s')
)

sleep_mask = sleep_labels.str.lower().isin(sleep_states)
sleep_cov = cov_30s[sleep_mask].copy()
sleep_state_for_cov = sleep_labels[sleep_mask].str.lower()

df_blocks = pd.DataFrame({
    'state': sleep_state_for_cov,
    'original_time': sleep_cov.index
})

df_blocks['time_diff'] = df_blocks['original_time'].diff()
df_blocks['block_id'] = (
    (df_blocks['state'] != df_blocks['state'].shift()) |
    (df_blocks['time_diff'] > pd.Timedelta('40s'))
).cumsum()

time_mapping = {}
current_time = pd.Timedelta(0)

for block_id in sorted(df_blocks['block_id'].unique()):
    group = df_blocks[df_blocks['block_id'] == block_id]
    block_start = group['original_time'].min()
    block_duration = (
        group['original_time'].max() - block_start +
        pd.Timedelta('30s')
    )

    for t in group['original_time']:
        time_mapping[t] = current_time + (t - block_start)

    current_time += block_duration

compressed_times = [time_mapping[t] for t in sleep_cov.index]
sleep_cov_compressed = sleep_cov.copy()
sleep_cov_compressed.index = (
    pd.to_datetime(sleep_start_dt) +
    pd.to_timedelta(compressed_times)
)

ax2.plot(
    sleep_cov_compressed.index,
    sleep_cov_compressed.values,
    color='magenta',
    lw=1.5
)

for state in sleep_states:
    mask = sleep_state_for_cov == state
    if not mask.any():
        continue
    idx = sleep_cov_compressed.index[mask]
    ax2.axvspan(
        idx.min(),
        idx.max(),
        color=sleep_colors[state],
        alpha=0.25
    )

legend2 = (
    [Patch(facecolor=color, alpha=0.25, label=state.upper())
     for state, color in sleep_colors.items()] +
    [Patch(facecolor='magenta', alpha=0.5, label='CoV')]
)

ax2.legend(handles=legend2, loc='upper right')
ax2.set_title("REM Sleep Movement CoV Aligned to Pseudo-Timeline — Low Risk 02")
ax2.set_xlabel("Compressed Sleep Time")
ax2.set_ylabel("Coefficient of Variation")
ax2.grid(True)

ax2.set_xlim(
    sleep_cov_compressed.index.min(),
    sleep_cov_compressed.index.max()
)
ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

plt.tight_layout()
plt.show()


# ===============================================================
# 9. ANALYTICS
# ===============================================================
print("\n========== ADVANCED CoV + SLEEP ANALYTICS ==========\n")

print("CoV STATISTICS:")
print(f"Mean:     {cov_30s.mean():.4f}")
print(f"Median:   {cov_30s.median():.4f}")
print(f"Std Dev:  {cov_30s.std():.4f}")
print(f"Variance: {cov_30s.var():.4f}")
print(f"Min:      {cov_30s.min():.4f}")
print(f"Max:      {cov_30s.max():.4f}")
print(f"P05:      {cov_30s.quantile(0.05):.4f}")
print(f"P25:      {cov_30s.quantile(0.25):.4f}")
print(f"P75:      {cov_30s.quantile(0.75):.4f}")
print(f"P95:      {cov_30s.quantile(0.95):.4f}")

sleep_start = sleep_only_df.index.min()
sleep_end   = sleep_only_df.index.max()
total_sleep_seconds = (sleep_end - sleep_start).total_seconds()
total_record_seconds = (cov_30s.index[-1] - cov_30s.index[0]).total_seconds()

print("\nOTHER METRICS:")
print(f"Total Sleep Window:   {total_sleep_seconds/3600:.2f} hrs")
print(f"Total Recording Time: {total_record_seconds/3600:.2f} hrs")
print(f"Sleep Efficiency:     {(total_sleep_seconds/total_record_seconds)*100:.1f}%")

# ---- State-aligned CoV summary (30s epochs labeled by nearest sleep state)
state_labels = sleep_df['state_norm'].reindex(
    cov_30s.index,
    method='nearest',
    tolerance=pd.Timedelta('40s')
)
state_cov = pd.DataFrame({
    'cov': cov_30s,
    'state': state_labels
}).dropna(subset=['state'])

print("\nSTATE-WISE CoV SUMMARY (30s epochs):")
if state_cov.empty:
    print("No matched sleep-state labels within tolerance.")
else:
    for st in plot1_states:
        st_cov = state_cov.loc[state_cov['state'] == st, 'cov']
        if st_cov.empty:
            continue
        hours = (len(st_cov) * 30) / 3600.0
        print(
            f"{st.upper():>12} | epochs: {len(st_cov):5d} | hrs: {hours:5.2f} | "
            f"mean: {st_cov.mean():.4f} | median: {st_cov.median():.4f} | "
            f"P95: {st_cov.quantile(0.95):.4f}"
        )

# ---- Movement bursts: high-CoV episodes above 95th percentile
hi_thr = cov_30s.quantile(0.95)
hi_mask = cov_30s > hi_thr
hi_blocks = (
    hi_mask.astype(int)
    .diff()
    .fillna(0)
    .ne(0)
    .cumsum()
)
hi_groups = cov_30s[hi_mask].groupby(hi_blocks[hi_mask])

print("\nHIGH-CoV BURSTS (>{:.4f}):".format(hi_thr))
if hi_mask.any():
    burst_durations = []
    burst_peaks = []
    for _, g in hi_groups:
        duration_s = len(g) * 30
        burst_durations.append(duration_s)
        burst_peaks.append(g.max())
    total_burst_s = sum(burst_durations)
    print(f"Bursts:           {len(burst_durations)}")
    print(f"Total time:       {total_burst_s/60:.1f} min")
    print(f"Mean duration:    {np.mean(burst_durations):.1f} s")
    print(f"Max duration:     {np.max(burst_durations):.1f} s")
    print(f"Peak CoV (max):   {np.max(burst_peaks):.4f}")
else:
    print("No bursts detected above threshold.")
print("\n=====================================================\n")


# ===============================================================
# 10. ADDITIONAL ANALYSIS MODULES
# ===============================================================

# ---- Activity counts based on CoV thresholds
cov_low = cov_30s.quantile(0.25)
cov_high = cov_30s.quantile(0.75)
activity_class = pd.cut(
    cov_30s,
    bins=[-float('inf'), cov_low, cov_high, float('inf')],
    labels=['low', 'medium', 'high']
)
activity_counts = activity_class.value_counts().reindex(['low','medium','high']).fillna(0).astype(int)
activity_hours = activity_counts * 30 / 3600.0

print("\nACTIVITY COUNTS (CoV-based):")
for label in ['low','medium','high']:
    print(f"{label.upper():>6} | epochs: {activity_counts[label]:5d} | hrs: {activity_hours[label]:5.2f}")

# ---- Per-state CoV distributions (summary + plot)
print("\nPER-STATE CoV DISTRIBUTIONS:")
state_cov_dist = state_cov.copy()
if state_cov_dist.empty:
    print("No state-aligned data for distributions.")
else:
    for st in plot1_states:
        st_cov = state_cov_dist.loc[state_cov_dist['state'] == st, 'cov']
        if st_cov.empty:
            continue
        print(
            f"{st.upper():>12} | mean: {st_cov.mean():.4f} | std: {st_cov.std():.4f} | "
            f"P25: {st_cov.quantile(0.25):.4f} | P50: {st_cov.median():.4f} | P75: {st_cov.quantile(0.75):.4f}"
        )

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    data = [state_cov_dist.loc[state_cov_dist['state'] == st, 'cov'] for st in plot1_states if not state_cov_dist.loc[state_cov_dist['state'] == st, 'cov'].empty]
    labels = [st.upper() for st in plot1_states if not state_cov_dist.loc[state_cov_dist['state'] == st, 'cov'].empty]
    if data:
        ax3.boxplot(data, tick_labels=labels, showfliers=False)
        ax3.set_title('CoV Distribution by Sleep State')
        ax3.set_ylabel('Coefficient of Variation')
        ax3.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

# ---- Transition analysis: CoV around state changes
print("\nTRANSITION-ALIGNED CoV (±5 min around state changes):")
transition_window = pd.Timedelta('5min')
transitions = sleep_df['state_norm'].ne(sleep_df['state_norm'].shift())
transition_times = sleep_df.index[transitions]
if len(transition_times) == 0:
    print("No transitions detected.")
else:
    aligned_vals = []
    for t in transition_times:
        window = cov_30s.loc[(cov_30s.index >= t - transition_window) & (cov_30s.index <= t + transition_window)]
        if not window.empty:
            aligned_vals.append(window.values)
    if aligned_vals:
        concat = np.concatenate(aligned_vals)
        print(f"Transitions: {len(transition_times)} | Samples: {len(concat)} | Mean CoV: {np.mean(concat):.4f} | P95: {np.percentile(concat, 95):.4f}")
    else:
        print("No CoV samples found in transition windows.")

# ---- Circadian profile: hourly CoV averages
print("\nCIRCADIAN PROFILE (hourly mean CoV):")
hourly_cov = cov_30s.groupby(cov_30s.index.hour).mean()
for hour, val in hourly_cov.items():
    print(f"{hour:02d}:00 | mean CoV: {val:.4f}")

fig4, ax4 = plt.subplots(figsize=(10, 4))
ax4.plot(hourly_cov.index, hourly_cov.values, marker='o')
ax4.set_title('Hourly Mean CoV (Circadian Profile)')
ax4.set_xlabel('Hour of Day')
ax4.set_ylabel('Mean CoV')
ax4.grid(True)
plt.tight_layout()
plt.show()

# ---- Spectral features: dominant frequency of motion magnitude
print("\nSPECTRAL FEATURES (dominant frequency per 5-min window):")
fs = 50  # sampling rate (Hz)
win = int(fs * 300)  # 5 minutes
mag = acc_df['mag'].values
if len(mag) >= win:
    dom_freqs = []
    for i in range(0, len(mag) - win + 1, win):
        seg = mag[i:i+win]
        seg = seg - np.mean(seg)
        freqs = np.fft.rfftfreq(len(seg), d=1/fs)
        psd = np.abs(np.fft.rfft(seg))**2
        dom = freqs[np.argmax(psd[1:]) + 1] if len(psd) > 1 else 0
        dom_freqs.append(dom)
    print(f"Windows: {len(dom_freqs)} | Mean dom freq: {np.mean(dom_freqs):.2f} Hz | P95: {np.percentile(dom_freqs,95):.2f} Hz")
else:
    print("Not enough data for 5-min spectral windows.")

# ---- Autocorrelation: periodicity estimate
print("\nAUTOCORRELATION (first 5 min of CoV):")
max_lag = int(5 * 60 / 30)  # 5 minutes in 30s epochs
if len(cov_30s) > max_lag:
    x = cov_30s.values - np.mean(cov_30s.values)
    ac = np.correlate(x, x, mode='full')[len(x)-1:len(x)+max_lag]
    ac /= ac[0] if ac[0] != 0 else 1
    peak_lag = np.argmax(ac[1:]) + 1
    print(f"Peak lag: {peak_lag*30} s | Peak corr: {ac[peak_lag]:.3f}")
else:
    print("Not enough data for autocorrelation window.")
