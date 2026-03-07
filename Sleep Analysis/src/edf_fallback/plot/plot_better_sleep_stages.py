import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D


def plot_sleep_stages(epoch_times, sleep_stages_smoothed):
    stage_info = {
        1: {"label": "Wake", "color": "#f4a261"},
        2: {"label": "Movement", "color": "#e76f51"},
        3: {"label": "Transitional", "color": "#e9c46a"},
        4: {"label": "NREM", "color": "#2a9d8f"},
    }

    fig, ax = plt.subplots(figsize=(15, 6))

    # Background color bands for each stage
    for stage_value, info in stage_info.items():
        ax.axhspan(
            stage_value - 0.5,
            stage_value + 0.5,
            color=info["color"],
            alpha=0.12
        )

    # Draw each segment in the color of its stage
    for i in range(len(epoch_times) - 1):
        stage = sleep_stages_smoothed[i]
        color = stage_info.get(stage, {}).get("color", "blue")

        ax.step(
            epoch_times[i:i+2],
            sleep_stages_smoothed[i:i+2],
            where="post",
            color=color,
            linewidth=2.5
        )

    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels([
        stage_info[1]["label"],
        stage_info[2]["label"],
        stage_info[3]["label"],
        stage_info[4]["label"],
    ])

    ax.set_xlabel("Time")
    ax.set_ylabel("Sleep Stage")
    ax.set_title("Low Risk 02 - Sleep Profile Analysis", fontweight="bold")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=info["color"], lw=3, label=info["label"])
        for info in stage_info.values()
    ]
    ax.legend(handles=legend_elements, title="Stages", loc="upper right")

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()