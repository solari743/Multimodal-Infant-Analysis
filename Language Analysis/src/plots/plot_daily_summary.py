import os
import matplotlib.pyplot as plt

def plot_daily_summary(daily_summary, output_dir):
    if daily_summary is None or daily_summary.empty:
        print("No daily summary data to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots()
    daily_summary.set_index('date').plot(
        kind='bar', y=['AWC', 'CVC', 'CTC'], ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1']
    )

    ax.set_title('Total AWC, CVC, and CTC per Day', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Total Count', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(['Adult Word Count (AWC)', 'Child Vocalization Count (CVC)', 'Conversational Turn Count (CTC)'])

    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', label_type='edge', fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "daily_summary_AWC_CVC_CTC.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved daily summary plot to: {output_path}")
