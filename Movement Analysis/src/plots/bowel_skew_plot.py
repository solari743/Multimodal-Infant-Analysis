import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_bowel_skew(epoch_features, output_plot_name):
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.figure(figsize=(15, 7))

    plt.plot(
        epoch_features.index, 
        epoch_features['bowley_skew'], 
        label='Bowley Skewness', 
        color='purple', 
        marker='.', 
        linestyle='-'
        )

    plt.axhline(
        y=0, 
        color='black', 
        linestyle='--', 
        linewidth=1, 
        label='Symmetric (Skew = 0)'
        )

    plt.title('Bowley-Galton Skewness Over Time', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Skewness Coefficient', fontsize=12)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    plt.legend()
    plt.tight_layout()

    output_plot_name = f"{os.path.splitext(filename)[0]}_bowley_skew_plot.png"
    
    plt.savefig(output_plot_name)
    
    print(f"Successfully generated and saved {output_plot_name}")
