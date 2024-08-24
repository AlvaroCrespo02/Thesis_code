import pandas as pd
import matplotlib.pyplot as plt

def plot_columns_separately(file_name, columns_to_plot):
    # Read the CSV file
    df = pd.read_csv(file_name)

    # Check if the columns exist in the DataFrame
    for column in columns_to_plot:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the CSV file")
    
    # Determine the layout of the subplots
    num_columns = len(columns_to_plot)
    num_rows = (num_columns + 1) // 2  # Adjust the number of rows

    # Create a figure with subplots
    fig, axs = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))

    # Flatten the axs array in case of a single row of subplots
    axs = axs.flatten()

    # Plot each column in a separate subplot
    for i, column in enumerate(columns_to_plot):
        axs[i].plot(df[column], label=column)
        axs[i].set_title(f'Plot of {column}')
        axs[i].set_xlabel('Index')
        axs[i].set_ylabel('Value')
        axs[i].legend()
        axs[i].grid(True)

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

# Example usage
file_name = 'audio_features_dynamic_wet.csv'
# columns_to_plot = ['Pitch (Hz)', 'Input delay (ms)']
columns_to_plot = ['Pitch (Hz)','Input delay (ms)','Spectral Centroid (Hz)','Spectral Bandwidth (Hz)','Spectral Roll-off (Hz)','RMS Energy','Spectral Flatness']

plot_columns_separately(file_name, columns_to_plot)



