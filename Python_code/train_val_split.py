import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def split_and_save_audio_files(parent_path, dry_folder, wet_folder, num_files, percentage_train, percentage_validation, seed):
    """
    Split audio files from dry and wet folders into training and validation sets and save them into new folders.

    :param parent_path: Path to the parent directory where new folders will be created.
    :param dry_folder: Path to the dry signals folder.
    :param wet_folder: Path to the wet signals folder.
    :param num_files: Number of audio files to use.
    :param percentage_train: Percentage of the dataset to use for training (between 0 and 100).
    :param percentage_validation: Percentage of the dataset to use for validation (between 0 and 100).
    :param seed: Random seed for reproducibility.
    :return: None
    """
    # Validate input percentages
    if percentage_train + percentage_validation != 100:
        raise ValueError("The sum of training and validation percentages must be 100.")
    
    # Get list of files from both folders
    dry_files = [os.path.join(dry_folder, f) for f in os.listdir(dry_folder) if f.endswith('.wav')]
    wet_files = [os.path.join(wet_folder, f) for f in os.listdir(wet_folder) if f.endswith('.wav')]
    
    # Ensure both folders have the same number of files and enough files
    if len(dry_files) != len(wet_files):
        raise ValueError("The number of files in dry and wet folders must be the same.")
    if len(dry_files) < num_files:
        raise ValueError(f"Not enough files in the folders. Required: {num_files}, Available: {len(dry_files)}")
    
    # Select the required number of files
    np.random.seed(seed)
    selected_indices = np.random.choice(len(dry_files), num_files, replace=False)
    selected_dry_files = [dry_files[i] for i in selected_indices]
    selected_wet_files = [wet_files[i] for i in selected_indices]
    
    # Convert percentages to proportions
    train_size = percentage_train / 100
    
    # Split the files into training and validation sets
    train_dry_files, val_dry_files = train_test_split(selected_dry_files, train_size=train_size, random_state=seed)
    train_wet_files, val_wet_files = train_test_split(selected_wet_files, train_size=train_size, random_state=seed)
    
    # Define output directories
    train_dry_dir = os.path.join(parent_path, 'train_dry')
    val_dry_dir = os.path.join(parent_path, 'val_dry')
    train_wet_dir = os.path.join(parent_path, 'train_wet')
    val_wet_dir = os.path.join(parent_path, 'val_wet')
    
    # Create directories if they don't exist
    os.makedirs(train_dry_dir, exist_ok=True)
    os.makedirs(val_dry_dir, exist_ok=True)
    os.makedirs(train_wet_dir, exist_ok=True)
    os.makedirs(val_wet_dir, exist_ok=True)
    
    # Copy files to the respective directories
    for file in train_dry_files:
        shutil.copy(file, train_dry_dir)
    for file in val_dry_files:
        shutil.copy(file, val_dry_dir)
    for file in train_wet_files:
        shutil.copy(file, train_wet_dir)
    for file in val_wet_files:
        shutil.copy(file, val_wet_dir)
    
    print("Files have been split and copied to the respective directories.")

# Example usage:
parent_path = 'parent/path'
dry_folder = 'dry/folder/path'
wet_folder = 'wet/folder/path'
num_files = 256  # Number of audio files to use
percentage_train = 70  # 70% training
percentage_validation = 30  # 30% validation
seed = 42

split_and_save_audio_files(parent_path, dry_folder, wet_folder, num_files, percentage_train, percentage_validation, seed)

