import os
import librosa
import soundfile as sf
from tqdm import tqdm
import pandas as pd

def get_audio_files(path):
    """Returns a list of paths to audio files in the given directory."""
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = [os.path.join(path, f) for f in os.listdir(path) if os.path.splitext(f)[1] in audio_extensions]
    return audio_files

def calculate_segments(file_path, segment_length_ms):
    """Returns the number of segments of the specified length (in ms) that can be extracted from the given audio file."""
    y, sr = librosa.load(file_path, sr=None)
    segment_length_samples = int((segment_length_ms / 1000) * sr)
    num_segments = len(y) // segment_length_samples
    return num_segments, y, sr, segment_length_samples

def save_segments(y, sr, segment_length_samples, num_segments, output_dir, segment_names, start_index):
    """Saves the segments of the audio file to the specified output directory with specified naming convention."""
    count = 0
    for i in range(num_segments):
        start_sample = i * segment_length_samples
        end_sample = start_sample + segment_length_samples
        segment = y[start_sample:end_sample]

        if start_index + count >= len(segment_names):
            print(f"Warning: Not enough names in the CSV file to name all segments. Skipping remaining segments.")
            break
        
        segment_filename = os.path.join(output_dir, f'{segment_names[start_index + count]}.wav')
        sf.write(segment_filename, segment, sr)
        
        count += 1

    return start_index + count

def main(input_directory, segment_length_ms, output_directory, csv_file):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    segment_names_df = pd.read_csv(csv_file)
    segment_names = segment_names_df['Name'].tolist()
    
    audio_files = get_audio_files(input_directory)
    name_index = 0
    
    for file in tqdm(audio_files, desc="Processing audio files"):
        num_segments, y, sr, segment_length_samples = calculate_segments(file, segment_length_ms)
        name_index = save_segments(y, sr, segment_length_samples, num_segments, output_directory, segment_names, name_index)
        print(f'File: {file}, Number of {segment_length_ms} ms segments: {num_segments}')

# Example usage
input_directory = 'input/path' # Input path for the files
segment_length_ms = 3000  # Duration in ms for each segment
output_directory = 'output/path' # Output path for the new files
csv_file = 'file_info.csv'  # Path to the CSV file containing the segment names
main(input_directory, segment_length_ms, output_directory, csv_file)
