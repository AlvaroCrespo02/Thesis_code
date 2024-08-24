import os
import librosa
import soundfile as sf
from tqdm import tqdm

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

def save_segments(y, sr, segment_length_samples, num_segments, output_dir, base_filename, note_list):
    """Saves the segments of the audio file to the specified output directory with specified naming convention."""
    count = 0
    for i in range(num_segments):
        start_sample = i * segment_length_samples
        end_sample = start_sample + segment_length_samples
        segment = y[start_sample:end_sample]
        
        y_index = count // len(note_list)
        z_index = count % len(note_list)
        
        segment_filename = os.path.join(output_dir, f'{base_filename}_r{note_list[y_index]}_c{note_list[z_index]}.wav')
        # print(segment_filename)
        sf.write(segment_filename, segment, sr)
        
        count += 1
        if y_index >= len(note_list):
            break

def main(input_directory, segment_length_ms, output_directory, note_list):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    audio_files = get_audio_files(input_directory)
    
    for file in tqdm(audio_files, desc="Processing audio files"):
        num_segments, y, sr, segment_length_samples = calculate_segments(file, segment_length_ms)
        base_filename = os.path.splitext(os.path.basename(file))[0]
        save_segments(y, sr, segment_length_samples, num_segments, output_directory, base_filename, note_list)
        print(f'File: {file}, Number of {segment_length_ms} ms segments: {num_segments}')

# Example usage
input_directory = "input/path" #Input path for the files
segment_length_ms = "duration"  #Duration in ms for each signal
output_directory = "output/path" #Output path for the new files
note_list = ['127', '111', '095', '079', '063', '047', '031', '015']
main(input_directory, segment_length_ms, output_directory, note_list)


