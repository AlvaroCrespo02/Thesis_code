import librosa
import mido
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Function to process MIDI and audio files, extract onsets, and calculate delays
def process_midi_audio(midi_file, audio_file):
    # Extract MIDI note onsets
    def extract_midi_onsets(midi_file):
        mid = mido.MidiFile(midi_file)
        onsets = []
        time = 0
        for msg in mid:
            time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                onsets.append(time)
        return np.array(onsets)

    # Extract audio onsets using Librosa
    def extract_audio_onsets(audio_file):
        y, sr = librosa.load(audio_file, sr=None)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onsets = librosa.frames_to_time(onset_frames, sr=sr)
        return onsets

    # Calculate delay between MIDI and audio onsets
    def calculate_delay(midi_onsets, audio_onsets):
        delays = []
        for midi_onset in midi_onsets:
            closest_audio_onset = audio_onsets[np.argmin(np.abs(audio_onsets - midi_onset))]
            delay = closest_audio_onset - midi_onset
            delays.append(delay)
        return delays[0]

    midi_onsets = extract_midi_onsets(midi_file)
    audio_onsets = extract_audio_onsets(audio_file)
    delay = calculate_delay(midi_onsets, audio_onsets)

    # For dry signals, there is no delay since the VCO is constantly playing
    # delay = 0 
    return delay*1000

# Pitch detection
def detect_pitch(y, sr):
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=75, fmax=16000)
    # get indexes of the maximum value in each time slice
    max_indexes = np.argmax(magnitudes, axis=0)
    # get the pitches of the max indexes per time slice
    pitches = pitches[max_indexes, range(magnitudes.shape[1])]
    pitch = np.mean(pitches)
    return pitch

# Function to extract features from an audio file
def extract_features(audio_path, midi_file, samplerate):

    delay = process_midi_audio(midi_file, audio_file)
    y, sr = librosa.load(audio_path, sr=samplerate)
    y_trimmed, _ = librosa.effects.trim(y, top_db=10)

    mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y_trimmed, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)
    rms = librosa.feature.rms(y=y_trimmed)
    spectral_flatness = librosa.feature.spectral_flatness(y=y_trimmed)
    pitch = detect_pitch(y_trimmed, sr)

    features_dict = {
        'file_name': os.path.basename(audio_path),
        'Pitch (Hz)': pitch,
        'Input delay (ms)': delay,
        'MFCCs': np.mean(mfccs, axis=1),
        'Spectral Centroid (Hz)': np.mean(spectral_centroid),
        'Spectral Bandwidth (Hz)': np.mean(spectral_bandwidth),
        'Spectral Contrast (dB)': np.mean(spectral_contrast, axis=1),
        'Spectral Roll-off (Hz)': np.mean(spectral_rolloff),
        'RMS Energy': np.mean(rms),
        'Spectral Flatness': np.mean(spectral_flatness)
    }

    return features_dict

# Function to save features to CSV
def save_features_to_csv(features_list, csv_path):
    # Convert the list of features to a DataFrame
    df = pd.DataFrame(features_list)
    
    # Split the 'MFCCs' and 'Spectral Contrast' lists into separate columns
    mfccs_df = pd.DataFrame(df['MFCCs'].tolist(), index=df.index)
    spectral_contrast_df = pd.DataFrame(df['Spectral Contrast (dB)'].tolist(), index=df.index)
    
    # Rename columns
    mfccs_df.columns = [f'MFCCs_{i+1}' for i in mfccs_df.columns]
    spectral_contrast_df.columns = [f'Spectral Contrast (dB)_{i+1}' for i in spectral_contrast_df.columns]
    
    # Drop original list columns
    df = df.drop(columns=['MFCCs', 'Spectral Contrast (dB)'])
    
    # Concatenate the new DataFrames
    df = pd.concat([df, mfccs_df, spectral_contrast_df], axis=1)
    
    # Save to CSV
    df.to_csv(csv_path, index=False)

# Paths
midi_file = 'MIDI_ref_note.mid'
folder_path = 'folder/path'

# Get list of all audio files in the folder
audio_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.wav')]

# Extract features for all files and store them in a list
all_features = []
for audio_file in tqdm(audio_files):
    features = extract_features(audio_file, midi_file, 48000)
    if features is not None:
        all_features.append(features)
    # break

# print(all_features)

# Save features to CSV
csv_path = 'audio_features_dynamic_wet.csv'
save_features_to_csv(all_features, csv_path)