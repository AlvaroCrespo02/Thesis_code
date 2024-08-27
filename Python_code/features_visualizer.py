import mido
import numpy as np
import librosa
import pandas as pd
import os
import matplotlib.pyplot as plt

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
        return np.array(delays)

    midi_onsets = extract_midi_onsets(midi_file)
    audio_onsets = extract_audio_onsets(audio_file)
    delays = calculate_delay(midi_onsets, audio_onsets)

    return delays

# Function to detect pitch in an audio signal
def detect_pitch(y, sr):
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=75, fmax=16000)
    # get indexes of the maximum value in each time slice
    max_indexes = np.argmax(magnitudes, axis=0)
    # get the pitches of the max indexes per time slice
    pitches = pitches[max_indexes, range(magnitudes.shape[1])]
    pitch = np.mean(pitches)
    return pitch

# Function to extract features from an audio file
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=48000)
    y_trimmed, _ = librosa.effects.trim(y,  top_db=10)

    mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y_trimmed, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)
    rms = librosa.feature.rms(y=y_trimmed)
    spectral_flatness = librosa.feature.spectral_flatness(y=y_trimmed)
    spectrogram = np.abs(librosa.stft(y_trimmed))  # Compute the spectrogram

    # Pitch estimation
    pitch = detect_pitch(y_trimmed, 48000)

    features_dict = {
        'file_name': os.path.basename(audio_path),
        'Pitch Mean': pitch,
        'MFCCs': mfccs,
        'Spectral Centroid': spectral_centroid,
        'Spectral Bandwidth': spectral_bandwidth,
        'Spectral Contrast': spectral_contrast,
        'Spectral Roll-off': spectral_rolloff,
        'RMS Energy': rms,
        'Spectral Flatness': spectral_flatness,
        'Spectrogram': spectrogram
    }
    # print(features_dict)
    return features_dict

# Function to plot features
def plot_features(features):
    mfccs = features['MFCCs']
    spectral_centroid = features['Spectral Centroid']
    spectral_bandwidth = features['Spectral Bandwidth']
    spectral_contrast = features['Spectral Contrast']
    spectral_rolloff = features['Spectral Roll-off']
    rms = features['RMS Energy']
    spectral_flatness = features['Spectral Flatness']
    spectrogram = features['Spectrogram']

    plt.figure(figsize=(15, 12))

    # Plot MFCCs
    plt.subplot(4, 2, 1)
    librosa.display.specshow(mfccs, sr=48000, x_axis='time')
    plt.colorbar()
    plt.title('MFCCs')

    # Plot Spectral Centroid
    plt.subplot(4, 2, 2)
    plt.semilogy(spectral_centroid.T, label='Spectral Centroid')
    plt.ylabel('Hz')
    plt.xticks([])
    plt.xlim([0, spectral_centroid.shape[-1]])
    plt.legend(loc='upper right')
    plt.title('Spectral Centroid')

    # Plot Spectral Bandwidth
    plt.subplot(4, 2, 3)
    plt.semilogy(spectral_bandwidth.T, label='Spectral Bandwidth')
    plt.ylabel('Hz')
    plt.xticks([])
    plt.xlim([0, spectral_bandwidth.shape[-1]])
    plt.legend(loc='upper right')
    plt.title('Spectral Bandwidth')

    # Plot Spectral Contrast
    plt.subplot(4, 2, 4)
    librosa.display.specshow(spectral_contrast, sr=48000, x_axis='time')
    plt.colorbar()
    plt.title('Spectral Contrast')

    # Plot Spectral Roll-off
    plt.subplot(4, 2, 5)
    plt.semilogy(spectral_rolloff.T, label='Spectral Roll-off')
    plt.ylabel('Hz')
    plt.xticks([])
    plt.xlim([0, spectral_rolloff.shape[-1]])
    plt.legend(loc='upper right')
    plt.title('Spectral Roll-off')

    # Plot RMS Energy
    plt.subplot(4, 2, 6)
    plt.semilogy(rms.T, label='RMS Energy')
    plt.ylabel('Value')
    plt.xlim([0, rms.shape[-1]])
    plt.legend(loc='upper right')
    plt.title('RMS Energy')

    # Plot Spectral Flatness
    plt.subplot(4, 2, 8)
    plt.semilogy(spectral_flatness.T, label='Spectral Flatness')
    plt.xlim([0, spectral_flatness.shape[-1]])
    plt.legend(loc='upper right')
    plt.title('Spectral Flatness')

    # Plot Spectrogram
    plt.subplot(4, 2, 7)
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=48000, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')

    plt.tight_layout()
    plt.show()

# Replace these file paths with your actual file paths
midi_file = 'MIDI_ref_note.mid'
audio_file = 'B2_101.wav'

# Extract MIDI and audio onsets
delays = process_midi_audio(midi_file, audio_file)
print(delays)

# Extract features for a file and store them in a list
all_features = [extract_features(audio_file)]

# Plot features for the first audio file as an example
plot_features(all_features[0])

