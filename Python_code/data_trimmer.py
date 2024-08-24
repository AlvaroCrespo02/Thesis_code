from pydub import AudioSegment
import os
from tqdm import tqdm

def process_audio_files(input_directory, output_directory, fade_duration=10, sample_rate=48000):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Get a list of all audio files in the input directory
    audio_files = [f for f in os.listdir(input_directory) if f.endswith('.wav') or f.endswith('.mp3')]

    # Loop through all the files in the input directory with a progress bar
    for filename in tqdm(audio_files, desc="Processing audio files"):
        # Load the audio file
        audio = AudioSegment.from_file(os.path.join(input_directory, filename))
        
        # Check if the audio has 144000 samples
        if len(audio.get_array_of_samples()) == 144000:
            # Calculate the duration of the audio in milliseconds
            audio_length_ms = (144000 / sample_rate) * 1000  # 3 seconds
            
            # Extract the middle 2 seconds
            middle_part = audio[500:2500]
            
            # Apply fade-in and fade-out to the middle part
            middle_part = middle_part.fade_in(fade_duration).fade_out(fade_duration)
            
            # Create .5 seconds of silence
            silence = AudioSegment.silent(duration=500)
            
            # Concatenate silence, middle part, and silence again
            processed_audio = silence + middle_part + silence
            
            # Ensure the processed audio also has 144000 samples
            if len(processed_audio.get_array_of_samples()) != 144000:
                # Adjust processed_audio length to be exactly 144000 samples
                processed_audio = processed_audio.set_frame_rate(sample_rate)
                processed_audio = processed_audio.set_channels(audio.channels)
                processed_audio = processed_audio.set_sample_width(audio.sample_width)
                processed_audio = processed_audio[:144000 * 1000 // sample_rate]  # Trim or extend to exact length
            
            # Export the processed audio
            output_file = os.path.join(output_directory, filename)
            processed_audio.export(output_file, format="wav")  # Adjust format as needed
            
            # print(f"Processed file: {filename}")
        else:
            print(f"Skipped file (not 144000 samples long): {filename}")

# Example usage:
input_directory = r'C:\Users\alvar\Documents\Thesis\Data\GoodOne\Kobol_VCO_dynamic\Dry'
output_directory = r'C:\Users\alvar\Documents\Thesis\Data\GoodOne\Kobol_VCO_dynamic\Dry_trimmed'
process_audio_files(input_directory, output_directory)

