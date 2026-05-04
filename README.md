# Thesis Code — RSF Kobol Expander Emulation

Scripts used in my thesis for preparing and analyzing audio data extracted from the **RSF Expander Kobol**, a vintage analog synthesizer. The goal is to model its audio processing behavior using neural networks.

---

## Datasets

Two datasets were recorded under different conditions:

- **Static dataset** — synth parameters (cutoff, resonance) are fixed throughout each recording  
- **Dynamic dataset** — synth parameters vary over time during recording

Each dataset contains **dry** (unprocessed) and **wet** (processed through the Kobol) signal pairs for supervised learning.

| Dataset | Link |
|---|---|
| Static | [Google Drive](https://drive.google.com/drive/folders/1EKZ2LMAII9UR5aEXQk_WEv7Boi2ipQ64?usp=sharing) |
| Dynamic | [Google Drive](https://drive.google.com/drive/folders/1d6f5vZuwfAvKZ0PByF3BoF4nRM7jVfpC?usp=sharing) |

---

## Pipeline Overview

```
DAW recordings (Bitwig / Ableton)
        │
        ▼
  Segment audio into fixed-length chunks
  (data_cutter.py / data_cutter_dynamic.py)
        │
        ▼
  Trim and normalize audio lengths
  (data_trimmer.py)
        │
        ▼
  Extract audio features to CSV
  (feature_extractor.py / feature_extractor_dynamic.py)
        │
        ▼
  Visualize features
  (features_visualizer.py / plot_multiple.py)
        │
        ▼
  Split into train / validation sets
  (train_val_split.py)
        │
        ▼
  Model training (mod_extraction / pedalnet)
```

---

## Scripts

### Audio Segmentation

**`data_cutter.py`**  
Cuts long recordings into fixed-length chunks. Segments are named using a row/column grid based on the MIDI note layout (`r{note}_c{note}.wav`).

**`data_cutter_dynamic.py`**  
Same as above but reads segment names from a CSV file. Used for dynamic recordings where the naming is not uniform.

**`data_trimmer.py`**  
Trims each audio file to a precise 3-second duration: silence (0.5 s) → content (2 s) → silence (0.5 s), with fade-in/fade-out applied. Ensures all files are exactly 144,000 samples at 48 kHz.

### Feature Extraction

**`feature_extractor.py`**  
Extracts audio features from static recordings and saves them to CSV. Features include:
- Pitch (via librosa piptrack, 75–16,000 Hz)
- MIDI onset delay (ms)
- Spectral centroid, bandwidth, roll-off, flatness, contrast (7 bands)
- RMS energy
- 13 MFCCs
- Cutoff frequency and resonance (MIDI and CV values, parsed from filename)

**`feature_extractor_dynamic.py`**  
Same as above but without cutoff/resonance columns, since those parameters vary during dynamic recordings.

### Visualization

**`features_visualizer.py`** / **`plot_multiple.py`**  
Plot extracted features from a CSV file as a multi-panel matplotlib figure. Each column gets its own subplot.

### Dataset Splitting

**`train_val_split.py`**  
Splits paired dry/wet audio files into training and validation sets. Creates four output folders: `train_dry`, `val_dry`, `train_wet`, `val_wet`. Supports configurable split ratio and random seed for reproducibility (default: 70/30, seed 42).

---

## Project Files

| File | Description |
|---|---|
| `Dynamic_recordings.als` | Ableton Live session for dynamic recordings |
| `Static_recordings.als` | Ableton Live session for static recordings |
| `MIDI_ref_note.mid` | Reference MIDI file used for onset detection |
| `Bitwig/Data_creation.bwproject` | Bitwig Studio project for data generation |
| `Bitwig/SERUM_NOISE.fxp` / `SERUM_SWEEP.fxp` | Serum presets used as audio sources |
| `audio_features_static.csv` | Extracted features from static recordings |
| `audio_features_dynamic_dry.csv` | Extracted features from dynamic dry signals |
| `audio_features_dynamic_wet.csv` | Extracted features from dynamic wet signals |
| `file_info.csv` | Metadata per recording (waveform, attack, decay, cutoff, resonance) |

---

## Models

Neural network models used for effects emulation:

**mod_extraction**  
- Original: https://github.com/christhetree/mod_extraction  
- My fork: https://github.com/AlvaroCrespo02/mod_extraction/tree/Tests_updated

**pedalnet**  
- Original: https://github.com/teddykoker/pedalnet  
- My fork: https://github.com/AlvaroCrespo02/pedalnet

---

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Key libraries: `librosa`, `soundfile`, `pydub`, `mido`, `numpy`, `pandas`, `scikit-learn`, `torch`, `torchaudio`, `pytorch-lightning`, `matplotlib`, `tqdm`, `auraloss`, `wandb`

---

## License

GNU General Public License v3 — see [LICENSE](LICENSE).
