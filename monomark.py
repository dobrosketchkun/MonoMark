#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import librosa
import librosa.display
import soundfile as sf
import os
from scipy import signal
import argparse

# ========================
# DEFAULT SETTINGS (can be overridden via CLI)
# ========================
AUDIO_PATH = "input_audio.wav"    # Path to input audio file
IMAGE_PATH = "input_image.png"    # Path to input image file
OUTPUT_FOLDER = "output"          # Folder to save outputs
START_TIME = 10.0                 # Start time (s) for embedding the image
DURATION = 2.0                    # Duration (s) for image embedding (ignored if auto-duration is enabled)
FFT_SIZE = 2048                   # FFT size for spectrogram generation
HOP_LENGTH = 512                  # Hop length for spectrogram generation
PREVIEW_SPECTROGRAMS = True       # Display spectrograms after processing
SAVE_SPECTROGRAMS = True          # Save spectrogram images to file
SAVE_AUDIO = True                 # Save modified audio to file

# Frequency range for image embedding (Hz)
MIN_FREQ = 4000                   # Minimum frequency for image embedding
MAX_FREQ = 16000                  # Maximum frequency for image embedding
FLIP_IMAGE = True                 # Flip the input image vertically
COMPENSATE_LOG_SCALE = True       # Pre-distort image for logarithmic frequency scale
PRESERVE_ASPECT_RATIO = True      # Preserve the image's original aspect ratio when resizing
AUTO_DURATION = True              # Automatically calculate embedding duration based on image aspect ratio
IMAGE_INTENSITY = 1.0             # Intensity of the image effect (0.0-1.0)

# New setting for visible stereo mode
VISIBLE_IN_STEREO = False         # Make image visible in stereo spectrograms too

# Spectrogram visualization settings
SPEC_MIN_FREQ = None              # Minimum frequency to display in spectrogram (Hz)
SPEC_MAX_FREQ = None              # Maximum frequency to display in spectrogram (Hz)
SPEC_START_TIME = None            # Start time for spectrogram visualization (s)
SPEC_END_TIME = None              # End time for spectrogram visualization (s)
ZOOM_TO_IMAGE = True              # Automatically zoom spectrogram display to image area

# ---------------------------
# FUNCTION DEFINITIONS
# ---------------------------
def load_audio(file_path):
    """Load an audio file and ensure it's stereo."""
    print(f"Loading audio file: {file_path}")
    try:
        audio, sr = librosa.load(file_path, sr=None, mono=False)
        if len(audio.shape) == 1:
            print("Converting mono audio to stereo")
            audio = np.array([audio, audio])
        elif audio.shape[0] > 2:
            print("Audio has more than 2 channels; using first 2 channels")
            audio = audio[:2]
        return audio, sr
    except Exception as e:
        print(f"Error loading audio file: {e}")
        raise

def load_image(file_path):
    """Load an image, convert it to grayscale (0–1), and flip vertically if needed."""
    print(f"Loading image file: {file_path}")
    try:
        img = Image.open(file_path).convert('L')
        img_array = np.array(img) / 255.0
        if FLIP_IMAGE:
            img_array = np.flipud(img_array)
        return img_array
    except Exception as e:
        print(f"Error loading image file: {e}")
        raise

def calculate_auto_duration(image, min_freq, max_freq, sr, hop_length, n_fft=FFT_SIZE):
    """
    Calculate the duration needed to preserve the image's aspect ratio in the spectrogram.
    """
    img_height, img_width = image.shape
    img_aspect_ratio = img_width / img_height
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    min_bin = np.argmin(np.abs(freqs - min_freq))
    max_bin = np.argmin(np.abs(freqs - max_freq))
    freq_bins = max_bin - min_bin + 1
    required_frames = int(freq_bins * img_aspect_ratio)
    duration = required_frames * hop_length / sr
    print(f"Auto-calculated duration: {duration:.2f}s (aspect ratio: {img_aspect_ratio:.2f})")
    return duration

def get_spectrogram(audio, sr, n_fft=FFT_SIZE, hop_length=HOP_LENGTH):
    """Calculate spectrogram magnitude for mono or stereo audio."""
    if len(audio.shape) == 1:
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        spec = np.abs(stft)
    else:
        stft_left = librosa.stft(audio[0], n_fft=n_fft, hop_length=hop_length)
        stft_right = librosa.stft(audio[1], n_fft=n_fft, hop_length=hop_length)
        spec = [np.abs(stft_left), np.abs(stft_right)]
    return spec

def display_spectrogram(spec, sr, hop_length, title, file_path=None, show_mono=False, mono_spec=None):
    """Display and/or save spectrogram(s) with optional mono version.
    
    In ZOOM_TO_IMAGE mode the plot width is adjusted to avoid horizontal stretching.
    """
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FFT_SIZE)
    if ZOOM_TO_IMAGE and MIN_FREQ is not None and MAX_FREQ is not None:
        y_min, y_max = MIN_FREQ, MAX_FREQ
    else:
        y_min = SPEC_MIN_FREQ if SPEC_MIN_FREQ is not None else freqs[0]
        y_max = SPEC_MAX_FREQ if SPEC_MAX_FREQ is not None else freqs[-1]

    if ZOOM_TO_IMAGE and START_TIME is not None and DURATION is not None:
        x_min, x_max = START_TIME, START_TIME + DURATION
    else:
        x_min = SPEC_START_TIME if SPEC_START_TIME is not None else 0
        x_max = SPEC_END_TIME if SPEC_END_TIME is not None else (spec[0].shape[1] if isinstance(spec, list) else spec.shape[1]) * hop_length / sr

    x_min_idx = int(x_min * sr / hop_length)
    x_max_idx = int(x_max * sr / hop_length)

    # Adjust figure size if zooming to image
    if ZOOM_TO_IMAGE:
        num_frames_in_zoom = x_max_idx - x_min_idx
        min_bin = np.argmin(np.abs(freqs - y_min))
        max_bin = np.argmin(np.abs(freqs - y_max))
        freq_bins_in_zoom = max_bin - min_bin + 1
        aspect_ratio = num_frames_in_zoom / freq_bins_in_zoom
        adjusted_aspect = aspect_ratio / 2.0  # reduce horizontal stretch
        base_height = 8
        figure_width = base_height * adjusted_aspect
        plt.figure(figsize=(figure_width, base_height))
    else:
        if show_mono and mono_spec is not None:
            plt.figure(figsize=(15, 10))
        elif isinstance(spec, list):
            plt.figure(figsize=(12, 8))
        else:
            plt.figure(figsize=(12, 8))

    if show_mono and mono_spec is not None:
        plt.subplot(3, 1, 1)
        librosa.display.specshow(librosa.amplitude_to_db(spec[0], ref=np.max),
                                 y_axis='log', x_axis='time', sr=sr,
                                 hop_length=hop_length, vmin=-80, vmax=0)
        plt.title(f"{title} - Left Channel")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.colorbar(format='%+2.0f dB')

        plt.subplot(3, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(spec[1], ref=np.max),
                                 y_axis='log', x_axis='time', sr=sr,
                                 hop_length=hop_length, vmin=-80, vmax=0)
        plt.title(f"{title} - Right Channel")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.colorbar(format='%+2.0f dB')

        plt.subplot(3, 1, 3)
        librosa.display.specshow(librosa.amplitude_to_db(mono_spec, ref=np.max),
                                 y_axis='log', x_axis='time', sr=sr,
                                 hop_length=hop_length, vmin=-80, vmax=0)
        plt.title(f"{title} - Mono (Collapsed)")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.colorbar(format='%+2.0f dB')
    elif isinstance(spec, list):
        plt.subplot(2, 1, 1)
        librosa.display.specshow(librosa.amplitude_to_db(spec[0], ref=np.max),
                                 y_axis='log', x_axis='time', sr=sr,
                                 hop_length=hop_length, vmin=-80, vmax=0)
        plt.title(f"{title} - Left Channel")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.colorbar(format='%+2.0f dB')

        plt.subplot(2, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(spec[1], ref=np.max),
                                 y_axis='log', x_axis='time', sr=sr,
                                 hop_length=hop_length, vmin=-80, vmax=0)
        plt.title(f"{title} - Right Channel")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.colorbar(format='%+2.0f dB')
    else:
        librosa.display.specshow(librosa.amplitude_to_db(spec, ref=np.max),
                                 y_axis='log', x_axis='time', sr=sr,
                                 hop_length=hop_length, vmin=-80, vmax=0)
        plt.title(title)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    if file_path:
        plt.savefig(file_path, dpi=300)
        print(f"Saved spectrogram to {file_path}")
    if PREVIEW_SPECTROGRAMS:
        plt.show()
    plt.close()

def compensate_log_scale(image, min_freq, max_freq):
    """Pre-distort image to compensate for logarithmic frequency scaling."""
    if not COMPENSATE_LOG_SCALE:
        return image
    height, width = image.shape
    log_min = np.log(min_freq)
    log_max = np.log(max_freq)
    src_rows = np.arange(height)
    log_points = np.linspace(log_min, log_max, height)
    target_freqs = np.exp(log_points)
    dst_rows = (target_freqs - min_freq) / (max_freq - min_freq) * (height - 1)
    dst_rows = np.clip(dst_rows, 0, height - 1)
    inverse_map = np.zeros((height, width, 2), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            idx = np.argmin(np.abs(dst_rows - y))
            inverse_map[y, x, 0] = x
            inverse_map[y, x, 1] = src_rows[idx]
    from scipy.ndimage import map_coordinates
    img_array = image * 255
    remapped = map_coordinates(img_array, [inverse_map[:,:,1].flatten(), inverse_map[:,:,0].flatten()], order=1)
    compensated_img = remapped.reshape(img_array.shape) / 255.0
    print("Applied logarithmic frequency scale compensation to image")
    return compensated_img

def resize_image_to_spectrogram(image, target_shape, min_freq=None, max_freq=None):
    """Resize image to match the spectrogram region."""
    if COMPENSATE_LOG_SCALE and min_freq is not None and max_freq is not None:
        image = compensate_log_scale(image, min_freq, max_freq)
    if PRESERVE_ASPECT_RATIO:
        orig_height, orig_width = image.shape
        target_height, target_width = target_shape
        aspect_ratio = orig_width / orig_height
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
        if new_width > target_width:
            new_width = target_width
            new_height = int(new_width / aspect_ratio)
        img = Image.fromarray((image * 255).astype(np.uint8))
        img_resized = img.resize((new_width, new_height))
        target_img = Image.new('L', (target_width, target_height), 0)
        left = (target_width - new_width) // 2
        top = (target_height - new_height) // 2
        target_img.paste(img_resized, (left, top))
        return np.array(target_img) / 255.0
    else:
        img = Image.fromarray((image * 255).astype(np.uint8))
        img_resized = img.resize((target_shape[1], target_shape[0]))
        return np.array(img_resized) / 255.0

def embed_image_in_stereo(audio, sr, image, start_time, duration, n_fft=FFT_SIZE, hop_length=HOP_LENGTH):
    """
    Embed an image into stereo audio.
    
    In default mode (VISIBLE_IN_STEREO=False), the image is hidden in stereo but visible in mono.
    In visible mode (VISIBLE_IN_STEREO=True), the image is directly visible in both stereo and mono.
    """
    print(f"Embedding image at {start_time}s for {duration}s duration")
    start_sample = int(start_time * sr)
    duration_samples = int(duration * sr)
    audio_section = audio[:, start_sample:start_sample + duration_samples]
    stft_left = librosa.stft(audio_section[0], n_fft=n_fft, hop_length=hop_length)
    stft_right = librosa.stft(audio_section[1], n_fft=n_fft, hop_length=hop_length)
    num_freqs, num_frames = stft_left.shape
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    min_bin = np.argmin(np.abs(freqs - MIN_FREQ))
    max_bin = np.argmin(np.abs(freqs - MAX_FREQ))
    print(f"Embedding image in frequency range: {freqs[min_bin]:.1f}Hz - {freqs[max_bin]:.1f}Hz (bins {min_bin}-{max_bin})")
    freq_range = max_bin - min_bin + 1
    target_shape = (freq_range, num_frames)
    frames_per_sec = sr / hop_length
    print(f"Spectrogram resolution: {frames_per_sec:.1f} frames per second")
    stft_duration = num_frames / frames_per_sec
    print(f"STFT covers {stft_duration:.2f} seconds with {num_frames} frames")
    original_aspect = image.shape[1] / image.shape[0]
    spectrogram_aspect = num_frames / freq_range
    print(f"Original image aspect ratio: {original_aspect:.2f}")
    print(f"Spectrogram aspect ratio: {spectrogram_aspect:.2f}")
    print(f"Image intensity: {IMAGE_INTENSITY:.2f}")
    if not PRESERVE_ASPECT_RATIO and abs(original_aspect - spectrogram_aspect) > 0.2:
        print("WARNING: Image will be significantly stretched. Consider enabling PRESERVE_ASPECT_RATIO.")
    if IMAGE_INTENSITY < 0.3:
        print("NOTE: Low intensity setting will make the image more subtle but still visible.")
    resized_image = resize_image_to_spectrogram(image, target_shape, freqs[min_bin], freqs[max_bin])
    stft_left_new = stft_left.copy()
    stft_right_new = stft_right.copy()
    
    if VISIBLE_IN_STEREO:
        print("Using VISIBLE_IN_STEREO mode: image will be visible in stereo spectrograms")
        # Direct embedding in magnitude for both channels
        for i in range(freq_range):
            freq_bin = i + min_bin
            for j in range(num_frames):
                img_val = resized_image[i, j]
                phase_left = np.angle(stft_left[freq_bin, j])
                phase_right = np.angle(stft_right[freq_bin, j])
                
                # Calculate original magnitudes
                mag_left = np.abs(stft_left[freq_bin, j])
                mag_right = np.abs(stft_right[freq_bin, j])
                
                # Base magnitude adjustment (higher for brighter pixels)
                mag_factor = 1.0 + (img_val * IMAGE_INTENSITY * 2.0)
                
                # Apply magnitude adjustments to both channels
                new_mag_left = mag_left * mag_factor
                new_mag_right = mag_right * mag_factor
                
                # Keep original phases
                stft_left_new[freq_bin, j] = new_mag_left * np.exp(1j * phase_left)
                stft_right_new[freq_bin, j] = new_mag_right * np.exp(1j * phase_right)
    else:
        print("Using default hidden mode: image will only be visible when collapsed to mono")
        np.random.seed(42)
        for i in range(freq_range):
            freq_bin = i + min_bin
            for j in range(num_frames):
                img_val = resized_image[i, j]
                mag_orig = (np.abs(stft_left[freq_bin, j]) + np.abs(stft_right[freq_bin, j])) / 2
                base_phase = np.random.random() * 2 * np.pi
                full_phase_diff = np.pi * (1 - img_val)
                if IMAGE_INTENSITY >= 1.0:
                    phase_diff = full_phase_diff
                    mag_boost = 1.5
                else:
                    min_phase_diff = full_phase_diff * 0.7
                    phase_diff = min_phase_diff + (full_phase_diff - min_phase_diff) * IMAGE_INTENSITY
                    mag_boost = 1.0 + (0.5 * IMAGE_INTENSITY)
                stft_left_new[freq_bin, j] = mag_orig * mag_boost * np.exp(1j * base_phase)
                stft_right_new[freq_bin, j] = mag_orig * mag_boost * np.exp(1j * (base_phase + phase_diff))
                
    audio_left_new = librosa.istft(stft_left_new, hop_length=hop_length, length=len(audio_section[0]))
    audio_right_new = librosa.istft(stft_right_new, hop_length=hop_length, length=len(audio_section[1]))
    max_val = max(np.max(np.abs(audio_left_new)), np.max(np.abs(audio_right_new)))
    if max_val > 1.0:
        print(f"Normalizing audio to prevent clipping (factor: {max_val:.2f})")
        audio_left_new = audio_left_new / max_val * 0.95
        audio_right_new = audio_right_new / max_val * 0.95
    audio_modified = audio.copy()
    audio_modified[0, start_sample:start_sample + duration_samples] = audio_left_new
    audio_modified[1, start_sample:start_sample + duration_samples] = audio_right_new
    return audio_modified

def create_mono(audio):
    """Convert stereo audio to mono by averaging channels."""
    return np.mean(audio, axis=0)

def main():
    """Main function to execute the audio processing and image embedding."""
    try:
        audio, sr = load_audio(AUDIO_PATH)
        image = load_image(IMAGE_PATH)
        print(f"Audio: {audio.shape} channels, {sr} Hz sample rate, {audio.shape[1]/sr:.2f} seconds")
        print(f"Image: {image.shape[0]}x{image.shape[1]} pixels")
        print(f"Embedding mode: {'VISIBLE in stereo' if VISIBLE_IN_STEREO else 'HIDDEN until mono collapse'}")
        global DURATION
        if AUTO_DURATION:
            DURATION = calculate_auto_duration(image, MIN_FREQ, MAX_FREQ, sr, HOP_LENGTH)
        print(f"Embedding image at {START_TIME:.2f}-{START_TIME+DURATION:.2f}s, {MIN_FREQ}-{MAX_FREQ}Hz")
        freqs = librosa.fft_frequencies(sr=sr, n_fft=FFT_SIZE)
        min_bin = np.argmin(np.abs(freqs - MIN_FREQ))
        max_bin = np.argmin(np.abs(freqs - MAX_FREQ))
        print(f"Embedding will use {max_bin - min_bin + 1} frequency bins")
        modified_audio = embed_image_in_stereo(audio, sr, image, START_TIME, DURATION)
        modified_mono_audio = create_mono(modified_audio)
        print("Calculating modified spectrograms...")
        mod_stereo_spec = get_spectrogram(modified_audio, sr)
        mod_mono_spec = get_spectrogram(modified_mono_audio, sr)
        if SAVE_SPECTROGRAMS:
            display_spectrogram(mod_stereo_spec, sr, HOP_LENGTH, "Modified Audio", 
                                os.path.join(OUTPUT_FOLDER, "modified_spectrograms.png"),
                                show_mono=True, mono_spec=mod_mono_spec)
        else:
            display_spectrogram(mod_stereo_spec, sr, HOP_LENGTH, "Modified Audio",
                                show_mono=True, mono_spec=mod_mono_spec)
        if SAVE_AUDIO:
            sf.write(os.path.join(OUTPUT_FOLDER, "modified_stereo.wav"), modified_audio.T, sr)
            sf.write(os.path.join(OUTPUT_FOLDER, "modified_mono.wav"), modified_mono_audio, sr)
            print(f"Saved audio files to {OUTPUT_FOLDER}")
        print("Processing complete!")
    except Exception as e:
        print(f"Error in main function: {e}")
        raise

# ---------------------------
# CLI INTERFACE
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed an image into an audio file's spectrogram with hidden stereo encoding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-a", "--audio-path", default=AUDIO_PATH,
                        help="Path to the input audio file.")
    parser.add_argument("-i", "--image-path", default=IMAGE_PATH,
                        help="Path to the input image file (black and white).")
    parser.add_argument("-o", "--output-folder", default=OUTPUT_FOLDER,
                        help="Folder to save output audio and spectrogram images.")
    parser.add_argument("-s", "--start-time", type=float, default=START_TIME,
                        help="Start time (s) for embedding the image.")
    parser.add_argument("-d", "--duration", type=float, default=DURATION,
                        help="Duration (s) for image embedding (ignored if auto-duration is enabled).")
    parser.add_argument("-f", "--fft-size", type=int, default=FFT_SIZE,
                        help="FFT size for spectrogram generation.")
    parser.add_argument("-l", "--hop-length", type=int, default=HOP_LENGTH,
                        help="Hop length for spectrogram generation.")
    parser.add_argument("-v", "--preview", dest="preview_spectrograms",
                        action="store_true", help="Display spectrograms after processing.")
    parser.add_argument("--no-preview", dest="preview_spectrograms",
                        action="store_false", help="Do not display spectrograms after processing.")
    parser.set_defaults(preview_spectrograms=PREVIEW_SPECTROGRAMS)
    parser.add_argument("-S", "--save-spec", dest="save_spectrograms",
                        action="store_true", help="Save spectrogram images to file.")
    parser.add_argument("--no-save-spec", dest="save_spectrograms",
                        action="store_false", help="Do not save spectrogram images to file.")
    parser.set_defaults(save_spectrograms=SAVE_SPECTROGRAMS)
    parser.add_argument("-A", "--save-audio", dest="save_audio",
                        action="store_true", help="Save modified audio to file.")
    parser.add_argument("--no-save-audio", dest="save_audio",
                        action="store_false", help="Do not save modified audio to file.")
    parser.set_defaults(save_audio=SAVE_AUDIO)
    parser.add_argument("-m", "--min-freq", type=float, default=MIN_FREQ,
                        help="Minimum frequency (Hz) for image embedding.")
    parser.add_argument("-M", "--max-freq", type=float, default=MAX_FREQ,
                        help="Maximum frequency (Hz) for image embedding.")
    parser.add_argument("-x", "--flip", dest="flip_image",
                        action="store_true", help="Flip the input image vertically.")
    parser.add_argument("--no-flip", dest="flip_image",
                        action="store_false", help="Do not flip the input image vertically.")
    parser.set_defaults(flip_image=FLIP_IMAGE)
    parser.add_argument("-c", "--comp-log", dest="compensate_log_scale",
                        action="store_true", help="Pre-distort image for logarithmic frequency scaling.")
    parser.add_argument("--no-comp-log", dest="compensate_log_scale",
                        action="store_false", help="Do not pre-distort image for logarithmic frequency scaling.")
    parser.set_defaults(compensate_log_scale=COMPENSATE_LOG_SCALE)
    parser.add_argument("-r", "--preserve", dest="preserve_aspect_ratio",
                        action="store_true", help="Preserve the original image aspect ratio when resizing.")
    parser.add_argument("--no-preserve", dest="preserve_aspect_ratio",
                        action="store_false", help="Do not preserve the image's original aspect ratio.")
    parser.set_defaults(preserve_aspect_ratio=PRESERVE_ASPECT_RATIO)
    parser.add_argument("-t", "--auto-duration", dest="auto_duration",
                        action="store_true", help="Automatically calculate embedding duration based on image aspect ratio.")
    parser.add_argument("--no-auto-duration", dest="auto_duration",
                        action="store_false", help="Do not automatically calculate embedding duration.")
    parser.set_defaults(auto_duration=AUTO_DURATION)
    parser.add_argument("-I", "--intensity", type=float, default=IMAGE_INTENSITY,
                        help="Intensity of the image effect (0.0-1.0).")
    parser.add_argument("-n", "--spec-min-freq", type=float, default=SPEC_MIN_FREQ,
                        help="Minimum frequency to display in the spectrogram.")
    parser.add_argument("-N", "--spec-max-freq", type=float, default=SPEC_MAX_FREQ,
                        help="Maximum frequency to display in the spectrogram.")
    parser.add_argument("-T", "--spec-start", type=float, default=SPEC_START_TIME,
                        help="Start time (s) for spectrogram visualization.")
    parser.add_argument("-E", "--spec-end", type=float, default=SPEC_END_TIME,
                        help="End time (s) for spectrogram visualization.")
    parser.add_argument("-z", "--zoom", dest="zoom_to_image",
                        action="store_true", help="Zoom spectrogram display to the image area.")
    parser.add_argument("--no-zoom", dest="zoom_to_image",
                        action="store_false", help="Do not zoom spectrogram display to the image area.")
    parser.set_defaults(zoom_to_image=ZOOM_TO_IMAGE)
    
    # New option for visible stereo mode - single flag to explicitly enable
    parser.add_argument("-V", "--visible-stereo", dest="visible_in_stereo",
                        action="store_true", help="Make image visible in stereo spectrograms too (not just in mono).")
    
    args = parser.parse_args()
    
    # Override globals with CLI arguments.
    globals().update({
        "AUDIO_PATH": args.audio_path,
        "IMAGE_PATH": args.image_path,
        "OUTPUT_FOLDER": args.output_folder,
        "START_TIME": args.start_time,
        "DURATION": args.duration,
        "FFT_SIZE": args.fft_size,
        "HOP_LENGTH": args.hop_length,
        "PREVIEW_SPECTROGRAMS": args.preview_spectrograms,
        "SAVE_SPECTROGRAMS": args.save_spectrograms,
        "SAVE_AUDIO": args.save_audio,
        "MIN_FREQ": args.min_freq,
        "MAX_FREQ": args.max_freq,
        "FLIP_IMAGE": args.flip_image,
        "COMPENSATE_LOG_SCALE": args.compensate_log_scale,
        "PRESERVE_ASPECT_RATIO": args.preserve_aspect_ratio,
        "AUTO_DURATION": args.auto_duration,
        "IMAGE_INTENSITY": args.intensity,
        "SPEC_MIN_FREQ": args.spec_min_freq,
        "SPEC_MAX_FREQ": args.spec_max_freq,
        "SPEC_START_TIME": args.spec_start,
        "SPEC_END_TIME": args.spec_end,
        "ZOOM_TO_IMAGE": args.zoom_to_image,
        "VISIBLE_IN_STEREO": args.visible_in_stereo
    })
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    main()
