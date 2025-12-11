import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import os

class OpticalSideChannelAnalyzer:
    def __init__(self, video_path, fps=60):
        self.video_path = video_path
        self.fps = fps
        self.frames = []
        self.intensity_signal = None
        self.detrended_signal = None
        self.time_axis = None
        self.frequencies = None
        self.fft_magnitude = None
        
    def load_video(self, downscale_factor=2):
        """Load video and extract all frames with downscaling."""
        print(f"Loading video: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Downscale to speed up processing 
            if downscale_factor > 1:
                h, w = gray_frame.shape
                gray_frame = cv2.resize(gray_frame, 
                                       (w // downscale_factor, h // downscale_factor))
            
            self.frames.append(gray_frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"  Loaded {frame_count} frames...", end='\r')
        
        cap.release()
        print(f"\nLoaded {frame_count} frames")
        
        self.time_axis = np.arange(len(self.frames)) / self.fps
        self.downscale_factor = downscale_factor
        return self.frames[0] if self.frames else None
    
    def select_roi_interactive(self, first_frame):
        print("\nSelect ROI:")
        print("- Click and drag to select the bright spot region")
        print("- Press SPACE or ENTER when done")
        print("- Press ESC to cancel")
        
        display_frame = cv2.convertScaleAbs(first_frame, alpha=2.0, beta=50)
        roi = cv2.selectROI("Select ROI (bright spot)", display_frame, False)
        cv2.destroyAllWindows()
        
        if roi[2] == 0 or roi[3] == 0:
            print("No ROI selected, finding brightest region automatically...")
            return self.find_brightest_roi(first_frame)
        
        return roi
    
    def find_brightest_roi(self, first_frame, roi_size=40):
        print("Finding brightest region...")
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(first_frame)
        x, y = max_loc
        
        if hasattr(self, 'downscale_factor') and self.downscale_factor > 1:
            x *= self.downscale_factor
            y *= self.downscale_factor
            roi_size *= self.downscale_factor
        
        half_size = roi_size // 2
        x_start = max(0, x - half_size)
        y_start = max(0, y - half_size)
        roi = (x_start, y_start, roi_size, roi_size)
        print(f"Auto-selected ROI at brightest point: ({x}, {y})")
        
        return roi
    
    def extract_intensity_signal(self, roi, visualize=True):
        x, y, w, h = roi
        print(f"Extracting intensity from ROI: x={x}, y={y}, w={w}, h={h}")
        
        if hasattr(self, 'downscale_factor') and self.downscale_factor > 1:
            scale = self.downscale_factor
            x_scaled = x // scale
            y_scaled = y // scale
            w_scaled = w // scale
            h_scaled = h // scale
        else:
            x_scaled, y_scaled, w_scaled, h_scaled = x, y, w, h
        
        intensities = []
        for i, frame in enumerate(self.frames):
            # Extract ROI and compute mean intensity
            roi_region = frame[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled]
            mean_intensity = np.mean(roi_region)
            intensities.append(mean_intensity)
            
            if (i + 1) % 100 == 0:
                print(f"  Processing frame {i+1}/{len(self.frames)}...", end='\r')
        
        print(f"\n  Extracted {len(intensities)} intensity values")
        self.intensity_signal = np.array(intensities)
        
        # Visualize ROI
        if visualize and len(self.frames) > 0:
            print("Creating ROI visualization...")
            first_frame_color = cv2.cvtColor(self.frames[0], cv2.COLOR_GRAY2BGR)
            cv2.rectangle(first_frame_color, 
                         (x_scaled, y_scaled), 
                         (x_scaled+w_scaled, y_scaled+h_scaled), 
                         (0, 255, 0), 2)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(first_frame_color, cv2.COLOR_BGR2RGB))
            plt.title("Selected ROI (green rectangle)")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('roi_selection.png', dpi=150, bbox_inches='tight')
            plt.close()  
            print("  Saved roi_selection.png")
        
        return self.intensity_signal
    
    def detrend_signal(self):
        if self.intensity_signal is None:
            raise ValueError("Must extract intensity signal first")
        self.detrended_signal = signal.detrend(self.intensity_signal)
        
        return self.detrended_signal
    
    def compute_fft(self):
        if self.detrended_signal is None:
            raise ValueError("Must detrend signal first")
        
        N = len(self.detrended_signal)
        
        # Compute FFT
        fft_values = fft(self.detrended_signal)
        self.fft_magnitude = np.abs(fft_values)
        self.frequencies = fftfreq(N, 1/self.fps)
        positive_freq_idx = self.frequencies > 0
        self.frequencies = self.frequencies[positive_freq_idx]
        self.fft_magnitude = self.fft_magnitude[positive_freq_idx]
        
        return self.frequencies, self.fft_magnitude
    
    def plot_intensity_signals(self, save_path='intensity_signals.png'):
        if self.intensity_signal is None or self.detrended_signal is None:
            raise ValueError("Must extract and detrend signals first")
        
        print(f"Creating intensity plots...")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Raw signal
        axes[0].plot(self.time_axis, self.intensity_signal, 'b-', linewidth=0.8)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Mean Intensity')
        axes[0].set_title('Raw Intensity Signal I(t)')
        axes[0].grid(True, alpha=0.3)
        
        # Detrended signal
        axes[1].plot(self.time_axis, self.detrended_signal, 'r-', linewidth=0.8)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Detrended Intensity')
        axes[1].set_title('Detrended Intensity Signal')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved {save_path}")
        plt.close()
    
    def plot_fft(self, max_freq=2000, save_path='fft_spectrum.png', highlight_freq=None):
        
        if self.frequencies is None or self.fft_magnitude is None:
            raise ValueError("Must compute FFT first")
        
        print(f"Creating FFT plot...")
        freq_mask = self.frequencies <= max_freq
        freq_plot = self.frequencies[freq_mask]
        mag_plot = self.fft_magnitude[freq_mask]
        
        plt.figure(figsize=(12, 6))
        plt.plot(freq_plot, mag_plot, 'b-', linewidth=1)
        
        if highlight_freq is not None:
            freq_window = 20  # Hz
            near_freq = np.abs(freq_plot - highlight_freq) < freq_window
            if np.any(near_freq):
                peak_idx = np.argmax(mag_plot[near_freq])
                peak_freq = freq_plot[near_freq][peak_idx]
                peak_mag = mag_plot[near_freq][peak_idx]
                plt.axvline(highlight_freq, color='r', linestyle='--', 
                           alpha=0.5, label=f'Expected: {highlight_freq} Hz')
                plt.plot(peak_freq, peak_mag, 'ro', markersize=8,
                        label=f'Peak: {peak_freq:.1f} Hz')
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Frequency Spectrum (FFT)')
        plt.grid(True, alpha=0.3)
        if highlight_freq is not None:
            plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved {save_path}")
        plt.close()
    
    def find_dominant_frequencies(self, n_peaks=5, min_freq=50):
        
        if self.frequencies is None or self.fft_magnitude is None:
            raise ValueError("Must compute FFT first")
        
        freq_mask = self.frequencies >= min_freq
        filtered_freqs = self.frequencies[freq_mask]
        filtered_mags = self.fft_magnitude[freq_mask]
        
        if len(filtered_mags) == 0:
            print("\nNo frequency data in the specified range")
            return filtered_freqs, filtered_mags
        
        max_mag = np.max(filtered_mags)
        if max_mag > 0:
            threshold = max_mag * 0.1
        else:
            print("\nNo signal detected (all magnitudes are zero)")
            return filtered_freqs, filtered_mags
            
        peaks, properties = signal.find_peaks(filtered_mags, height=threshold)
        
        # Sort by magnitude
        if len(peaks) > 0:
            peak_mags = filtered_mags[peaks]
            sorted_idx = np.argsort(peak_mags)[::-1]
            top_peaks = peaks[sorted_idx[:n_peaks]]
            
            print(f"\nTop {min(n_peaks, len(top_peaks))} dominant frequencies:")
            for i, peak_idx in enumerate(top_peaks, 1):
                freq = filtered_freqs[peak_idx]
                mag = filtered_mags[peak_idx]
                print(f"  {i}. {freq:.2f} Hz (magnitude: {mag:.2f})")
        else:
            print("\nNo significant peaks found (signal may be mostly noise)")
        
        return filtered_freqs, filtered_mags


def analyze_video(video_path, fps=60, expected_freq=None, output_prefix=None, auto_roi=False, roi_size=40, downscale_factor=2):
    
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(video_path))[0]
    
    analyzer = OpticalSideChannelAnalyzer(video_path, fps)
    first_frame = analyzer.load_video(downscale_factor=downscale_factor)
    
    if auto_roi:
        roi = analyzer.find_brightest_roi(first_frame, roi_size)
    else:
        roi = analyzer.select_roi_interactive(first_frame)
    
    print("Extracting intensity signal...")
    analyzer.extract_intensity_signal(roi, visualize=True)
    
    print("Detrending signal...")
    analyzer.detrend_signal()
    
    print("Computing FFT...")
    analyzer.compute_fft()
    
    analyzer.plot_intensity_signals(f'{output_prefix}_intensity.png')
    analyzer.plot_fft(max_freq=2000, save_path=f'{output_prefix}_fft.png',
                      highlight_freq=expected_freq)
    
    analyzer.find_dominant_frequencies(n_peaks=5)
    print(f"✓ Analysis complete for {video_path}\n")
    return analyzer


if __name__ == "__main__":
    print("Optical Side-Channel Attack Analysis")
    print("=" * 50)
    
    print("\n### Analyzing CONTROL (silence) ###")
    control = analyze_video(
        "control_silence.mov",
        fps=60,
        output_prefix="control",
        auto_roi=True  
    )
    
    print("\n### Analyzing BASELINE (440 Hz) ###")
    baseline = analyze_video(
        "baseline_440hz.mov",
        fps=60,
        expected_freq=440,
        output_prefix="baseline",
        auto_roi=True
    )
    
    print("\n### Analyzing SPEECH ###")
    speech = analyze_video(
        "speech_newsanchor.mov",
        fps=60,
        output_prefix="speech",
        auto_roi=True
    )
    
    print("\n" + "=" * 50)
    print("Analysis complete! Check the output PNG files.")