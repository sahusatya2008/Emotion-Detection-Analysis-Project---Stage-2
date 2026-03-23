# COMPLETE CODE MANUAL FOR FINAL EMOTION DETECTION PROGRAM BY SATYA N SAHU

## Overview

This manual provides a comprehensive, line-by-line explanation of the `FINAL EMOTION DETECTION PROGRAM BY SATYA N SAHU.py` script. The program implements an ultra-precise emotion detection and paranormal activity recognition system using computer vision and machine learning techniques.

**Total Lines**: 1282
**Main Class**: `UltraPreciseEmotionParanormalRecognition`
**Key Technologies**: OpenCV, DeepFace, TensorFlow, SciPy, Matplotlib

---

## SECTION 1: IMPORTS AND SETUP (Lines 1-23)

```python
import os
```

**Purpose**: Import the `os` module for operating system interactions. Used for file system operations and environment variable access.

```python
# macOS CV2 fixes
import cv2
cv2.startWindowThread()  # Start window thread to prevent crashes
import numpy as np
from deepface import DeepFace
import time
import pandas as pd
```

**Explanations**:
- **cv2 (OpenCV)**: Core computer vision library for image processing, camera access, and video analysis
- **cv2.startWindowThread()**: macOS-specific fix to prevent OpenCV window crashes by starting a separate window thread
- **numpy as np**: Fundamental package for scientific computing with arrays and matrices
- **DeepFace**: Advanced facial analysis library for emotion detection and face recognition
- **time**: Standard library for time-related functions (delays, timestamps)
- **pandas as pd**: Data manipulation and analysis library (used for data structures)

```python
# Set matplotlib backend for compatibility
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless environments
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque
import threading
from scipy import stats, signal, fft
import warnings
import gc  # Garbage collection for memory management

warnings.filterwarnings('ignore')
```

**Explanations**:
- **matplotlib**: Plotting library for creating real-time charts
- **matplotlib.use('Agg')**: Set non-GUI backend to prevent display issues in headless environments
- **FigureCanvasAgg**: Backend for rendering matplotlib figures to images
- **collections.deque**: Double-ended queue for efficient append/pop operations (used for buffers)
- **threading**: Library for concurrent execution (multi-threaded processing)
- **scipy.stats, signal, fft**: Scientific computing for statistical analysis, signal processing, and Fourier transforms
- **warnings.filterwarnings('ignore')**: Suppress warning messages for cleaner output
- **gc**: Garbage collection module for manual memory management

---

## SECTION 2: CLASS DEFINITION AND INITIALIZATION (Lines 25-142)

### Class Declaration (Line 25)

```python
class UltraPreciseEmotionParanormalRecognition:
```

**Purpose**: Defines the main class that encapsulates all functionality for emotion and paranormal detection.

### Constructor (__init__ method) (Lines 26-142)

```python
def __init__(self):
    # Initialize emotion history tracking with enhanced precision
    self.emotion_history = deque(maxlen=30)  # Reduced from 50 to save memory
    self.emotion_confidence_history = deque(maxlen=30)  # Reduced from 50
    self.current_emotion = "Neutral"
    self.confidence = 0
    self.engagement_score = 0
    self.mood_trend = "Stable"
    self.start_time = time.time()
```

**Explanations**:
- **emotion_history**: Buffer storing recent emotion detections (30 entries max for memory efficiency)
- **emotion_confidence_history**: Parallel buffer for confidence scores
- **current_emotion/confidence**: Current state variables
- **engagement_score**: Calculated engagement level (0-100)
- **mood_trend**: Temporal trend analysis ("Improving", "Declining", "Stable")
- **start_time**: Program start timestamp for duration calculations

```python
# Paranormal detection parameters
self.anomaly_detected = False
self.paranormal_confidence = 0
self.invisible_presence = False
self.energy_fluctuations = 0
self.temporal_anomalies = 0
```

**Paranormal Detection Variables**:
- **anomaly_detected**: Boolean flag for detected anomalies
- **paranormal_confidence**: Confidence score (0-100) for paranormal activity
- **invisible_presence**: Detection of invisible entities
- **energy_fluctuations**: Measured energy variation levels
- **temporal_anomalies**: Temporal inconsistency scores

```python
# Analysis timing
self.last_analysis_time = 0
self.analysis_interval = 0.5  # Increased from 0.3 to reduce CPU load
self.last_enhanced_analysis_time = 0
self.enhanced_analysis_interval = 3.0  # Increased from 2.0 to reduce CPU load
```

**Timing Controls**:
- **analysis_interval**: Time between emotion analysis cycles (0.5 seconds)
- **enhanced_analysis_interval**: Interval for advanced analysis (3.0 seconds)
- Optimized for performance by reducing analysis frequency

### Camera Initialization (Lines 50-83)

```python
# Initialize the webcam with optimized settings
self.cap = cv2.VideoCapture(0)
self.camera_available = True
self.camera_index = 0
```

**Camera Setup**:
- **cv2.VideoCapture(0)**: Open default camera (index 0)
- **camera_available**: Flag indicating camera status
- **camera_index**: Current camera index being used

```python
if not self.cap.isOpened() or not self.cap.read()[0]:
    print("Primary camera not available, trying alternatives...")
    # Try different camera indices if default fails
    for i in range(1, 4):
        self.cap = cv2.VideoCapture(i)
        if self.cap.isOpened() and self.cap.read()[0]:
            print(f"Using camera index {i}")
            self.camera_index = i
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            break
    if not self.cap.isOpened() or not self.cap.read()[0]:
        print("No camera available, running in demo mode")
        self.camera_available = False
        # Create a dummy frame for demo mode
        self.demo_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(self.demo_frame, "NO CAMERA - DEMO MODE", (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
```

**Camera Fallback Logic**:
- Tries camera indices 0-3 if primary fails
- Sets resolution to 640x480 for optimal performance
- Falls back to demo mode with synthetic frame if no camera available
- Displays "NO CAMERA - DEMO MODE" message on demo frame

```python
if self.camera_available:
    # Set optimized resolution for better performance
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    self.cap.set(cv2.CAP_PROP_FPS, 30)
    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    except:
        pass  # Some cameras don't support these properties
```

**Camera Optimization**:
- Sets resolution, FPS, and buffer size for consistent performance
- Enables autofocus and auto-exposure if supported
- Gracefully handles cameras without these features

### Emotion Color Mapping (Lines 85-94)

```python
# Enhanced emotion colors with gradients for confidence levels
self.emotion_colors = {
    'happy': [(0, 255, 0), (0, 200, 0), (0, 150, 0)],  # Green gradient
    'sad': [(255, 0, 0), (200, 0, 0), (150, 0, 0)],  # Blue gradient
    'angry': [(0, 0, 255), (0, 0, 200), (0, 0, 150)],  # Red gradient
    'surprise': [(255, 255, 0), (200, 200, 0), (150, 150, 0)],  # Cyan gradient
    'fear': [(255, 0, 255), (200, 0, 200), (150, 0, 150)],  # Magenta gradient
    'disgust': [(0, 255, 255), (0, 200, 200), (0, 150, 150)],  # Yellow gradient
    'neutral': [(255, 255, 255), (200, 200, 200), (150, 150, 150)]  # White gradient
}
```

**Color Coding System**:
- Each emotion has 3 color intensities: high, medium, low confidence
- Used for visual feedback in bounding boxes and text overlays
- BGR format for OpenCV compatibility

### Buffer Management (Lines 96-122)

```python
# Advanced background subtractor for precise motion detection
self.fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=16,
                                              detectShadows=True)  # Reduced history
```

**Background Subtraction**:
- MOG2 algorithm for foreground/background separation
- History=50 frames for background model
- varThreshold=16 for sensitivity
- detectShadows=True for better accuracy

```python
# Multi-layer frame analysis buffers with reduced sizes
self.frame_buffer = deque(maxlen=10)  # Reduced from 20
self.motion_history = deque(maxlen=20)  # Reduced from 30
self.energy_readings = deque(maxlen=20)  # Reduced from 30
self.spectral_history = deque(maxlen=15)  # Reduced from 25
```

**Analysis Buffers**:
- **frame_buffer**: Recent frames for optical flow analysis
- **motion_history**: Motion level readings over time
- **energy_readings**: Energy fluctuation measurements
- **spectral_history**: Spectral analysis data
- All optimized for memory usage

### Pre-allocated Resources (Lines 107-121)

```python
# Pre-allocated arrays for performance
self.engagement_chart = None
self.paranormal_chart = None
self.emotion_distribution_chart = None
self.last_chart_update = 0
self.chart_update_interval = 1.5  # Increased from 0.8 to reduce CPU load

# Threading for parallel processing
self.analysis_results = {}
self.enhanced_analysis_results = {}
self.analysis_lock = threading.Lock()
self.enhanced_analysis_lock = threading.Lock()
self.analysis_thread = None
self.enhanced_analysis_thread = None
self.latest_frame = None
self.running = True  # Flag to control thread execution
```

**Performance Optimizations**:
- Pre-allocated chart objects to prevent memory allocation during runtime
- Threading locks for safe concurrent access to shared data
- Thread objects for parallel emotion and enhanced analysis
- Running flag for controlled thread termination

### Calibration and Model Initialization (Lines 123-142)

```python
# Calibration parameters
self.calibration_complete = False
self.calibration_start_time = 0
self.calibration_duration = 3.0  # 3 seconds calibration

# Advanced statistical models
self.emotion_model = self.initialize_emotion_model()
self.anomaly_model = self.initialize_anomaly_model()

# Initialize with default values to prevent empty errors
self.motion_mean = 100
self.motion_std = 50
self.energy_mean = 0.5
self.energy_std = 0.2
self.spectral_mean = 1000
self.spectral_std = 500

print("Loading Ultra-Precise Advanced Models...")
self.setup_advanced_detection()
print("System initialized successfully! Calibrating...")
```

**Initialization Sequence**:
- Sets calibration parameters (3-second duration)
- Initializes emotion and anomaly models
- Sets default statistical baselines
- Calls setup method and begins calibration

---

## SECTION 3: MODEL INITIALIZATION METHODS (Lines 144-165)

```python
def initialize_emotion_model(self):
    """Initialize advanced emotion recognition model"""
    return {
        'baseline_emotions': {},
        'temporal_patterns': deque(maxlen=50),  # Reduced from 100
        'confidence_thresholds': {
            'high': 85,
            'medium': 60,
            'low': 40
        }
    }
```

**Emotion Model Structure**:
- **baseline_emotions**: Dictionary for storing baseline emotion states
- **temporal_patterns**: Buffer for emotion pattern analysis
- **confidence_thresholds**: Thresholds for confidence level classification

```python
def initialize_anomaly_model(self):
    """Initialize advanced anomaly detection model"""
    return {
        'motion_baseline': 0,
        'energy_baseline': 0,
        'spectral_baseline': 0,
        'temporal_consistency': deque(maxlen=30),  # Reduced from 50
        'anomaly_patterns': []
    }
```

**Anomaly Model Structure**:
- **motion/energy/spectral_baseline**: Baseline values for different metrics
- **temporal_consistency**: Buffer for consistency analysis
- **anomaly_patterns**: List to store detected anomaly patterns

```python
def setup_advanced_detection(self):
    """Setup advanced models for ultra-precise detection"""
    # Start calibration
    self.calibration_start_time = time.time()
```

**Setup Method**: Initializes calibration timer to begin environmental adaptation.

---

## SECTION 4: CAMERA AND FRAME MANAGEMENT (Lines 171-198)

```python
def get_frame(self):
    """Safely get a frame from camera or use demo frame"""
    if not self.camera_available:
        return True, self.demo_frame.copy()

    ret, frame = self.cap.read()
    if not ret:
        # Try to reinitialize camera
        print("Camera error, trying to reinitialize...")
        self.cap.release()
        time.sleep(0.5)
        self.cap = cv2.VideoCapture(self.camera_index)
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not ret:
            print("Camera still not working, switching to demo mode")
            self.camera_available = False
            self.demo_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(self.demo_frame, "CAMERA ERROR - DEMO MODE", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return True, self.demo_frame.copy()

    return ret, frame
```

**Frame Acquisition Logic**:
1. Check if camera is available, return demo frame if not
2. Attempt to read frame from camera
3. If failed, try to reinitialize camera
4. If still failed, switch to demo mode permanently
5. Return success flag and frame

---

## SECTION 5: CALIBRATION SYSTEM (Lines 199-237)

```python
def calibrate_system(self, frame):
    """Calibrate the system for current environment"""
    current_time = time.time()
    elapsed = current_time - self.calibration_start_time

    if elapsed < self.calibration_duration:
        # Collect baseline data during calibration
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate motion level
        fgmask = self.fgbg.apply(frame)
        motion_level = np.sum(fgmask) / (frame.shape[0] * frame.shape[1])
        self.motion_history.append(motion_level)

        # Calculate energy level
        energy_level = self.calculate_energy_level(frame)
        self.energy_readings.append(energy_level)

        # Calculate spectral energy
        spectral_energy = self.calculate_spectral_energy(gray)
        self.spectral_history.append(spectral_energy)

        progress = int((elapsed / self.calibration_duration) * 100)
        return f"Calibrating: {progress}% complete"

    else:
        # Finalize calibration
        self.calibration_complete = True
        if self.motion_history:
            self.motion_mean = np.mean(list(self.motion_history))
            self.motion_std = max(np.std(list(self.motion_history)), 0.001)
        if self.energy_readings:
            self.energy_mean = np.mean(list(self.energy_readings))
            self.energy_std = max(np.std(list(self.energy_readings)), 0.001)
        if self.spectral_history:
            self.spectral_mean = np.mean(list(self.spectral_history))
            self.spectral_std = max(np.std(list(self.spectral_history)), 0.001)

        return "Calibration complete! System ready."
```

**Calibration Process**:
1. Runs for 3 seconds collecting baseline data
2. Calculates motion, energy, and spectral baselines
3. Computes mean and standard deviation for each metric
4. Returns progress or completion message

---

## SECTION 6: ENERGY AND SPECTRAL ANALYSIS (Lines 239-272)

```python
def calculate_energy_level(self, frame):
    """Calculate energy level from frame with enhanced precision"""
    # Convert to LAB color space for better perceptual accuracy
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # Calculate energy using multiple methods
    energy_rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
    energy_std = np.std(l_channel)
    energy_gradient = np.mean(np.abs(cv2.Sobel(l_channel, cv2.CV_64F, 1, 1, ksize=3)))

    # Combined energy metric
    combined_energy = (energy_rms + energy_std + energy_gradient) / 3
    return combined_energy
```

**Multi-Method Energy Calculation**:
- **RMS Energy**: Root mean square of pixel intensities
- **Standard Deviation**: Variation in luminance channel
- **Gradient Energy**: Edge detection using Sobel operator
- **Combined**: Average of three methods for robust measurement

```python
def calculate_spectral_energy(self, gray_frame):
    """Calculate spectral energy using FFT"""
    try:
        # Apply window function to reduce spectral leakage
        window = np.hanning(gray_frame.shape[0])[:, None] * np.hanning(gray_frame.shape[1])
        windowed_frame = gray_frame * window

        # Compute 2D FFT
        fft_frame = fft.fft2(windowed_frame.astype(np.float32))
        fft_shifted = fft.fftshift(fft_frame)

        # Calculate magnitude spectrum
        magnitude_spectrum = np.abs(fft_shifted)

        # Calculate spectral energy
        spectral_energy = np.sum(magnitude_spectrum ** 2) / (gray_frame.size)
        return spectral_energy
    except:
        return 1000  # Default value if FFT fails
```

**Spectral Analysis Process**:
1. Apply Hanning window to reduce spectral leakage
2. Compute 2D Fast Fourier Transform
3. Shift zero frequency to center
4. Calculate magnitude spectrum
5. Sum squared magnitudes and normalize
6. Return default value if FFT computation fails

---

## SECTION 7: EMOTION ANALYSIS (Lines 274-348)

```python
def analyze_emotion_threaded(self, frame):
    """Run ultra-precise emotion analysis in a separate thread"""
```

**Threaded Emotion Analysis**: Runs emotion detection in background thread for performance.

```python
def analyze():
    try:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Enhanced face detection with multiple backends for accuracy
        try:
            analysis = DeepFace.analyze(
                rgb_frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True,
                detector_backend='opencv'  # More reliable backend
            )
        except Exception as e:
            print(f"DeepFace analysis failed: {e}")
            # Fallback to simple analysis
            analysis = {
                'dominant_emotion': 'neutral',
                'emotion': {'neutral': 100, 'happy': 0, 'sad': 0, 'angry': 0, 'surprise': 0, 'fear': 0,
                            'disgust': 0},
                'region': {}
            }

        if isinstance(analysis, list):
            analysis = analysis[0]

        # Get the dominant emotion and its confidence with enhanced precision
        emotion = analysis.get('dominant_emotion', 'neutral')
        emotions_dict = analysis.get('emotion', {})
        confidence = emotions_dict.get(emotion, 0)

        # Calculate confidence boost based on secondary emotions
        sorted_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_emotions) > 1:
            confidence_boost = max(0, (sorted_emotions[0][1] - sorted_emotions[1][1]) / 2)
            confidence = min(100, confidence + confidence_boost)

        # Extract additional details with enhanced precision
        details = {
            'emotions': emotions_dict,
            'region': analysis.get('region', {})
        }

        with self.analysis_lock:
            self.analysis_results = {
                'emotion': emotion,
                'confidence': confidence,
                'details': details,
                'timestamp': time.time()
            }

    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        with self.analysis_lock:
            self.analysis_results = {
                'emotion': "neutral",
                'confidence': 0,
                'details': {'emotions': {}},
                'timestamp': time.time()
            }
    finally:
        # Force garbage collection to prevent memory leaks
        gc.collect()

# Start analysis in a thread if not already running
if (self.analysis_thread is None or not self.analysis_thread.is_alive()) and self.running:
    self.analysis_thread = threading.Thread(target=analyze)
    self.analysis_thread.daemon = True
    self.analysis_thread.start()
```

**Emotion Analysis Flow**:
1. Convert frame to RGB for DeepFace compatibility
2. Run DeepFace analysis with OpenCV detector backend
3. Extract dominant emotion and confidence
4. Apply confidence boost based on emotion distribution
5. Store results in thread-safe manner
6. Handle exceptions with fallback values
7. Start new thread if previous completed

---

## SECTION 8: ENHANCED ANALYSIS (Lines 349-392)

```python
def enhanced_analysis_threaded(self, frame):
    """Run enhanced analysis for better precision"""
```

Similar threading approach for advanced analysis features.

```python
def analyze():
    try:
        # Perform multi-layer analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Advanced motion analysis
        motion_analysis = self.advanced_motion_analysis(frame)

        # Spectral analysis
        spectral_analysis = self.spectral_analysis(gray)

        # Temporal consistency analysis
        temporal_analysis = self.temporal_consistency_analysis()

        with self.enhanced_analysis_lock:
            self.enhanced_analysis_results = {
                'motion': motion_analysis,
                'spectral': spectral_analysis,
                'temporal': temporal_analysis,
                'timestamp': time.time()
            }

    except Exception as e:
        print(f"Error in enhanced analysis: {e}")
        with self.enhanced_analysis_lock:
            self.enhanced_analysis_results = {
                'motion': {'motion_consistency': 0},
                'spectral': {'spectral_balance': 0, 'high_freq': 0},
                'temporal': {'abnormality': 0},
                'timestamp': time.time()
            }
    finally:
        # Force garbage collection to prevent memory leaks
        gc.collect()

# Start enhanced analysis in a separate thread
if (self.enhanced_analysis_thread is None or not self.enhanced_analysis_thread.is_alive()) and self.running:
    self.enhanced_analysis_thread = threading.Thread(target=analyze)
    self.enhanced_analysis_thread.daemon = True
    self.enhanced_analysis_thread.start()
```

**Enhanced Analysis Components**:
- **Motion Analysis**: Optical flow and motion consistency
- **Spectral Analysis**: Frequency domain analysis
- **Temporal Analysis**: Consistency over time

---

## SECTION 9: ADVANCED MOTION ANALYSIS (Lines 393-427)

```python
def advanced_motion_analysis(self, frame):
    """Perform advanced motion analysis with multiple techniques"""
    try:
        # Optical flow for precise motion detection
        if len(self.frame_buffer) > 1:
            prev_frame = self.frame_buffer[-2]
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Calculate magnitude and angle of flow vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            motion_metrics = {
                'mean_magnitude': np.mean(magnitude),
                'std_magnitude': np.std(magnitude),
                'mean_angle': np.mean(angle),
                'motion_consistency': np.std(angle) if np.std(angle) > 0 else 0.001
            }
        else:
            motion_metrics = {
                'mean_magnitude': 0,
                'std_magnitude': 0,
                'mean_angle': 0,
                'motion_consistency': 0
            }
    except:
        motion_metrics = {
            'mean_magnitude': 0,
            'std_magnitude': 0,
            'mean_angle': 0,
            'motion_consistency': 0
        }

    return motion_metrics
```

**Optical Flow Analysis**:
1. Convert frames to grayscale
2. Calculate dense optical flow using Farneback algorithm
3. Convert to polar coordinates (magnitude and angle)
4. Compute statistical metrics for motion characterization

---

## SECTION 10: SPECTRAL ANALYSIS (Lines 431-481)

```python
def spectral_analysis(self, gray_frame):
    """Perform advanced spectral analysis"""
    try:
        # Compute 2D FFT
        fft_frame = fft.fft2(gray_frame.astype(np.float32))
        fft_shifted = fft.fftshift(fft_frame)

        # Calculate magnitude spectrum
        magnitude_spectrum = np.abs(fft_shifted)

        # Analyze spectral distribution
        height, width = gray_frame.shape
        center_y, center_x = height // 2, width // 2

        # Create circular masks for different frequency bands
        y, x = np.ogrid[:height, :width]
        distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        # Low frequencies (close to center)
        low_freq_mask = distances < min(center_x, center_y) * 0.3
        # Medium frequencies
        mid_freq_mask = (distances >= min(center_x, center_y) * 0.3) & (distances < min(center_x, center_y) * 0.6)
        # High frequencies (edges)
        high_freq_mask = distances >= min(center_x, center_y) * 0.6

        # Calculate energy in each frequency band
        low_freq_energy = np.sum(magnitude_spectrum[low_freq_mask])
        mid_freq_energy = np.sum(magnitude_spectrum[mid_freq_mask])
        high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])
        total_energy = low_freq_energy + mid_freq_energy + high_freq_energy

        # Normalize energies
        if total_energy > 0:
            low_freq_energy /= total_energy
            mid_freq_energy /= total_energy
            high_freq_energy /= total_energy

        return {
            'low_freq': low_freq_energy,
            'mid_freq': mid_freq_energy,
            'high_freq': high_freq_energy,
            'spectral_balance': high_freq_energy / (low_freq_energy + 1e-10)
        }
    except Exception as e:
        print(f"[ERROR] Spectral analysis failed: {e}")
        return {
            'low_freq': 0.5,
            'mid_freq': 0.3,
            'high_freq': 0.2,
            'spectral_balance': 0.4
        }
```

**Spectral Analysis Process**:
1. Compute 2D FFT of grayscale frame
2. Create circular masks for frequency bands
3. Calculate energy in low, medium, and high frequency regions
4. Normalize energies and compute spectral balance ratio
5. Return default values if analysis fails

---

## SECTION 11: TEMPORAL CONSISTENCY ANALYSIS (Lines 483-508)

```python
def temporal_consistency_analysis(self):
    """Analyze temporal patterns for anomalies"""
    if len(self.emotion_confidence_history) < 5:
        return {'consistency': 0, 'stability': 0, 'abnormality': 0}

    try:
        confidences = list(self.emotion_confidence_history)

        # Calculate rate of change
        changes = np.diff(confidences)
        mean_change = np.mean(np.abs(changes))

        # Calculate stability metric
        stability = 100 - min(100, mean_change * 10)

        # Check for abnormal patterns
        abnormal_changes = np.sum(np.abs(changes) > 20)  # Changes greater than 20%
        abnormality_score = min(100, (abnormal_changes / len(changes)) * 200)

        return {
            'consistency': 100 - mean_change,
            'stability': stability,
            'abnormality': abnormality_score
        }
    except:
        return {'consistency': 0, 'stability': 0, 'abnormality': 0}
```

**Temporal Analysis Metrics**:
- **Consistency**: Inverse of average change rate
- **Stability**: Measure of confidence stability over time
- **Abnormality**: Detection of sudden confidence changes

---

## SECTION 12: PARANORMAL DETECTION (Lines 510-607)

```python
def detect_paranormal_activity(self, frame, motion_analysis, spectral_analysis, temporal_analysis):
    """
    Ultra-precise method to detect paranormal activity and invisible presence
    """
    if not self.calibration_complete:
        return {
            'paranormal_confidence': 0,
            'invisible_detected': False,
            'energy_fluctuations': 0,
            'temporal_anomalies': 0
        }
```

**Calibration Check**: Returns zero values if system not yet calibrated.

```python
try:
    # Convert to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction
    fgmask = self.fgbg.apply(frame)

    # Calculate motion level
    motion_level = np.sum(fgmask) / (frame.shape[0] * frame.shape[1])
    self.motion_history.append(motion_level)

    # Calculate energy level
    energy_level = self.calculate_energy_level(frame)
    self.energy_readings.append(energy_level)

    # Calculate spectral energy
    spectral_energy = self.calculate_spectral_energy(gray)
    self.spectral_history.append(spectral_energy)

    # Calculate multiple anomaly scores
    motion_anomaly = self.calculate_enhanced_anomaly(list(self.motion_history), motion_level, self.motion_mean,
                                                      self.motion_std)
    energy_anomaly = self.calculate_enhanced_anomaly(list(self.energy_readings), energy_level, self.energy_mean,
                                                      self.energy_std)
    spectral_anomaly = self.calculate_enhanced_anomaly(list(self.spectral_history), spectral_energy,
                                                        self.spectral_mean, self.spectral_std)
```

**Anomaly Calculation**:
- Computes motion, energy, and spectral anomaly scores using statistical methods
- Updates rolling history buffers

```python
# Analyze motion patterns from optical flow
motion_consistency_anomaly = 0
if motion_analysis['motion_consistency'] > 0:
    motion_consistency_anomaly = min(100, motion_analysis['motion_consistency'] * 50)

# Analyze spectral patterns
spectral_balance_anomaly = 0
if spectral_analysis['spectral_balance'] > 2.0:  # Unusually high frequency content
    spectral_balance_anomaly = min(100, (spectral_analysis['spectral_balance'] - 2.0) * 25)

# Detect invisible presence (multiple criteria)
contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
large_contours = [c for c in contours if cv2.contourArea(c) > 300]

# Multiple criteria for invisible presence detection
motion_without_contours = motion_level > 500 and len(large_contours) == 0
high_freq_without_motion = spectral_analysis['high_freq'] > 0.4 and motion_level < 100
energy_spike_without_source = energy_anomaly > 70 and motion_anomaly < 30

invisible_detected = motion_without_contours or high_freq_without_motion or energy_spike_without_source
```

**Invisible Presence Detection Criteria**:
- Motion detected without visible contours
- High frequency content without corresponding motion
- Energy spikes without motion anomalies

```python
# Calculate comprehensive paranormal confidence
paranormal_confidence = min(100, max(0, (
        motion_anomaly * 0.15 +
        energy_anomaly * 0.20 +
        spectral_anomaly * 0.15 +
        motion_consistency_anomaly * 0.10 +
        spectral_balance_anomaly * 0.10 +
        temporal_analysis['abnormality'] * 0.10 +
        (100 if invisible_detected else 0) * 0.20
)))

# Calculate energy fluctuations
energy_fluctuations = min(100, energy_anomaly * 1.5)

# Calculate temporal anomalies
temporal_anomalies = min(100, temporal_analysis['abnormality'] * 1.2)

# Update status
self.invisible_presence = invisible_detected
self.paranormal_confidence = paranormal_confidence
self.energy_fluctuations = energy_fluctuations
self.temporal_anomalies = temporal_anomalies
self.anomaly_detected = paranormal_confidence > 75  # Higher threshold for precision

return {
    'paranormal_confidence': paranormal_confidence,
    'invisible_detected': invisible_detected,
    'energy_fluctuations': energy_fluctuations,
    'temporal_anomalies': temporal_anomalies
}
```

**Paranormal Confidence Calculation**:
- Weighted combination of all anomaly metrics
- Special weight for invisible presence detection
- Updates instance variables for display

---

## SECTION 13: ANOMALY CALCULATION (Lines 609-647)

```python
def calculate_enhanced_anomaly(self, history, current_value, baseline_mean, baseline_std):
    """
    Calculate enhanced anomaly score with multiple statistical methods
    """
    if len(history) < 5:
        return 0

    try:
        # Z-score anomaly detection
        if baseline_std < 0.001:
            z_score_anomaly = 0
        else:
            z_score = abs((current_value - baseline_mean) / baseline_std)
            z_score_anomaly = min(100, (z_score / 3) * 100)

        # Percentile-based anomaly detection
        sorted_history = sorted(history)
        percentile = stats.percentileofscore(sorted_history, current_value)
        percentile_anomaly = min(100, abs(percentile - 50) * 2)

        # Moving average anomaly detection
        window_size = min(10, len(history))
        moving_avg = np.mean(history[-window_size:])
        moving_std = np.std(history[-window_size:])

        if moving_std < 0.001:
            moving_anomaly = 0
        else:
            moving_z = abs((current_value - moving_avg) / moving_std)
            moving_anomaly = min(100, (moving_z / 3) * 100)

        # Combined anomaly score (weighted average)
        combined_anomaly = (z_score_anomaly * 0.4 +
                            percentile_anomaly * 0.3 +
                            moving_anomaly * 0.3)

        return combined_anomaly
    except:
        return 0
```

**Multi-Method Anomaly Detection**:
1. **Z-score**: Standard deviation from mean
2. **Percentile**: Position in historical distribution
3. **Moving Average**: Recent trend analysis
4. **Combined**: Weighted average of all methods

---

## SECTION 14: ENGAGEMENT CALCULATION (Lines 649-693)

```python
def calculate_engagement(self, emotion, confidence, emotions_dict):
    """
    Calculate advanced engagement score based on multiple factors
    """
    # Emotion weights for engagement scoring
    emotion_weights = {
        'happy': 1.0,
        'surprise': 0.9,
        'neutral': 0.6,
        'sad': 0.3,
        'fear': 0.2,
        'angry': 0.2,
        'disgust': 0.2,
        'unknown': 0.0
    }

    # Calculate base engagement score
    base_engagement = emotion_weights.get(emotion, 0.0) * 100 * (confidence / 100)

    # Calculate engagement boost from positive secondary emotions
    sorted_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)
    positive_boost = 0

    if len(sorted_emotions) > 1:
        # Boost if secondary emotions are also positive
        secondary_emotion, secondary_confidence = sorted_emotions[1]
        if secondary_emotion in ['happy', 'surprise'] and secondary_confidence > 20:
            positive_boost = min(15, secondary_confidence / 2)

    # Calculate engagement penalty from negative secondary emotions
    negative_penalty = 0
    for emo, conf in sorted_emotions[1:4]:  # Check top 3 secondary emotions
        if emo in ['sad', 'fear', 'angry', 'disgust'] and conf > 25:
            negative_penalty += min(20, conf / 2)

    # Apply confidence-based scaling
    confidence_factor = confidence / 100

    # Final engagement score
    engagement = max(0, min(100, base_engagement + positive_boost - negative_penalty))

    # Apply non-linear scaling for better differentiation
    engagement = engagement ** 1.1 / (100 ** 0.1)

    return engagement
```

**Engagement Scoring Algorithm**:
- Base score from primary emotion weight and confidence
- Positive boost for complementary secondary emotions
- Negative penalty for conflicting secondary emotions
- Non-linear scaling for better differentiation

---

## SECTION 15: MOOD TREND ANALYSIS (Lines 695-730)

```python
def calculate_mood_trend(self):
    """Calculate mood trend based on emotion history"""
    if len(self.emotion_history) < 5:
        return "Insufficient data"

    try:
        # Get recent emotions
        recent_emotions = list(self.emotion_history)[-10:]

        # Calculate mood score for each emotion
        mood_scores = {
            'happy': 1.0,
            'surprise': 0.8,
            'neutral': 0.5,
            'sad': -0.5,
            'fear': -0.7,
            'angry': -0.8,
            'disgust': -0.8,
            'unknown': 0.0
        }

        scores = [mood_scores.get(emo.get('emotion', 'neutral'), 0) for emo in recent_emotions]

        # Calculate trend using linear regression
        x = np.arange(len(scores))
        slope, _, _, _, _ = stats.linregress(x, scores)

        # Determine trend direction
        if slope > 0.05:
            return "Improving"
        elif slope < -0.05:
            return "Declining"
        else:
            return "Stable"
    except:
        return "Stable"
```

**Mood Trend Calculation**:
1. Requires minimum 5 emotion history entries
2. Assigns numerical mood scores to emotions (-1 to +1 scale)
3. Uses linear regression on recent scores to determine slope
4. Classifies trend as Improving, Declining, or Stable based on slope thresholds

---

## SECTION 16: CHART GENERATION METHODS (Lines 732-891)

### Engagement Chart (Lines 732-777)

```python
def create_engagement_chart(self):
    """Create a real-time engagement chart with enhanced visuals"""
    if len(self.emotion_history) < 2:
        return None

    try:
        # Create a figure with better styling
        fig, ax = plt.subplots(figsize=(5, 2), dpi=80, facecolor='#2E2E2E')
        ax.set_facecolor('#2E2E2E')

        # Extract engagement scores from history
        engagement_scores = [entry.get('engagement', 0) for entry in self.emotion_history]
        timestamps = [entry.get('timestamp', 0) - self.start_time for entry in self.emotion_history]

        # Plot the data with enhanced styling
        ax.plot(timestamps, engagement_scores, 'b-', linewidth=2, alpha=0.8)
        ax.fill_between(timestamps, engagement_scores, alpha=0.3, color='blue')

        ax.set_ylim(0, 100)
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('Engagement (%)', color='white')
        ax.set_title('Real-time Engagement', color='white', fontweight='bold')
        ax.grid(True, alpha=0.2, color='white')

        # Set tick colors
        ax.tick_params(colors='white')

        # Set spine colors
        for spine in ax.spines.values():
            spine.set_color('white')

        # Convert plot to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Convert to numpy array
        buf = canvas.buffer_rgba()
        image = np.asarray(buf)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        plt.close(fig)

        return image
    except Exception as e:
        print(f"Error creating engagement chart: {e}")
        return None
```

**Chart Creation Process**:
1. Extracts engagement scores and relative timestamps
2. Creates matplotlib figure with dark theme
3. Plots line graph with area fill
4. Converts matplotlib figure to OpenCV-compatible image
5. Returns None if creation fails

### Paranormal Activity Chart (Lines 779-830)

```python
def create_paranormal_chart(self):
    """Create a chart showing paranormal activity detection with enhanced visuals"""
    if len(self.motion_history) < 3 or len(self.energy_readings) < 3:
        return None

    try:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), dpi=60, facecolor='#2E2E2E')
        ax1.set_facecolor('#2E2E2E')
        ax2.set_facecolor('#2E2E2E')

        # Motion chart
        motion_data = list(self.motion_history)
        timestamps = [i for i in range(len(motion_data))]
        ax1.plot(timestamps, motion_data, 'r-', linewidth=1.5, alpha=0.8)
        ax1.fill_between(timestamps, motion_data, alpha=0.3, color='red')
        ax1.set_ylabel('Motion Level', color='white')
        ax1.set_title('Motion Detection', color='white', fontweight='bold')
        ax1.grid(True, alpha=0.2, color='white')
        ax1.tick_params(colors='white')
        for spine in ax1.spines.values():
            spine.set_color('white')

        # Energy chart
        energy_data = list(self.energy_readings)
        timestamps = [i for i in range(len(energy_data))]
        ax2.plot(timestamps, energy_data, 'c-', linewidth=1.5, alpha=0.8)
        ax2.fill_between(timestamps, energy_data, alpha=0.3, color='cyan')
        ax2.set_ylabel('Energy Level', color='white')
        ax2.set_xlabel('Time (frames)', color='white')
        ax2.set_title('Energy Fluctuations', color='white', fontweight='bold')
        ax2.grid(True, alpha=0.2, color='white')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_color('white')

        plt.tight_layout()

        # Convert plot to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Convert to numpy array
        buf = canvas.buffer_rgba()
        image = np.asarray(buf)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        plt.close(fig)

        return image
    except Exception as e:
        print(f"Error creating paranormal chart: {e}")
        return None
```

**Dual Chart Creation**:
- Top subplot: Motion level over time (red)
- Bottom subplot: Energy level over time (cyan)
- Converts to OpenCV image format for overlay

### Emotion Distribution Chart (Lines 833-891)

```python
def create_emotion_distribution_chart(self, emotions_dict):
    """Create a chart showing emotion distribution"""
    if not emotions_dict:
        return None

    try:
        # Create a figure
        fig, ax = plt.subplots(figsize=(5, 2), dpi=80, facecolor='#2E2E2E')
        ax.set_facecolor('#2E2E2E')

        # Prepare data for plotting
        emotions = list(emotions_dict.keys())
        values = list(emotions_dict.values())

        # Create colormap based on emotion type
        colors = []
        for emotion in emotions:
            if emotion in ['happy', 'surprise']:
                colors.append('green')
            elif emotion == 'neutral':
                colors.append('gray')
            else:
                colors.append('red')

        # Create bar chart
        bars = ax.bar(emotions, values, color=colors, alpha=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', color='white', fontweight='bold')

        ax.set_ylim(0, 100)
        ax.set_ylabel('Confidence (%)', color='white')
        ax.set_title('Emotion Distribution', color='white', fontweight='bold')
        ax.tick_params(axis='x', rotation=45, colors='white')
        ax.tick_params(axis='y', colors='white')

        for spine in ax.spines.values():
            spine.set_color('white')

        plt.tight_layout()

        # Convert plot to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Convert to numpy array
        buf = canvas.buffer_rgba()
        image = np.asarray(buf)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        plt.close(fig)

        return image
    except Exception as e:
        print(f"Error creating emotion distribution chart: {e}")
        return None
```

**Bar Chart Creation**:
- Color-codes bars by emotion type (green for positive, red for negative, gray for neutral)
- Displays confidence percentages on each bar
- Rotates x-axis labels for readability

### Chart Update Method (Lines 893-906)

```python
def update_charts(self, emotions_dict):
    """Update charts less frequently to improve performance"""
    current_time = time.time()
    if current_time - self.last_chart_update > self.chart_update_interval:
        if len(self.emotion_history) >= 5:
            self.engagement_chart = self.create_engagement_chart()

        if len(self.motion_history) >= 3 and len(self.energy_readings) >= 3:
            self.paranormal_chart = self.create_paranormal_chart()

        if emotions_dict:
            self.emotion_distribution_chart = self.create_emotion_distribution_chart(emotions_dict)

        self.last_chart_update = current_time
```

**Performance-Optimized Updates**:
- Updates charts at controlled intervals (1.5 seconds)
- Only recreates charts when sufficient data is available
- Prevents excessive CPU usage during real-time operation

---

## SECTION 17: EMOTION INTERPRETATION (Lines 908-986)

```python
def get_emotion_interpretation(self, emotion, confidence, emotions_dict):
    """Provide detailed interpretation of emotions with enhanced precision"""
    interpretations = {
        'happy': {
            'high': "Extremely positive engagement, clearly enjoying and connected with content",
            'medium': "Positive engagement, showing interest and enjoyment",
            'low': "Mild positive response, somewhat engaged with content"
        },
        'sad': {
            'high': "Strong emotional response, possibly disengaged or affected by content",
            'medium': "Moderate negative response, may need encouragement or content adjustment",
            'low': "Mild disengagement or thoughtful response to content"
        },
        'angry': {
            'high': "Strong negative reaction, likely frustrated or disagreeing with content",
            'medium': "Moderate frustration or disagreement with aspects of content",
            'low': "Mild irritation or critical response to content"
        },
        'surprise': {
            'high': "Strong unexpected reaction, highly engaged with surprising content",
            'medium': "Moderate surprise, engaged with unexpected developments",
            'low': "Mild surprise, noticing unexpected elements in content"
        },
        'fear': {
            'high': "Strong anxious response, possibly uncomfortable with content",
            'medium': "Moderate anxiety or concern about content direction",
            'low': "Mild apprehension or cautious response to content"
        },
        'disgust': {
            'high': "Strong rejection response, clearly disliking content aspects",
            'medium': "Moderate dislike or disapproval of content elements",
            'low': "Mild disapproval or critical assessment of content"
        },
        'neutral': {
            'high': "Focused attention, processing information carefully",
            'medium': "Balanced engagement, thoughtful consideration of content",
            'low': "Passive observation, may need more stimulating content"
        }
    }

    # Determine confidence level
    if confidence > 80:
        confidence_level = 'high'
    elif confidence > 60:
        confidence_level = 'medium'
    else:
        confidence_level = 'low'

    base_interpretation = interpretations.get(emotion, {}).get(confidence_level, "Emotional state detected")

    # Add confidence level description
    confidence_text = "Highly certain" if confidence > 80 else \
        "Moderately certain" if confidence > 60 else \
            "Somewhat certain" if confidence > 40 else "Uncertain"

    # Check for mixed emotions with enhanced analysis
    secondary_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)[1:4]
    mixed_emotions = ""

    if secondary_emotions and secondary_emotions[0][1] > 15:
        primary_type = "positive" if emotion in ['happy', 'surprise'] else \
            "negative" if emotion in ['sad', 'angry', 'fear', 'disgust'] else "neutral"

        secondary_emo, secondary_conf = secondary_emotions[0]
        secondary_type = "positive" if secondary_emo in ['happy', 'surprise'] else \
            "negative" if secondary_emo in ['sad', 'angry', 'fear', 'disgust'] else "neutral"

        if primary_type != secondary_type and secondary_conf > 20:
            mixed_emotions = f" with significant conflicting {secondary_emo} elements"
        elif secondary_conf > 25:
            mixed_emotions = f" with notable {secondary_emo} undertones"

    # Add temporal context
    temporal_context = ""
    mood_trend = self.calculate_mood_trend()
    if mood_trend != "Stable":
        temporal_context = f" Mood is {mood_trend.lower()}."

    return f"{confidence_text} {base_interpretation}{mixed_emotions}.{temporal_context}"
```

**Interpretation Generation**:
1. Selects interpretation based on emotion and confidence level
2. Adds confidence descriptor ("Highly certain", etc.)
3. Analyzes secondary emotions for mixed emotional states
4. Includes temporal context from mood trend
5. Returns comprehensive interpretation string

---

## SECTION 18: VISUAL FEEDBACK METHODS (Lines 988-1000)

```python
def get_emotion_color(self, emotion, confidence):
    """Get appropriate color based on emotion and confidence"""
    if emotion not in self.emotion_colors:
        return (255, 255, 255)

    colors = self.emotion_colors[emotion]
    if confidence > 80:
        return colors[0]  # Bright color for high confidence
    elif confidence > 60:
        return colors[1]  # Medium color for medium confidence
    else:
        return colors[2]  # Dark color for low confidence
```

**Dynamic Color Selection**:
- Maps emotion to predefined color gradients
- Selects intensity based on confidence level
- Returns BGR tuple for OpenCV compatibility

---

## SECTION 19: MAIN EXECUTION LOOP (Lines 1001-1220)

### Main Run Method (Lines 1001-1007)

```python
def run(self):
    """
    Main loop for the ultra-precise emotion and paranormal recognition system
    """
    print("Starting Ultra-Precise Emotion & Paranormal Recognition System...")
    print("Press 'q' to quit")
```

**Initialization Messages**: Displays startup information and controls.

### Performance Monitoring (Lines 1009-1031)

```python
# For FPS calculation
prev_time = time.time()
fps = 0
frame_count = 0

try:
    print("[INFO] Starting main processing loop")
    while self.running:
        # Safely get frame from camera
        ret, frame = self.get_frame()
        if not ret:
            print("Failed to get frame, exiting...")
            break

        # Store the latest frame for processing
        self.latest_frame = frame.copy()
        if self.camera_available:
            self.frame_buffer.append(frame.copy())

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time
        frame_count += 1
```

**Main Loop Structure**:
1. Frame acquisition with error handling
2. Frame buffering for optical flow
3. Real-time FPS calculation
4. Frame counter for statistics

### Calibration Phase (Lines 1033-1044)

```python
# Handle calibration phase
calibration_status = ""
if not self.calibration_complete:
    calibration_status = self.calibrate_system(frame)
    # Display calibration status
    cv2.putText(frame, calibration_status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # Skip advanced processing during calibration
    cv2.imshow('Ultra-Precise Emotion & Paranormal Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    continue
```

**Calibration Display**:
- Shows progress percentage during 3-second calibration
- Displays on frame with yellow text
- Skips analysis until calibration completes

### Analysis Triggers (Lines 1046-1054)

```python
# Analyze emotion at a controlled interval
if current_time - self.last_analysis_time > self.analysis_interval:
    self.analyze_emotion_threaded(frame)
    self.last_analysis_time = current_time

# Perform enhanced analysis less frequently
if current_time - self.last_enhanced_analysis_time > self.enhanced_analysis_interval:
    self.enhanced_analysis_threaded(frame)
    self.last_enhanced_analysis_time = current_time
```

**Staggered Analysis**:
- Emotion analysis every 0.5 seconds
- Enhanced analysis every 3 seconds
- Prevents CPU overload while maintaining responsiveness

### Results Processing (Lines 1056-1076)

```python
# Get analysis results if available
emotion = "neutral"
confidence = 0
details = {'emotions': {}}

with self.analysis_lock:
    if self.analysis_results:
        emotion = self.analysis_results['emotion']
        confidence = self.analysis_results['confidence']
        details = self.analysis_results['details']

# Get enhanced analysis results if available
motion_analysis = {'motion_consistency': 0}
spectral_analysis = {'spectral_balance': 0, 'high_freq': 0}
temporal_analysis = {'abnormality': 0}

with self.enhanced_analysis_lock:
    if self.enhanced_analysis_results:
        motion_analysis = self.enhanced_analysis_results['motion']
        spectral_analysis = self.enhanced_analysis_results['spectral']
        temporal_analysis = self.enhanced_analysis_results['temporal']
```

**Thread-Safe Result Retrieval**:
- Uses locks to safely access thread-shared data
- Provides default values if analysis not yet complete
- Extracts emotion details and enhanced metrics

### Paranormal Detection (Lines 1078-1084)

```python
# Detect paranormal activity with enhanced precision
paranormal_data = self.detect_paranormal_activity(frame, motion_analysis, spectral_analysis,
                                                  temporal_analysis)

# Calculate engagement with enhanced algorithm
engagement = self.calculate_engagement(emotion, confidence, details.get('emotions', {}))
```

**Core Analysis Execution**:
- Runs paranormal detection with all available metrics
- Calculates engagement score from emotion data

### State Updates (Lines 1086-1099)

```python
# Update current state
self.current_emotion = emotion
self.confidence = confidence
self.engagement_score = engagement
self.mood_trend = self.calculate_mood_trend()

# Add to history
self.emotion_history.append({
    'emotion': emotion,
    'confidence': confidence,
    'engagement': engagement,
    'timestamp': current_time
})
self.emotion_confidence_history.append(confidence)

# Update charts less frequently
self.update_charts(details.get('emotions', {}))
```

**State Management**:
- Updates instance variables with latest results
- Appends to rolling history buffers
- Triggers chart updates at controlled intervals

### Visual Overlay - Face Detection (Lines 1103-1113)

```python
# Draw face bounding box if detected
region = details.get('region', {})
if region and 'x' in region and 'y' in region and 'w' in region and 'h' in region:
    x, y, w, h = region['x'], region['y'], region['w'], region['h']
    color = self.get_emotion_color(emotion, confidence)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Draw emotion label with confidence
    label = f"{emotion}: {confidence:.1f}%"
    cv2.putText(frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
```

**Face Visualization**:
- Draws colored rectangle around detected face
- Displays emotion and confidence above bounding box
- Color indicates emotion type and confidence level

### HUD Display (Lines 1115-1147)

```python
# Draw engagement info
engagement_text = f"Engagement: {engagement:.1f}%"
cv2.putText(frame, engagement_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Draw FPS
fps_text = f"FPS: {fps:.1f}"
cv2.putText(frame, fps_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Draw mood trend
mood_text = f"Mood: {self.mood_trend}"
cv2.putText(frame, mood_text, (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

# Draw paranormal detection info
if self.anomaly_detected:
    warning_text = "PARANORMAL ACTIVITY DETECTED!"
    cv2.putText(frame, warning_text, (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

paranormal_text = f"Paranormal: {self.paranormal_confidence:.1f}%"
cv2.putText(frame, paranormal_text, (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

energy_text = f"Energy Fluctuations: {self.energy_fluctuations:.1f}%"
cv2.putText(frame, energy_text, (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

if self.invisible_presence:
    invisible_text = "INVISIBLE PRESENCE DETECTED!"
    cv2.putText(frame, invisible_text, (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
```

**Heads-Up Display**:
- Engagement score (green)
- FPS counter (red)
- Mood trend (yellow)
- Paranormal alerts (red/cyan)
- Energy fluctuations (magenta)
- Invisible presence warning (orange)

### Emotion Interpretation Display (Lines 1149-1170)

```python
# Draw emotion interpretation
interpretation = self.get_emotion_interpretation(emotion, confidence, details.get('emotions', {}))
# Split interpretation into multiple lines if needed
words = interpretation.split()
lines = []
current_line = ""
for word in words:
    if len(current_line) + len(word) < 40:
        current_line += word + " "
    else:
        lines.append(current_line)
        current_line = word + " "
if current_line:
    lines.append(current_line)

y_offset = 240
for line in lines:
    cv2.putText(frame, line, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_offset += 20
    if y_offset > 400:  # Don't go beyond frame bottom
        break
```

**Interpretation Text Rendering**:
- Word-wraps long interpretation text
- Limits to 40 characters per line
- Displays in light gray at bottom of frame

### Detailed Emotion Display (Lines 1172-1182)

```python
# Draw detailed emotion information (top emotions only)
y_offset = 350
emotions = details.get('emotions', {})
if emotions:
    top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:4]
    for emo, value in top_emotions:
        emo_text = f"{emo}: {value:.1f}%"
        color = self.get_emotion_color(emo, value)
        cv2.putText(frame, emo_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 20
```

**Top Emotions List**:
- Shows top 4 emotions with confidence
- Color-coded by emotion type
- Positioned below interpretation text

### Chart Overlays (Lines 1184-1203)

```python
# Display engagement chart if available
if self.engagement_chart is not None:
    # Resize chart to fit
    chart = cv2.resize(self.engagement_chart, (320, 128))
    # Place chart in top-right corner
    frame[10:138, -330:-10] = chart

# Display paranormal chart if available
if self.paranormal_chart is not None:
    # Resize chart to fit
    paranormal_chart = cv2.resize(self.paranormal_chart, (320, 256))
    # Place chart in bottom-right corner
    frame[150:406, -330:-10] = paranormal_chart

# Display emotion distribution chart if available
if self.emotion_distribution_chart is not None:
    # Resize chart to fit
    emotion_chart = cv2.resize(self.emotion_distribution_chart, (320, 128))
    # Place chart in middle-right
    frame[280:408, -330:-10] = emotion_chart
```

**Real-time Chart Overlay**:
- Engagement chart: Top-right corner
- Paranormal chart: Bottom-right corner
- Emotion distribution: Middle-right
- All charts resized to fit frame dimensions

### Exit Handling (Lines 1208-1216)

```python
# Exit on 'q' key press with error handling
try:
    key = cv2.waitKey(1) & 0xFF
except Exception as e:
    print(f"Waiting for key press, press 'q' to quit: {e}")
    key = None

if key == ord('q'):
    break

# Periodically clean up memory
if frame_count % 100 == 0:
    gc.collect()
```

**User Interaction**:
- Checks for 'q' key press to exit
- Handles OpenCV key detection errors
- Periodic garbage collection every 100 frames

---

## SECTION 20: CLEANUP AND FINALIZATION (Lines 1222-1261)

```python
except Exception as e:
    print(f"Error in main loop: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Release resources
    self.running = False
    if self.camera_available:
        self.cap.release()
    cv2.destroyAllWindows()

    # Print comprehensive summary
    print("\n=== Session Summary ===")
    print(f"Duration: {time.time() - self.start_time:.1f} seconds")
    print(f"Frames processed: {frame_count}")
    if frame_count > 0:
        print(f"Average FPS: {frame_count / (time.time() - self.start_time):.1f}")
    print(f"Final Engagement: {self.engagement_score:.1f}%")
    print(f"Dominant Emotion: {self.current_emotion} ({self.confidence:.1f}% confidence)")
    print(f"Mood Trend: {self.mood_trend}")

    # Safely calculate paranormal events
    paranormal_events = 0
    paranormal_confidences = []
    energy_fluctuations = []

    for h in self.emotion_history:
        if 'paranormal_confidence' in h:
            paranormal_confidences.append(h['paranormal_confidence'])
            if h['paranormal_confidence'] > 75:
                paranormal_events += 1
        if 'energy_fluctuations' in h:
            energy_fluctuations.append(h['energy_fluctuations'])

    print(f"Paranormal Events Detected: {paranormal_events}")
    if paranormal_confidences:
        print(f"Highest Paranormal Confidence: {max(paranormal_confidences):.1f}%")
    if energy_fluctuations:
        print(f"Maximum Energy Fluctuations: {max(energy_fluctuations):.1f}%")
```

**Cleanup and Summary**:
1. **Exception Handling**: Catches and reports main loop errors
2. **Resource Cleanup**: Releases camera and destroys windows
3. **Session Statistics**: Duration, frame count, average FPS
4. **Final State**: Last emotion, confidence, engagement, mood trend
5. **Paranormal Summary**: Events detected, peak confidence and fluctuations

---

## SECTION 21: MAIN SCRIPT EXECUTION (Lines 1263-1282)

```python
if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import deepface
        import cv2
    except ImportError:
        print("Please install required packages:")
        print("pip install deepface opencv-python matplotlib pandas numpy scipy")
        exit(1)

    # Create and run the ultra-precise recognition system
    try:
        system = UltraPreciseEmotionParanormalRecognition()
        system.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
```

**Script Entry Point**:
1. **Dependency Check**: Verifies critical imports
2. **Installation Guidance**: Provides pip command if imports fail
3. **System Initialization**: Creates recognition system instance
4. **Execution**: Runs main processing loop
5. **Error Handling**: Catches and reports startup errors with traceback

---

## PROGRAM FLOW DIAGRAM

```
[START]
    ↓
[Import Libraries]
    ↓
[Class Definition: UltraPreciseEmotionParanormalRecognition]
    ↓
[__init__ Method]
    ├── Initialize emotion/color mappings
    ├── Setup camera with fallbacks
    ├── Create analysis buffers
    ├── Initialize threading components
    └── Setup calibration parameters
    ↓
[Model Initialization Methods]
    ├── initialize_emotion_model()
    ├── initialize_anomaly_model()
    └── setup_advanced_detection()
    ↓
[Main Run Loop]
    ├── Frame Acquisition (get_frame())
    ├── Calibration Phase (calibrate_system())
    ├── Emotion Analysis (analyze_emotion_threaded())
    ├── Enhanced Analysis (enhanced_analysis_threaded())
    ├── Paranormal Detection (detect_paranormal_activity())
    ├── State Updates & History
    ├── Chart Generation (update_charts())
    ├── Visual Overlays (HUD, Charts)
    └── User Input Check ('q' to quit)
    ↓
[Cleanup & Summary]
    ├── Resource Release
    ├── Performance Statistics
    └── Paranormal Event Summary
    ↓
[END]
```

---

## KEY ALGORITHMS SUMMARY

1. **Emotion Detection**: DeepFace library with OpenCV backend
2. **Motion Analysis**: OpenCV optical flow (Farneback algorithm)
3. **Background Subtraction**: MOG2 algorithm for foreground detection
4. **Spectral Analysis**: 2D Fast Fourier Transform
5. **Anomaly Detection**: Multi-method statistical analysis (Z-score, percentile, moving average)
6. **Energy Calculation**: RMS, standard deviation, and gradient-based metrics
7. **Engagement Scoring**: Weighted emotion analysis with secondary emotion boosts
8. **Mood Trend**: Linear regression on emotion score time series
9. **Paranormal Detection**: Multi-criteria analysis with confidence weighting

---

## MEMORY MANAGEMENT

- **Deque Buffers**: Fixed-size rolling buffers prevent memory growth
- **Garbage Collection**: Manual GC calls in threads to prevent leaks
- **Pre-allocated Objects**: Charts and arrays allocated once, reused
- **Thread Cleanup**: Proper thread termination and resource release

---

## PERFORMANCE OPTIMIZATIONS

- **Threading**: Parallel emotion and enhanced analysis
- **Staggered Updates**: Different intervals for different analysis types
- **Resolution Limits**: 640x480 optimized frame size
- **Chart Intervals**: 1.5-second chart update intervals
- **Buffer Size Limits**: Reduced buffer sizes for memory efficiency

This concludes the comprehensive code manual for the FINAL EMOTION DETECTION PROGRAM. Each section has been explained with detailed technical information about the implementation, algorithms, and program flow.