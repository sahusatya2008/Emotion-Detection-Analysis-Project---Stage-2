# EMOTION DETECTION PROGRAM BY SATYA Narayan SAHU

## Description

This is a comprehensive ultra-precise emotion detection and paranormal activity recognition program developed by Satya N Sahu. The program utilizes advanced computer vision and machine learning techniques to analyze facial expressions in real-time through a webcam feed. Beyond standard emotion recognition, it includes sophisticated paranormal detection algorithms that monitor for anomalous activities, invisible presences, energy fluctuations, and temporal anomalies.

The system employs multi-threaded processing for optimal performance, real-time charting, and adaptive calibration to ensure maximum accuracy. It features a robust fallback mechanism that switches to demo mode if no camera is available, making it accessible even without hardware.

## Features

### Core Emotion Detection
- **Real-time Emotion Recognition**: Detects 7 basic emotions: happy, sad, angry, fear, surprise, disgust, neutral
- **Confidence Scoring**: Provides percentage confidence for each emotion detection
- **Multi-emotion Analysis**: Tracks secondary emotions and mixed emotional states
- **Adaptive Confidence Boost**: Enhances accuracy by analyzing emotion distribution patterns

### Advanced Analytics
- **Engagement Calculation**: Computes real-time engagement scores based on emotion weights and confidence levels
- **Mood Trend Analysis**: Monitors emotional patterns over time (improving, declining, stable)
- **Emotion History Tracking**: Maintains rolling history with configurable maxlen for temporal analysis
- **Confidence History**: Tracks confidence levels over time for stability analysis

### Paranormal Detection System
- **Anomaly Detection**: Advanced statistical anomaly detection using Z-score, percentile, and moving average methods
- **Invisible Presence Detection**: Multi-criteria detection including motion without contours, high-frequency spectral content, and energy spikes
- **Energy Fluctuation Analysis**: Monitors energy patterns using RMS, standard deviation, and gradient calculations
- **Temporal Anomaly Detection**: Analyzes temporal consistency and detects abnormal patterns
- **Spectral Analysis**: Performs 2D FFT analysis for frequency domain anomaly detection
- **Motion Analysis**: Utilizes optical flow for precise motion tracking and consistency measurement

### Performance Optimization
- **Multi-threaded Processing**: Dedicated threads for emotion analysis, enhanced analysis, and main processing
- **Memory Management**: Automatic garbage collection and memory buffer management
- **Background Subtraction**: Uses MOG2 algorithm for efficient foreground detection
- **FPS Monitoring**: Real-time frames per second calculation and display
- **Adaptive Intervals**: Configurable processing intervals to balance accuracy and performance

### Real-time Visualization
- **Live Emotion Charts**: Real-time engagement, paranormal activity, and emotion distribution charts
- **Color-coded Emotion Display**: Dynamic color gradients based on emotion type and confidence
- **Face Detection Overlay**: Bounding boxes around detected faces with emotion labels
- **Comprehensive HUD**: Displays engagement, FPS, mood trend, paranormal confidence, and energy fluctuations

### System Resilience
- **Camera Auto-detection**: Automatically tries multiple camera indices
- **Fallback Demo Mode**: Switches to demo mode when no camera is available
- **Error Handling**: Robust exception handling and recovery mechanisms
- **Calibration System**: 3-second initial calibration for environmental adaptation
- **Platform Compatibility**: Optimized for macOS with Windows/Linux support

### Technical Specifications
- **Resolution Support**: 640x480 optimized resolution with configurable settings
- **Buffer Management**: Configurable deque buffers for motion, energy, spectral, and emotion history
- **Statistical Models**: Integrated scipy.stats for advanced statistical analysis
- **DeepFace Integration**: Utilizes DeepFace library for robust face detection and emotion analysis
- **OpenCV Integration**: Full OpenCV functionality for image processing and computer vision tasks

## Installation

### Prerequisites
- **Python Version**: 3.8 or higher
- **Operating System**: macOS (optimized), Windows 10/11, or Linux
- **Hardware Requirements**:
  - Webcam (optional - demo mode available)
  - Minimum 4GB RAM (8GB recommended)
  - Multi-core CPU (GPU optional for enhanced performance)
- **Storage**: ~500MB for dependencies and models

### Installation Steps

1. **Clone or Download the Repository**
   ```bash
   # If using git (assuming repository exists)
   git clone [repository_url]
   cd [repository_directory]
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   python3 "FINAL EMOTION DETECTION PROGRAM BY SATYA N SAHU.py"
   ```

## Usage

### Basic Usage
```bash
python3 "FINAL EMOTION DETECTION PROGRAM BY SATYA N SAHU.py"
```

### Controls
- **Start**: The program starts automatically after calibration
- **Quit**: Press `q` key to exit the program

### Program Flow
1. **Initialization**: Loads models and initializes camera
2. **Calibration**: 3-second environmental calibration phase
3. **Main Loop**: Continuous real-time analysis and display
4. **Cleanup**: Automatic resource cleanup on exit

### Output Display
- **Main Window**: "Ultra-Precise Emotion & Paranormal Recognition"
- **Real-time Metrics**: Emotion, confidence, engagement, FPS, mood trend
- **Paranormal Indicators**: Confidence levels, energy fluctuations, anomaly alerts
- **Visual Charts**: Engagement timeline, motion/energy graphs, emotion distribution
- **Face Detection**: Bounding boxes with emotion labels and confidence scores

## System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.8+
- **RAM**: 4GB
- **Storage**: 1GB free space
- **Camera**: Any webcam (fallback to demo mode)

### Recommended Requirements
- **OS**: macOS 12+ or Windows 11
- **Python**: 3.9+
- **RAM**: 8GB+
- **Storage**: 2GB+ free space
- **CPU**: Quad-core 2.5GHz+
- **GPU**: NVIDIA GPU with CUDA support (optional)

### Supported Platforms
- **macOS**: Fully optimized with window thread fixes
- **Windows**: Full compatibility
- **Linux**: Ubuntu/Debian based distributions

## Dependencies

### Core Dependencies
- **opencv-python>=4.5.0**: Computer vision library for image processing and camera handling
- **deepface>=0.0.75**: Deep learning face detection and emotion recognition
- **matplotlib>=3.5.0**: Plotting library for real-time charts and visualization
- **pandas>=1.3.0**: Data manipulation and analysis
- **numpy>=1.21.0**: Numerical computing library
- **scipy>=1.7.0**: Scientific computing with signal processing capabilities

### DeepFace Sub-dependencies
- **tensorflow>=2.8.0**: Machine learning framework
- **pillow>=8.0.0**: Image processing library
- **mtcnn>=0.1.0**: Multi-task Cascaded Convolutional Networks for face detection
- **retina-face>=0.0.12**: Accurate face detection model
- **gdown>=4.4.0**: Google Drive downloader for model files
- **tqdm>=4.62.0**: Progress bar library

## How It Works

### Architecture Overview
The program is built around the `UltraPreciseEmotionParanormalRecognition` class with the following key components:

1. **Initialization Module**: Sets up camera, buffers, and models
2. **Calibration System**: Adapts to environmental conditions
3. **Multi-threaded Analysis Engine**: Parallel processing for emotions and anomalies
4. **Real-time Visualization**: Live display and charting system
5. **Resource Management**: Memory and thread cleanup

### Emotion Detection Pipeline
1. **Face Detection**: Uses DeepFace with OpenCV backend
2. **Emotion Analysis**: Multi-model emotion recognition
3. **Confidence Calculation**: Weighted scoring with secondary emotion boost
4. **Engagement Scoring**: Complex algorithm based on emotion weights
5. **History Tracking**: Rolling buffers for temporal analysis

### Paranormal Detection Pipeline
1. **Motion Analysis**: Background subtraction and optical flow
2. **Energy Calculation**: Multi-method energy level computation
3. **Spectral Analysis**: 2D FFT for frequency domain analysis
4. **Anomaly Detection**: Statistical methods for outlier identification
5. **Pattern Recognition**: Temporal consistency and spectral balance analysis

### Technical Algorithms
- **Optical Flow**: Farneback algorithm for motion estimation
- **Background Subtraction**: MOG2 algorithm for foreground detection
- **FFT Analysis**: 2D Fast Fourier Transform for spectral processing
- **Statistical Modeling**: Z-score, percentile, and moving average anomaly detection
- **Temporal Analysis**: Rate of change and stability calculations

## Output and Visualization

### Real-time Display Elements
- **Primary Emotion**: Dominant emotion with confidence percentage
- **Confidence Score**: Accuracy level of emotion detection
- **Engagement Level**: Calculated engagement percentage
- **FPS Counter**: Current frames per second
- **Mood Trend**: Current emotional trend direction
- **Paranormal Confidence**: Overall paranormal activity confidence
- **Energy Fluctuations**: Current energy fluctuation level
- **Invisible Presence Alert**: Binary detection indicator

### Chart Displays
- **Engagement Chart**: Time-series plot of engagement scores
- **Paranormal Chart**: Dual plot of motion and energy levels
- **Emotion Distribution**: Bar chart of all emotion probabilities

### Face Detection Overlay
- **Bounding Box**: Color-coded rectangle around detected face
- **Emotion Label**: Emotion name with confidence score
- **Color Coding**: Dynamic colors based on emotion and confidence

## Troubleshooting

### Common Issues and Solutions

**Camera Not Detected**
```
Error: Primary camera not available
```
- Try different camera indices (0, 1, 2, etc.)
- Check camera permissions on macOS/Linux
- Restart camera application if in use
- Program will automatically switch to demo mode

**Dependencies Installation Failed**
```
ImportError: No module named 'deepface'
```
- Ensure Python 3.8+ is installed
- Use virtual environment: `python -m venv venv && source venv/bin/activate`
- Upgrade pip: `pip install --upgrade pip`
- Install specific versions: `pip install deepface==0.0.75`

**Performance Issues**
- **Low FPS**: Reduce resolution in code or close other applications
- **High Memory Usage**: Program automatically manages memory with GC
- **Lag**: Increase analysis intervals in code (analysis_interval, enhanced_analysis_interval)

**Display Issues**
- **Charts Not Showing**: Ensure matplotlib backend is set correctly
- **Window Not Responding**: Check for threading conflicts (macOS specific)
- **Color Issues**: Verify OpenCV installation with GUI support

**Paranormal Detection Not Working**
- Ensure calibration completes (3 seconds after start)
- Check lighting conditions for better motion detection
- Paranormal features require camera input (demo mode disables)

### Debug Mode
Run with additional logging:
```python
# Add print statements in key functions for debugging
print(f"Emotion: {emotion}, Confidence: {confidence}")
```

### System Compatibility
- **macOS**: Use `cv2.startWindowThread()` for window management
- **Windows**: Ensure firewall allows camera access
- **Linux**: Install camera drivers and permissions

## Author

**Satya N Sahu**
- Developer and maintainer of the FINAL EMOTION DETECTION PROGRAM
- Specializes in computer vision and machine learning applications

## Version History

### Version 1.0.0 (Current)
- Initial release with full functionality
- Ultra-precise emotion and paranormal detection
- Multi-threaded processing and real-time visualization
- Cross-platform compatibility

## License

This project is proprietary software developed by Satya Narayan Sahu. All rights reserved.

## Contributing

Currently not accepting external contributions. This is a personal research project.

## Acknowledgments

### Libraries and Frameworks
- **DeepFace**: For robust face detection and emotion recognition
- **OpenCV**: For computer vision and image processing capabilities
- **Matplotlib**: For real-time data visualization
- **NumPy & SciPy**: For numerical computing and signal processing
- **TensorFlow**: For deep learning model execution

### Research and Inspiration
- Facial emotion recognition research
- Paranormal investigation methodologies
- Computer vision advancements in anomaly detection
- Real-time processing optimization techniques

## Support

For technical support or questions:
- Review the troubleshooting section above
- Check dependency versions compatibility
- Ensure system meets minimum requirements
- Monitor console output for error messages

## Future Enhancements

### Planned Features
- Multi-face tracking and analysis
- Cloud synchronization for data logging
- Advanced machine learning model integration
- Mobile device support
- API development for integration
- Extended paranormal detection algorithms

### Performance Improvements
- GPU acceleration for FFT calculations
- Optimized threading for multi-core systems
- Reduced memory footprint for embedded devices
- Enhanced calibration algorithms

---

**Disclaimer**: This program is for research and entertainment purposes. Paranormal detection features are experimental and based on statistical analysis, not scientifically proven methods.
