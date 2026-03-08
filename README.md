# RPi-VespAI - Vesp Detection System

🐝 **Vesp detection system** with real-time computer vision, web dashboard, and SMS alerts (To be tested).

RPi-VespAI uses a YOLOv8 deep learning trained model to identify and differentiate between Asian hornets (Vespa velutina) and European hornets (Vespa crabro) in real-time. 

**Based on the research:** *VespAI: a deep learning-based system for the detection of invasive hornets* published in Communications Biology (2024). DOI: [10.1038/s42003-024-05979-z](https://doi.org/10.1038/s42003-024-05979-z)

**Credits** This project was based on the work done by Jakob Zeise (https://github.com/jakobzeise/vespai/)


## ✨ Features

- 🔍 **Real-time Detection**: YOLOv8-based computer vision with custom hornet model
- 📊 **Web Dashboard**: Live video feed with statistics and detection analytics
- 🌍 **Bilingual Support**: Complete English/German/French interface with flag-based switching
- 📱 **SMS Alerts**: Automated notifications via Lox24 API with intelligent rate limiting - To be Tested
- 🎯 **Motion Detection**: CPU-efficient motion-based optimization
- 📈 **Data Analytics**: Comprehensive logging, hourly statistics, and detection history
- 📱 **Mobile Responsive**: Optimized web interface with adaptive charts (24h/4h views)
- ⚡ **Performance Optimized**: Non-blocking operations, data caching, reduced API calls
- 🏗️ **Modular Architecture**: Clean, maintainable codebase with separation of concerns


---

## 🚀 Quick Start

This will propably work on a normal PC with Linux. However its been designed and tested for Raspberry Pi 5.
A normal USB WebCam e.g Logitech can work
The Raspberry Pi Camera can work with a few code changes

### Option 1: One-Click Setup (Recommended)

**Raspberry Pi:**
```bash
chmod +x scripts/raspberry-pi-setup.sh
./scripts/raspberry-pi-setup.sh
```

```bash
# Clone to home directory (recommended for permissions)
cd ~
git clone https://github.com/jordaaaa1970/RPi-vespai
cd RPi-vespai

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Setup and run
python scripts/setup.py
python vespai.py --web --resolution 720p --motion
```
📋 **For complete installation instructions including Raspberry Pi setup, see [docs/INSTALL.md](docs/INSTALL.md)**


### 2. Run the System

```bash
# Basic usage with web interface
python vespai.py --web

# With motion detection and image saving
python vespai.py --web --motion --save

# Performance mode for Raspberry Pi
python vespai.py --web --resolution 720p --motion --conf 0.7
```

### 3. Access Dashboard
Open your browser to: `http://localhost:8081`

PS: To get a demo: Click on the Red Live button in the Live Feed. This will change the feed from Camera to the Dataset, streaming images from the dataset into the detector with results then logged as if running live.


## Configuration

### Environment Variables (.env file)

```bash

# SMS Configuration (Lox24 API)
LOX24_API_KEY=your_customer_number:your_api_key
LOX24_SENDER=VespAI
PHONE_NUMBER=+1234567890
SMS_DELAY_MINUTES=5
ENABLE_SMS=true

# Web Server
DOMAIN_NAME=localhost
USE_HTTPS=false

# Detection
CONFIDENCE_THRESHOLD=0.8
VESPAI_CLASS_MAP=0:crabro,1:velutina
# For the current ONNX dataset model used in this repository:
# VESPAI_CLASS_MAP=1:crabro,2:velutina
SAVE_DETECTIONS=false
SAVE_DIRECTORY=monitor/detections
```

### Command Line Options

```bash

# Usage:
python vespai.py [OPTIONS]

Options:
  --web                    Enable web dashboard
  -c, --conf FLOAT        Detection confidence threshold (default: 0.8)
  --model-path PATH       Path to model weights/artifact
  --class-map TEXT        Class mapping (e.g. "0:crabro,1:velutina")
  -s, --save              Save detection images
  -sd, --save-dir PATH    Directory for saved images
  -v, --video PATH        Use video file instead of camera
  -r, --resolution WxH    Camera resolution (default: 1920x1080)
  -m, --motion            Enable motion detection
  -a, --min-motion-area INT  Minimum motion area threshold
  --dataset-delay FLOAT   Minimum frame delay for finite dataset inputs
  --web-host HOST         Web server host
  --web-port PORT         Web server port
  -b, --brake FLOAT       Frame processing delay (default: 0.1)
  -p, --print             Print detection details to console
```

## Installation

### System Requirements

- **Python**: 3.7+ (3.9+ recommended for Raspberry Pi)  
- **Memory**: 2GB RAM minimum, 4GB recommended
- **Storage**: 1GB free space for models and dependencies
- **Camera**: USB camera or CSI camera (Raspberry Pi)

### Supported Platforms
- ✅ **Raspberry Pi 5** (full support)


### Dependencies Installation

#### Quick Install (All Platforms)
```bash
python scripts/setup.py
```

#### Manual Install
```bash
pip install -r requirements.txt
```

#### Raspberry Pi Optimizations
```bash
# Enable GPU memory (128MB recommended)
sudo raspi-config
# Advanced Options > Memory Split > 128

# Install system dependencies
sudo apt update && sudo apt install python3-opencv python3-pip git
```

### VespAI Model Setup

1. **VespAI Hornet Model** (Recommended):
   - Specialized model trained for hornet detection
   - **Classes**: 0=Vespa crabro, 1=Vespa velutina  
   - **Size**: 14MB
   - **Download**: Automated via `python scripts/setup.py`
   - **Manual**: See [docs/INSTALL.md](docs/INSTALL.md) for manual download

2. **Fallback Model**:
   - Generic `yolov5s.pt` as fallback
   - ⚠️ Not optimized for hornet detection - may produce false alerts

## Usage Examples

### Run Modes
```bash
# A) Webcam (live camera)
python vespai.py --web --resolution 720p --motion --conf 0.7

# B) Dataset (TFRecord file)
python vespai.py --web \
  --model-path models/L4-yolov8_asianhornet_2026-03-06_19-45-38.onnx \
  --class-map "1:crabro,2:velutina" \
  --video "datasets/Detection Asian-hornet.v1i.tfrecord/test/asianhornet.tfrecord" \
  --resolution 720p --conf 0.7 --print
```

### Basic Monitoring
```bash

# Start with web interface
python vespai.py --web

# Add motion detection for better performance
python vespai.py --web --motion
```

### Production Deployment
```bash
# Full featured production setup
python vespai.py --web --motion --save --conf 0.85

# Raspberry Pi optimized
python vespai.py --web --resolution 720p --motion --conf 0.7

# Custom ONNX/class ordering
python vespai.py --web --model-path models/custom.onnx --class-map "1:crabro,2:velutina"

# Process recorded video
python vespai.py --video input.mp4 --save --conf 0.9
```


## Web Interface

### Dashboard Features
- **Live Video Feed**: Real-time camera stream with detection overlays
- **Statistics Cards**: Frame count, detection counts, system stats
- **Detection Log**: Chronological list of all detections with timestamps
- **Hourly Chart**: 24-hour detection history visualization
- **System Monitor**: CPU, RAM, temperature monitoring
- **Inference Time**: Time taken by model to classify image

### API Endpoints
- `GET /` - Main dashboard
- `GET /video_feed` - Live video stream
- `GET /api/stats` - Real-time statistics JSON
- `GET /frame/<frame_id>` - Specific detection frame

## SMS Alerts

### Lox24 Configuration
1. Register at [Lox24](https://www.lox24.eu/)
2. Get your API credentials
3. Set in `.env`:
   ```bash
   LOX24_API_KEY=customer_number:api_key
   PHONE_NUMBER=+1234567890
   ```

### Alert Behavior
- **Asian Hornet**: High priority alert sent immediately
- **European Hornet**: Lower priority info message
- **Rate Limiting**: Minimum 5-minute delay between SMS
- **Cost Tracking**: Monitors SMS costs and delivery status


### Security Considerations
- Never commit `.env` files to git
- Use strong SMS API credentials
- Consider VPN access for remote monitoring
- Regular security updates on Pi OS

### Performance Optimization
- Use motion detection (`--motion`) to reduce CPU usage
- Adjust confidence threshold based on your environment
- Consider GPU acceleration for better performance
- Monitor system resources via dashboard

## Development

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Testing
- Test with various lighting conditions
- Verify SMS delivery and costs
- Check web interface on mobile devices
- Validate motion detection accuracy

---

## Troubleshooting

### Common Issues

**Camera not detected:**
```bash

# Check camera devices
ls /dev/video*
# Try different camera indices
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

**YOLOv8 model loading errors:**
- Ensure model path is correct
- Check PyTorch installation
- Verify model compatibility

**SMS not working:**
- Check API credentials in `.env`
- Verify phone number format (+country_code)
- Check Lox24 account balance

**Web interface not accessible:**
- Confirm port 8081 is not blocked
- Check firewall settings
- Verify Flask is running

### Logs
Check `vespai.log` for manual runs, or `sudo journalctl -u vespai-web.service` for the boot service.

## Citation

If you use this project in your research or work, please cite the original research:

```bibtex
@article{vespai2024,
  title={VespAI: a deep learning-based system for the detection of invasive hornets},
  journal={Communications Biology},
  year={2024},
  volume={7},
  pages={318},
  doi={10.1038/s42003-024-05979-z},
  url={https://doi.org/10.1038/s42003-024-05979-z}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

**Important:** This implementation is based on research published in Communications Biology. The original research methodology and concepts are attributed to the authors of the cited paper.

## Acknowledgments

- Original VespAI research published in Communications Biology (2024)
- YOLOv8 by Ultralytics
- Lox24 SMS API
- Flask web framework
- OpenCV computer vision library