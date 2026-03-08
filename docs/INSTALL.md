# VespAI Installation Guide

**Complete Setup Guide for VespAI Hornet Detection System**

## 🚀 Quick Start

### Standard Systems (Windows, macOS, Linux)
```bash
# Clone repository
git clone https://github.com/jakobzeise/vespai.git
cd vespai

# Run automated setup
python scripts/setup.py

# Start the system
python vespai.py --web
```

### Raspberry Pi (PEP 668 Compatible)
```bash

# Clone repository to home directory (recommended for permissions)
cd ~
git clone https://github.com/jakobzeise/vespai.git
cd vespai

# Install Git LFS to download model files properly
sudo apt update
sudo apt install git-lfs
git lfs install
git lfs pull

# Make setup script executable and run
chmod +x scripts/raspberry-pi-setup.sh
./scripts/raspberry-pi-setup.sh

# Or manual setup with virtual environment
python3 -m venv .venv
source .venv/bin/activate
python scripts/setup.py
python vespai.py --web
```

Open http://localhost:8081 in your browser.

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.7+ (3.9+ recommended for Raspberry Pi)
- **Memory**: 2GB RAM minimum, 4GB recommended
- **Storage**: 1GB free space for models and dependencies
- **Camera**: USB camera or CSI camera (Raspberry Pi)

### Supported Platforms
- ✅ **Windows** 10/11 (x64)
- ✅ **macOS** 10.15+ (Intel/Apple Silicon)
- ✅ **Linux** Ubuntu 18.04+, Debian 10+
- ✅ **Raspberry Pi 4** (4GB/8GB RAM recommended)
- ✅ **Raspberry Pi 5** (full support)

### Raspberry Pi Specific
- **OS**: Raspberry Pi OS (64-bit recommended)
- **Camera**: Pi Camera Module or USB webcam
- **SD Card**: Class 10, 32GB minimum
- **Power**: Official 5V 3A power supply

## 🔧 Automated Installation

### Using Setup Script (Recommended)

The setup script handles everything automatically:

**Standard Systems:**
```bash

cd vespai
python scripts/setup.py
```

**Raspberry Pi (requires virtual environment):**
```bash

# Clone to home directory first (recommended for permissions)
cd ~
git clone https://github.com/jakobzeise/vespai.git
cd vespai

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
python scripts/setup.py
```

**What it does:**
- ✅ Detects PEP 668 and creates virtual environment if needed
- ✅ Verifies Python version compatibility
- ✅ Installs all required Python packages
- ✅ Downloads VespAI hornet detection model (14MB)
- ✅ Creates necessary directories
- ✅ Sets up configuration templates
- ✅ Tests camera availability
- ✅ Provides virtual environment activation instructions

**For all models (optional):**
```bash

python scripts/setup.py --all-models
```

## 🛠 Manual Installation

If you prefer manual setup or need custom configuration:

### 1. Install Python Dependencies

**Standard Systems:**
```bash
pip install -r requirements.txt
```

**Raspberry Pi (create virtual environment first):**
```bash
# Clone to home directory (recommended for permissions)
cd ~
git clone https://github.com/jakobzeise/vespai.git
cd vespai

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Key packages installed:**
- `opencv-python` - Computer vision
- `torch` + `torchvision` - Deep learning
- `yolov5` - Object detection
- `flask` - Web interface
- `ultralytics` - YOLOv5 utilities

### 2. Download VespAI Model

The system requires the specialized hornet detection model:

```bash
# Create model directory
mkdir -p models

# Download VespAI hornet model (14MB)
curl -L -o models/yolov5s-all-data.pt \
  "https://github.com/jakobzeise/vespai/raw/main/models/yolov5s-all-data.pt"
```

### 3. Create Directory Structure

```bash
mkdir -p logs monitor/detections docs
touch logs/.gitkeep monitor/.gitkeep monitor/detections/.gitkeep
```

## ⚙️ Configuration

### Basic Configuration

1. **Copy configuration template:**
```bash
cp .env.template .env
```

2. **Edit configuration (optional):**
```bash
# .env file contents
MODEL_PATH=models/yolov5s-all-data.pt
CONFIDENCE_THRESHOLD=0.8
# For the hornet-trained YOLOv5 model:
VESPAI_CLASS_MAP=0:crabro,1:velutina
# For the current ONNX dataset model used in this repository:
# VESPAI_CLASS_MAP=1:crabro,2:velutina

# Camera Settings
RESOLUTION=1280x720
FRAME_DELAY=0.35
DATASET_FRAME_DELAY=5.0

# Web Interface
WEB_HOST=0.0.0.0
WEB_PORT=8081

# Detection Settings
SAVE_DETECTIONS=true
SAVE_DIRECTORY=monitor/detections

# SMS Alerts (Optional)
# LOX24_API_KEY=your_api_key_here
# PHONE_NUMBER=+1234567890
# DOMAIN_NAME=your-domain.com

# Motion Detection (Optional)
ENABLE_MOTION_DETECTION=false
MIN_MOTION_AREA=5000
```

### Raspberry Pi Optimizations

For Raspberry Pi 4/5, add these optimizations:

```bash
# Enable GPU memory (128MB recommended)
sudo raspi-config
# Advanced Options > Memory Split > 128

# Optimize camera settings in .env
RESOLUTION=1280x720         # Good balance for Pi
FRAME_DELAY=0.35            # Reduce CPU load
DATASET_FRAME_DELAY=5.0     # Slower dataset playback
CONFIDENCE_THRESHOLD=0.7    # Lower for better detection
```

## 🎯 Running VespAI

### Basic Usage

**Standard Systems:**
```bash
# Start with web dashboard
python vespai.py --web

# Start with motion detection (saves CPU)
python vespai.py --web --motion

# Process video file instead of camera
python vespai.py --web --video hornets.mp4

# Save all detection images
python vespai.py --web --save
```

**Raspberry Pi (activate virtual environment first):**
```bash
# Activate virtual environment
source .venv/bin/activate

# Start with web dashboard (optimized for Pi)
python vespai.py --web --resolution 720p --motion

# Performance mode
python vespai.py --web --resolution 640x480 --motion --conf 0.7
```

### Command Line Options

```bash
python vespai.py --web [OPTIONS]

Options:
  --conf FLOAT          Detection confidence (0.0-1.0) [default: 0.8]
  --model-path PATH     Path to model weights/artifact
  --class-map TEXT      Class mapping (e.g. "0:crabro,1:velutina")
  --save               Save detection images
  --save-dir PATH      Directory for saved images [default: monitor/detections]
  --video PATH         Use video file instead of camera
  --resolution RES     Camera resolution (1920x1080, 1280x720, 720p) [default: 1920x1080]
  --motion             Enable motion detection
  --min-motion-area N  Minimum motion area threshold [default: 100]
  --brake FLOAT        Frame processing delay [default: 0.1]
  --dataset-delay FLOAT  Minimum frame delay for finite dataset inputs [default: 0.6]
  --web-host HOST      Web server host [default: 0.0.0.0]
  --web-port PORT      Web server port [default: 5000 unless overridden in .env]
  --print              Print detection details to console
```

### Examples

```bash
# A) Webcam (live camera)
python vespai.py --web --resolution 720p --motion --conf 0.7

# B) Dataset (TFRecord file)
python vespai.py --web \
  --model-path models/L4-yolov8_asianhornet_2026-03-06_19-45-38.onnx \
  --class-map "1:crabro,2:velutina" \
  --video "datasets/Detection Asian-hornet.v1i.tfrecord/test/asianhornet.tfrecord" \
  --resolution 720p --conf 0.7 --print

# High accuracy mode
python vespai.py --web --conf 0.9 --save

# Performance mode for Raspberry Pi
python vespai.py --web --resolution 720p --motion --conf 0.7

# Process recorded video
python vespai.py --web --video /path/to/hornet_video.mp4 --save

# Debug mode
python vespai.py --web --print
```

### Web Interface Access

Once started, access the dashboard:
- **Local**: http://localhost:8081
- **Network**: http://YOUR-RASPBERRY-PI-IP:8081
- **All interfaces**: http://0.0.0.0:8081

## 📱 Web Dashboard Features

### Live Detection
- ✅ **Real-time video feed** - Smooth canvas-based display (no flickering)
- ✅ **Hornet detection** - Identifies Vespa velutina (Asian) vs Vespa crabro (European)
- ✅ **Detection overlays** - Bounding boxes with confidence scores
- ✅ **Live statistics** - Frame rate, detection counts, system status

### Statistics & Analytics
- ✅ **Real-time counters** - Total detections, species breakdown
- ✅ **Hourly charts** - 24-hour detection history
- ✅ **Detection log** - Timestamped detection history with images
- ✅ **System monitoring** - CPU temp, RAM usage, uptime

### Smart Features
- ✅ **SMS alerts** - Optional notifications via Lox24 API
- ✅ **Rate limiting** - Prevents alert spam (5-minute delays)
- ✅ **Cost tracking** - SMS cost monitoring
- ✅ **Motion optimization** - Only process frames with motion

## 🐞 Troubleshooting

### Common Issues

**"Module not found" errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or use setup script
python scripts/setup.py
```

**"Model not found" error:**
```bash
# Check model exists
ls -la models/yolov5s-all-data.pt

# Re-download if missing
curl -L -o models/yolov5s-all-data.pt \
  "https://github.com/jakobzeise/vespai/raw/main/models/yolov5s-all-data.pt"
```

**Camera not detected:**
```bash
# Test camera manually
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: Available')
        cap.release()
    else:
        print(f'Camera {i}: Not available')
"

# For Raspberry Pi camera
sudo raspi-config
# Interface Options > Camera > Enable
```

**Web interface not loading:**
```bash
# Check if server is running
curl http://localhost:8081

# Check logs when started manually with nohup
tail -f vespai.log

# Check logs when started by systemd
sudo journalctl -u vespai-web.service -f

# Try different port
python vespai.py --web --web-port 5000
```

**Performance issues on Raspberry Pi:**
```bash
# Use lower resolution
python vespai.py --web --resolution 640x480

# Enable motion detection
python vespai.py --web --motion

# Check GPU memory split
vcgencmd get_mem gpu  # Should be 128+

# Monitor temperature
vcgencmd measure_temp
```

**Unicode/logging errors (Windows):**
- Fixed automatically in current version
- Ensure Windows Terminal supports UTF-8
- Use PowerShell or WSL if issues persist

### Raspberry Pi Specific Issues

**Camera Module not detected:**
```bash
# Enable camera interface
sudo raspi-config
# Interface Options > Camera > Enable
# Reboot required

# Test Pi camera
raspistill -o test.jpg

# Check camera connection
vcgencmd get_camera
```

**Out of memory errors:**
```bash
# Check available RAM
free -h

# Reduce camera resolution
python vespai.py --web --resolution 640x480

# Enable swap if needed (not recommended for SD cards)
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

**Slow performance:**
```bash
# Check CPU frequency
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq

# Enable motion detection to reduce processing
python vespai.py --web --motion --min-motion-area 8000

# Use lower confidence threshold
python vespai.py --web --conf 0.6
```

## 🔬 Model Information

VespAI uses a specialized YOLOv5 model trained specifically for hornet detection:

- **Model**: `yolov5s-all-data.pt` (14MB)
- **Classes**: 
  - **0**: Vespa crabro (European hornet)
  - **1**: Vespa velutina (Asian hornet - invasive)
- **Research**: Based on Communications Biology 2024 paper
- **Accuracy**: Optimized for hornet species differentiation

### Model Performance
- **Input size**: 640x640 pixels
- **Parameters**: ~7M parameters
- **Speed**: ~15-30 FPS (depending on hardware)
- **Accuracy**: >95% on hornet detection task

## 🚀 Production Deployment

### Systemd Service (Linux)
```bash
# Create service file
sudo nano /etc/systemd/system/vespai-web.service

[Unit]
Description=VespAI Web Dashboard
After=network.target

[Service]
Type=simple
User=sysadmin
Group=sysadmin
WorkingDirectory=/home/sysadmin/vespai
ExecStart=/home/sysadmin/vespai/start_vespai_web.sh
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable vespai-web.service
sudo systemctl start vespai-web.service
sudo systemctl status vespai-web.service
```

### Performance Monitoring
```bash
# Monitor system resources
htop

# Monitor service logs
sudo journalctl -u vespai-web.service -f

# Check web access logs
# Available in web dashboard at http://localhost:8081
```

## 📚 Additional Resources

- **Research Paper**: [VespAI: Communications Biology 2024](https://doi.org/10.1038/s42003-024-05979-z)
- **Original Repository**: https://github.com/andrw3000/vespai
- **YOLOv5 Documentation**: https://docs.ultralytics.com/yolov5
- **Raspberry Pi Setup**: https://www.raspberrypi.org/documentation/

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📞 Support

**Need help?** Create an issue with:
- Operating system and version
- Python version (`python --version`)
- Hardware specs (especially for Raspberry Pi)
- Complete error messages from `vespai.log` or `sudo journalctl -u vespai-web.service`
- Steps to reproduce the problem

**For Raspberry Pi issues**, also include:
- Pi model (`cat /proc/device-tree/model`)
- OS version (`cat /etc/os-release`)
- Camera type (USB/CSI)
- Available RAM (`free -h`)
- GPU memory (`vcgencmd get_mem gpu`)