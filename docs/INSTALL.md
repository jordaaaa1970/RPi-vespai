# RPi-VespAI Installation Guide

**Complete Setup Guide for VespAI Hornet Detection System**

Since the goal is to make this a small, portable, cheap detector that is self contained, battery powered and solar charged, the installation notes focus on the Raspberry Pi 5 and Debian Distro.

It may well work in other scenarios, Win/Mac, but I have to tried that.


## 1. Dependencies Installation

### Install system dependencies
``` bash
sudo apt update && sudo apt install python3-opencv python3-pip git

# Install Git LFS to download model files properly
sudo apt install git-lfs
git lfs install



# Clone repository
git clone https://github.com/jordaaaa1970/RPi-vespai.git
cd RPi-vespai
git lfs pull



# Make setup script executable and run
chmod +x scripts/setup.sh
chmod +x scripts/raspberry-pi-setup.sh
./scripts/raspberry-pi-setup.sh

# Or manual setup with virtual environment
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements-venv-py.txt

```


---

# ⚙️ Configuration
See the Configuration-Guide.md


## 🎯 Running VespAI
**Raspberry Pi (activate virtual environment first):**
```bash
# Activate virtual environment
source .venv/bin/activate

# Start with web dashboard (optimized for Pi)
python vespai.py --web --resolution 720p --motion

# With motion detection and image saving
python vespai.py --web --resolution 720p --motion --save --conf 0.7
```


### More Examples

```bash
# A) Webcam (live camera)
python vespai.py --web --resolution 720p --motion --conf 0.7

# B) Dataset (TFRecord file) - aka DEMO Mode
python vespai.py --web \
  --model-path models/L4-yolov8_asianhornet_2026-03-06_19-45-38.onnx \
  --class-map "1:crabro,2:velutina" \
  --video "datasets/Detection Asian-hornet.v1i.tfrecord/test/asianhornet.tfrecord" \
  --resolution 720p --conf 0.7 --print

# High accuracy mode
python vespai.py --web --conf 0.9 --save

# Performance mode for Raspberry Pi --motion
python vespai.py --web --resolution 720p --motion --conf 0.7

# Process recorded video
python vespai.py --web --video /path/to/hornet_video.mp4 --save

# Debug mode
python vespai.py --web --print
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




## 🚀 Production Deployment - Auto Start at Bootup

### Systemd Service (Linux)

**Edit the file shown below!**
Change User and Group and Check the Paths!

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
```
