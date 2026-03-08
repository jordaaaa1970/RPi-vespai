# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VespAI is a hornet detection system that uses YOLO-based computer vision to identify Asian hornets (Vespa velutina) and European hornets (Vespa crabro) in real-time.

**Important:** This implementation is based on the research paper "VespAI: a deep learning-based system for the detection of invasive hornets" published in Communications Biology (2024), DOI: 10.1038/s42003-024-05979-z.

The system provides:

- Real-time hornet detection using YOLOv5 and ONNX-based YOLO models
- Web dashboard with live video feed and statistics
- SMS alerts via Lox24 API for hornet detections
- Motion detection optimization
- Detection logging and visualization
- Runtime switching between live camera and dataset sources

## Architecture

The system is built with a modular architecture in `src/vespai/` that provides:

1. **Computer Vision Pipeline** (`core/detection.py`): OpenCV for camera input and YOLO/ONNX object detection
2. **Web Interface** (`web/routes.py`): Flask-based dashboard with real-time statistics and video streaming
3. **Alert System** (`sms/lox24.py`): SMS notifications through Lox24 API with rate limiting
4. **Configuration Management** (`core/config.py`): Centralized configuration and validation
5. **Main Application** (`main.py`): Entry point that orchestrates all components

## Key Components

### Detection Engine
- Supports hornet-trained YOLOv5 `.pt` weights and ONNX exports
- Motion detection using background subtraction (optional)
- Configurable confidence thresholds and detection parameters
- Dataset playback from image directories, video files, and TFRecord sources

### Web Dashboard
- Live video feed at `/video_feed`
- Real-time statistics API at `/api/stats`
- Detection frame viewer at `/frame/<frame_id>`
- Responsive design with mobile optimization
- Input source switching between Camera and Dataset in the UI

### SMS Alert System
- Lox24 API integration with rate limiting (5-minute delays)
- Different alert levels for Asian vs European hornets
- Cost tracking and delivery confirmation

## Running the Application

### Basic Usage
```bash
source .venv/bin/activate
python vespai.py --web
```

Or use the included launchers:
```bash
./start_vespai.sh
./start_vespai_web.sh
```

### Command Line Options
- `--web`: Enable web dashboard
- `-c, --conf <float>`: Detection confidence threshold (default: 0.8)
- `--model-path <path>`: Path to model weights or ONNX artifact
- `--class-map <text>`: Class mapping such as `0:crabro,1:velutina` or `1:crabro,2:velutina`
- `-s, --save`: Save detection images
- `-sd, --save-dir <path>`: Directory for saved images (default: monitor/detections)
- `-v, --video <path>`: Use video file instead of camera
- `-r, --resolution <WxH>`: Camera resolution (default: 1920x1080)
- `-m, --motion`: Enable motion detection
- `-a, --min-motion-area <int>`: Minimum motion area threshold (default: 100)
- `-b, --brake <float>`: Frame processing delay (default: 0.1)
- `--dataset-delay <float>`: Minimum frame delay for finite dataset inputs
- `--web-port <int>`: Web server port (defaults to config/env)
- `-p, --print`: Print detection details to console

### Examples
```bash
# Run with web interface and motion detection
python vespai.py --web --motion --save

# Process video file with high confidence threshold
python vespai.py --video input.mp4 --conf 0.9

# Run with 720p resolution and custom save directory
python vespai.py --web --resolution 720p --save-dir ./detections

# Run the current ONNX dataset model with explicit class mapping
python vespai.py --web \
	--model-path models/L4-yolov8_asianhornet_2026-03-06_19-45-38.onnx \
	--class-map "1:crabro,2:velutina" \
	--video "datasets/Detection Asian-hornet.v1i.tfrecord/test/asianhornet.tfrecord"
```

## Dependencies

The application requires:
- Python 3.13 in the current workspace (`pyproject.toml`)
- OpenCV (cv2)
- PyTorch
- ONNX Runtime
- Flask
- RPi.GPIO (for Raspberry Pi deployment)
- psutil
- requests
- numpy

## Model Requirements

The system currently uses one of these model artifacts:
- Hornet-trained YOLOv5 weights: `models/yolov5s-all-data.pt`
- Current ONNX dataset model: `models/L4-yolov8_asianhornet_2026-03-06_19-45-38.onnx`
- Generic fallbacks: `yolov5s.pt`, `models/yolov5s.pt`

The model should be trained to detect:
- Class 0: Vespa crabro (European hornet)
- Class 1: Vespa velutina (Asian hornet)

For the current ONNX dataset model used in this workspace, the effective mapping is:
- Class 1: Vespa crabro
- Class 2: Vespa velutina

## Configuration

Key configuration constants in `src/vespai/core/config.py`:
- `LOX24_API_KEY`: SMS service API credentials
- `PHONE_NUMBER`: Target phone for alerts
- `DOMAIN_NAME`: Public domain for SMS links
- `SMS_DELAY_MINUTES`: Minimum time between SMS alerts
- `RESOLUTION`: Default camera resolution
- `FRAME_DELAY`: Live camera frame pacing
- `DATASET_FRAME_DELAY`: Dataset playback pacing
- `VESPAI_CLASS_MAP`: Runtime class-to-species override

## Web Interface

- Main dashboard: `http://localhost:8081/`
- Video stream: `http://localhost:8081/video_feed`
- Statistics API: `http://localhost:8081/api/stats`
- Detection frames: `http://localhost:8081/frame/<frame_id>`

## Deployment Notes

- Boot-time web startup is configured with `vespai-web.service`
- Service launcher script: `start_vespai_web.sh`
- Manual launcher scripts: `start_vespai.sh`, `start_vespai_web.sh`
- Service logs: `sudo journalctl -u vespai-web.service`

## Development Notes

- Modular architecture with clear separation of concerns
- Comprehensive test suite (62 tests) ensuring reliability
- In-memory storage (no database required)
- Thread-safe web frame updates using locks
- Detection frames stored temporarily (max 20 frames)
- Hourly statistics tracking with 24-hour rolling window
- Mobile-responsive web interface with honeycomb design theme
- Dataset mode now switches back to the live USB camera when the finite source is exhausted

## Hardware Requirements

- USB camera (tested with Logitech Brio)
- Raspberry Pi 4+ recommended for deployment
- GPU support optional but recommended for better performance