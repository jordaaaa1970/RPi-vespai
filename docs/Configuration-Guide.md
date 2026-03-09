## Configuration


### Basic Configuration

**Copy configuration template:**
```bash
cp .env.template .env
```


### Environment Variables (.env file)

**Edit configuration:**

```bash
# SMS Configuration (Lox24 API)
# VespAI Configuration
# Copy to .env and customize for your setup

# SMS Alert Configuration (Optional)
# LOX24_API_KEY=your_api_key_here
# PHONE_NUMBER=+1234567890
# DOMAIN_NAME=your-domain.com

# Model Configuration
MODEL_PATH=models/L4-yolov8_asianhornet_2026-03-06_19-45-38.onnx
CONFIDENCE_THRESHOLD=0.6
VESPAI_CLASS_MAP=1:crabro,2:velutina
VESPAI_DATASET_PATH=datasets/Detection Asian-hornet.v1i.tfrecord/test/asianhornet.tfrecord

# Camera Configuration
CAMERA_INDEX=0
VESPAI_CAMERA_DEVICE=/dev/video8
RESOLUTION=1280x720
CAMERA_RESOLUTION=1280x720
CAMERA_FPS=30

# Detection Configuration
SAVE_DETECTIONS=true
SAVE_DIRECTORY=monitor/detections
DETECTION_RETENTION_DAYS=21
FRAME_DELAY=0.35
DATASET_FRAME_DELAY=5.0

# Motion Detection (Optional)
ENABLE_MOTION_DETECTION=false
MIN_MOTION_AREA=5000

# Web Interface
WEB_HOST=0.0.0.0
WEB_PORT=8081
```
