
## Usage Examples

### Run Modes
```bash
# A) Webcam (live camera)
python vespai.py --web --resolution 720p --motion --conf 0.7

# A1) Force USB webcam
python vespai.py --web --camera-source usb --resolution 720p --motion --conf 0.7

# A2) Force Raspberry Pi Camera Module 3
python vespai.py --web --camera-source picamera3 --resolution 720p --motion --conf 0.7

# B) Dataset (TFRecord file) - aka DEMO mode
python vespai.py --web \
  --model-path models/L4-YOLOV26-asianhornet_2026-03-13_08-57-52.onnx \
  --class-map "1:crabro,2:velutina" \
  --video "datasets/Detection Asian-hornet.v1i.tfrecord/test/asianhornet.tfrecord" \
  --resolution 720p --conf 0.7 --print
```

### Basic Monitoring
```bash
# Start with web interface
python vespai.py --web

# Add motion detection for better performance on RPi
python vespai.py --web --motion

# Auto-select live camera by default: USB first, then the Pi camera backend if available
python vespai.py --web --camera-source auto
```

The default `vespai-web.service` boot startup now launches the dashboard with `--web --motion`.

### Raspberry Pi Camera Module 3

On Raspberry Pi OS, install the native camera userspace once and validate that the camera works:

```bash
sudo apt update
sudo apt install -y python3-picamera2
libcamera-hello
python vespai.py --web --camera-source picamera3
```

`picamera3` is a CLI alias for Camera Module 3 hardware. Internally VespAI still uses the Picamera2 library on Raspberry Pi OS.

To prefer the Pi camera without changing the command line each time:

```bash
export VESPAI_CAMERA_SOURCE=picamera3
```

### Production Deployment
```bash
# Full featured production setup - saving detections
python vespai.py --web --resolution 720p --motion --save --conf 0.7

# Same mode used by the default systemd boot service
./start_vespai_web.sh --motion
```

### Experimental Run
```bash
# Custom ONNX/class ordering
python vespai.py --web --model-path models/custom.onnx --class-map "1:crabro,2:velutina"

# Process recorded video
python vespai.py --video input.mp4 --save --conf 0.9
```


## Web Interface Access

Once started, access the dashboard:
- **Local**: http://localhost:8081
- **Network**: http://YOUR-RASPBERRY-PI-IP:8081
- **All interfaces**: http://0.0.0.0:8081

### 📱 Web Dashboard Features

#### Live Detection
- ✅ **Real-time video feed** - Smooth canvas-based display (no flickering)
- ✅ **Hornet detection** - Identifies Vespa velutina (Asian) vs Vespa crabro (European)
- ✅ **Detection overlays** - Bounding boxes with confidence scores
- ✅ **Live statistics** - Frame rate, detection counts, system status

#### Statistics & Analytics
- ✅ **Real-time counters** - Total detections, species breakdown
- ✅ **Hourly charts** - 24-hour detection history
- ✅ **Detection log** - Timestamped detection history with images
- ✅ **System monitoring** - CPU temp, RAM usage, uptime

#### Smart Features
- ✅ **SMS alerts** - Optional notifications via Lox24 API
- ✅ **Rate limiting** - Prevents alert spam (5-minute delays)
- ✅ **Cost tracking** - SMS cost monitoring
- ✅ **Motion optimization** - Only process frames with motion

### API Endpoints
- `GET /` - Main dashboard
- `GET /api/current_frame` - Current live frame as a JPEG image
- `GET /api/stats` - Real-time statistics JSON
- `POST /api/input_source` - Switch between live camera and dataset mode
- `GET /frame/<frame_id>` - Specific detection frame
- `GET /video_feed` - Legacy MJPEG live stream endpoint

### Command Line Options

```text
python vespai.py [OPTIONS]

Options:
  --web                    Enable web dashboard
  --camera-source SOURCE   Live camera backend: auto, usb, picamera2, picamera3
  -c, --conf FLOAT         Detection confidence threshold
  --model-path PATH        Path to model weights or export artifact
  --class-map TEXT         Class mapping (e.g. "0:crabro,1:velutina")
  -s, --save               Save detection images
  -sd, --save-dir PATH     Directory for saved images
  -v, --video PATH         Use video file, image directory, or TFRecord instead of live camera
  -r, --resolution WxH     Camera resolution
  -m, --motion             Enable motion detection
  -a, --min-motion-area N  Minimum motion area threshold
  -d, --dilation N         Dilation iterations for motion detection
  --dataset-delay FLOAT    Minimum frame delay for finite dataset inputs
  --web-host HOST          Web server host
  --web-port PORT          Web server port
  -b, --brake FLOAT        Frame processing delay
  -p, --print              Print detection details to console
  --sms                    Enable SMS alerts
  --no-sms                 Disable SMS alerts
```



## SMS Alerts

### Lox24 Configuration
1. Register at [Lox24](https://www.lox24.eu/)
2. Get your API credentials
3. Set in `.env`:
   ```bash
   LOX24_API_KEY=customer_number:api_key
   PHONE_NUMBER=+1234567890
   ```

### Push Notification
  < To be completed>

### Alert Behavior
- **Asian Hornet**: High priority alert sent immediately
- **European Hornet**: Lower priority info message
- **Rate Limiting**: Minimum 5-minute delay between SMS
- **Cost Tracking**: Monitors SMS costs and delivery status

