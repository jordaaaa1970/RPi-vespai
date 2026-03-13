## 🐞 Troubleshooting

# Logs
Check `vespai.log` for manual runs, or `sudo journalctl -u vespai-web.service` for the boot service.


# Notification
**SMS not working:**
- Check API credentials in `.env`
- Verify phone number format (+country_code)
- Check Lox24 account balance

# Common Issues
## Modules missing
**"Module not found" errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or use setup script
python scripts/setup.py
```


## Model loading
**YOLO model loading errors:**
- Ensure model path is correct
- Check PyTorch installation
- Verify model compatibility

```bash
# Check model exists
ls -la models/*
```

### Camera devices
**Camera not detected:**
``` bash
ls /dev/video*

#### Try different camera indices
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

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

# Enable camera interface
sudo raspi-config
# Interface Options > Camera > Enable
# Reboot required

# Test Pi camera
libcamera-hello

# Install Picamera2 support on Raspberry Pi OS
sudo apt update
sudo apt install -y python3-picamera2

# Force the CSI camera backend for Camera Module 3
python vespai.py --web --camera-source picamera3

# Force the USB camera backend
python vespai.py --web --camera-source usb

# Check camera connection
vcgencmd get_camera
```

If both a USB webcam and the Pi Camera are connected, `--camera-source auto` prefers the USB path first. Use `--camera-source picamera3` to force Camera Module 3.



## Web interface
**Web interface not accessible:**
- Confirm port 8081 is not blocked
- Check firewall settings
- Verify Flask is running

``` bash
# Check if server is running
curl http://localhost:8081

# Check logs when started manually with nohup
tail -f vespai.log

# Check logs when started by systemd
sudo journalctl -u vespai-web.service -f

# Try different port
python vespai.py --web --web-port 5000
```


## Memory Issues
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


## Performance Monitoring
**Slow performance:**

```bash
# Monitor system resources
htop

# Monitor service logs
sudo journalctl -u vespai-web.service -f

# Check CPU frequency
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq

# Try Enable motion detection to reduce processing
python vespai.py --web --motion --min-motion-area 8000

# Try lower confidence threshold
python vespai.py --web --conf 0.6
```

**Performance issues on Raspberry Pi - CPU Throtteling?**
```bash
# Check GPU memory split
vcgencmd get_mem gpu  # Should be 128+

# Monitor temperature
vcgencmd measure_temp
```

