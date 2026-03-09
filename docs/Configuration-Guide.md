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
