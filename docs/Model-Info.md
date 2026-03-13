

## 🔬 Model Information

VespAI uses a specialized YOLOv26 model trained specifically for hornet detection:

- **Model**: `yolov26` (14MB)
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