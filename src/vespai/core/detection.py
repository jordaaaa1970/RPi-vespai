#!/usr/bin/env python3
"""
VespAI Core Detection Module

This module contains the main detection logic for hornet identification
using YOLOv5 computer vision models.

Author: Jakob Zeise (Zeise Digital)
Version: 1.0
"""

import cv2
import time
import datetime
import numpy as np
import logging
import warnings
import json
import base64
import random
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from collections import deque
import torch

# Suppress specific PyTorch autocast deprecation warning from YOLOv5
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)
# Suppress pkg_resources deprecation warning
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)

# NOTE: Avoid monkeypatching `torch.load` globally — callers should pass
# `weights_only=False` explicitly when needed. Monkeypatching caused
# duplicate keyword argument errors in some PyTorch/loader combinations.

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Manages camera initialization and configuration for video capture.
    
    Handles different camera backends and resolution settings with fallbacks
    for cross-platform compatibility.
    """
    
    def __init__(self, resolution: Tuple[int, int] = (1920, 1080)):
        """
        Initialize camera manager.
        
        Args:
            resolution: Tuple of (width, height) for camera resolution
        """
        self.width, self.height = resolution
        self.cap: Optional[cv2.VideoCapture] = None
        self.device = None
        self.image_files: List[str] = []
        self.image_index = 0
        self.image_sequence_mode = False
        self.image_sequence_exhausted = False
        self.tfrecord_files: List[str] = []
        self.tfrecord_index = 0
        self.tfrecord_iterator = None
        self.tfrecord_mode = False
        self.tfrecord_exhausted = False
        self.current_tfrecord_file: Optional[str] = None
        self.last_frame_source: str = ""
        self.randomizer = random.SystemRandom()
    
    def initialize_camera(self, video_file: Optional[str] = None) -> cv2.VideoCapture:
        """
        Initialize camera capture with multiple backend fallbacks.
        
        Args:
            video_file: Path to video file, or None for live camera
            
        Returns:
            cv2.VideoCapture: Initialized video capture object
            
        Raises:
            RuntimeError: If no camera can be opened
        """
        # Reset source state on each initialization
        self.image_files = []
        self.image_index = 0
        self.image_sequence_mode = False
        self.image_sequence_exhausted = False
        self.tfrecord_files = []
        self.tfrecord_index = 0
        self.tfrecord_iterator = None
        self.tfrecord_mode = False
        self.tfrecord_exhausted = False
        self.current_tfrecord_file = None
        self.last_frame_source = ""

        if video_file:
            import os
            if not os.path.exists(video_file):
                raise RuntimeError(f"Video file not found: {video_file}")

            if os.path.isdir(video_file):
                tfrecord_files = self._discover_tfrecord_files(video_file)
                if tfrecord_files:
                    logger.info("Opening TFRecord dataset directory: %s", video_file)
                    self.tfrecord_files = list(tfrecord_files)
                    self.randomizer.shuffle(self.tfrecord_files)
                    self._advance_tfrecord_iterator()
                    self.tfrecord_mode = True
                    self.device = f"tfrecord_dir:{video_file}"
                    logger.info("Loaded %d TFRecord files for dataset playback", len(self.tfrecord_files))
                    logger.info("TFRecord dataset initialized successfully")
                    return self.cap

            if video_file.lower().endswith('.tfrecord'):
                logger.info("Opening TFRecord dataset file: %s", video_file)
                self.tfrecord_files = [video_file]
                self._advance_tfrecord_iterator()
                self.tfrecord_mode = True
                self.device = f"tfrecord:{video_file}"
                logger.info("TFRecord playback initialized successfully")
                return self.cap

            if os.path.isdir(video_file):
                logger.info("Opening image dataset directory: %s", video_file)
                supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
                entries = sorted(os.listdir(video_file))
                self.image_files = [
                    os.path.join(video_file, entry)
                    for entry in entries
                    if os.path.splitext(entry)[1].lower() in supported_ext
                ]

                if not self.image_files:
                    raise RuntimeError(f"No supported image files found in directory: {video_file}")

                self.randomizer.shuffle(self.image_files)

                self.image_sequence_mode = True
                self.device = f"image_dir:{video_file}"
                logger.info("Loaded %d images for dataset playback", len(self.image_files))
            else:
                logger.info("Opening video file: %s", video_file)
                self.cap = cv2.VideoCapture(video_file)
                if not self.cap.isOpened():
                    raise RuntimeError(f"Failed to open video file: {video_file}")
        else:
            logger.info("Initializing camera with resolution %dx%d", self.width, self.height)

            # Allow overriding camera device with env var (useful for libcamera v4l2 nodes)
            import os
            env_dev = os.environ.get('VESPAI_CAMERA_DEVICE')

            # Prefer explicit override first, then detected USB webcam nodes,
            # then libcamera compatibility nodes commonly used on Raspberry Pi.
            preferred_nodes = [env_dev] if env_dev else []
            preferred_nodes += self._discover_usb_video_nodes()
            preferred_nodes += ["/dev/video0", "/dev/video8", "/dev/video23", "/dev/video24", "/dev/video25", "/dev/video26"]

            # Preserve order while removing duplicates/empty entries
            preferred_nodes = list(dict.fromkeys([node for node in preferred_nodes if node]))

            # Build a list of (device, backend) candidates to try
            candidates: List[Tuple[Any, Optional[int]]] = []
            for dev in preferred_nodes:
                if not dev:
                    continue
                # if dev looks like a path, use V4L2 backend
                if isinstance(dev, str) and dev.startswith('/dev/video'):
                    candidates.append((dev, cv2.CAP_V4L2))

            # Fallback generic candidates for other platforms
            candidates += [
                (0, cv2.CAP_V4L2),      # Linux index 0
                (0, cv2.CAP_DSHOW),     # Windows DirectShow
                (0, cv2.CAP_AVFOUNDATION),  # macOS
                (0, None),              # Default backend
            ]

            # Try each candidate until a capture device opens
            for device, backend in candidates:
                try:
                    if backend is not None:
                        self.cap = cv2.VideoCapture(device, backend)
                    else:
                        self.cap = cv2.VideoCapture(device)

                    if self.cap.isOpened():
                        # remember which device path/index we opened
                        self.device = device
                        logger.info("Camera opened with device %s, backend %s", device, backend)
                        break
                except Exception as e:
                    logger.debug("Failed to open camera with device %s, backend %s: %s", device, backend, e)
                    continue
            
            if not self.cap or not self.cap.isOpened():
                raise RuntimeError("Cannot open camera with any backend")
            
            # Configure camera properties
            self._configure_camera()
        
        if self.image_sequence_mode:
            logger.info("Image dataset initialized successfully")
            return self.cap

        if self.tfrecord_mode:
            logger.info("TFRecord dataset initialized successfully")
            return self.cap

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Failed to initialize video capture")
            
        logger.info("Camera initialized successfully")
        # Reduced stabilization time for better performance
        time.sleep(0.5)  # Quick stabilization
        return self.cap

    def _discover_usb_video_nodes(self) -> List[str]:
        """Discover likely USB webcam capture nodes from sysfs (e.g. /dev/video8)."""
        import glob
        import os

        nodes: List[str] = []
        for video_dir in sorted(glob.glob('/sys/class/video4linux/video*')):
            node_name = os.path.basename(video_dir)
            dev_path = f"/dev/{node_name}"

            try:
                device_name_path = os.path.join(video_dir, 'name')
                with open(device_name_path, 'r', encoding='utf-8', errors='ignore') as handle:
                    device_name = handle.read().strip().lower()
            except Exception:
                device_name = ''

            try:
                driver_link = os.path.realpath(os.path.join(video_dir, 'device', 'driver'))
            except Exception:
                driver_link = ''

            is_uvc = 'uvcvideo' in driver_link
            looks_like_camera = any(token in device_name for token in ['webcam', 'camera', 'hd webcam'])

            if is_uvc or looks_like_camera:
                nodes.append(dev_path)

        return nodes
    
    def _configure_camera(self):
        """Configure camera properties for optimal capture."""
        if not self.cap:
            return
        # Avoid forcing pixel formats on libcamera-provided compatibility nodes,
        # which may not support MJPG negotiation. Only set FOURCC for generic V4L2 devices.
        try:
            is_libcamera_node = isinstance(self.device, str) and any(x in str(self.device) for x in ['video23', 'video24', 'video25', 'video26'])
        except Exception:
            is_libcamera_node = False

        if not is_libcamera_node:
            # Set MJPEG codec for better performance on generic devices
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Set frame rate
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Log actual settings
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info("Camera configured - Resolution: %dx%d, FPS: %.1f", 
                   actual_width, actual_height, actual_fps)
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple of (success, frame) where success is bool and frame is numpy array
        """
        if self.image_sequence_mode:
            while self.image_index < len(self.image_files):
                image_path = self.image_files[self.image_index]
                self.image_index += 1

                frame = cv2.imread(image_path)
                if frame is None:
                    logger.warning("Failed to read dataset image: %s", image_path)
                    continue

                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

                if self.image_index >= len(self.image_files):
                    self.image_sequence_exhausted = True

                self.last_frame_source = (
                    f"image:{Path(image_path).name} ({self.image_index}/{len(self.image_files)})"
                )

                return True, frame

            self.image_sequence_exhausted = True
            return False, None

        if self.tfrecord_mode:
            frame = self._read_tfrecord_frame()
            if frame is None:
                self.tfrecord_exhausted = True
                return False, None
            return True, frame
        if not self.cap:
            return False, None
            
        success, frame = self.cap.read()
        if success and frame is not None:
            self.last_frame_source = f"camera:{self.device}"
        return success, frame

    def source_exhausted(self) -> bool:
        """Return True when a finite input source (image sequence) is exhausted."""
        return (
            (self.image_sequence_mode and self.image_sequence_exhausted)
            or (self.tfrecord_mode and self.tfrecord_exhausted)
        )

    def is_finite_source(self) -> bool:
        """Return True when source is finite (images or TFRecord dataset)."""
        return self.image_sequence_mode or self.tfrecord_mode

    def get_last_frame_source(self) -> str:
        """Return a human-readable source string for the most recently read frame."""
        return self.last_frame_source or "unknown"

    def _discover_tfrecord_files(self, directory: str) -> List[str]:
        """Recursively find TFRecord files in a directory."""
        import os

        found: List[str] = []
        for root, _, files in os.walk(directory):
            for file_name in files:
                if file_name.lower().endswith('.tfrecord'):
                    found.append(os.path.join(root, file_name))
        return sorted(found)

    def _advance_tfrecord_iterator(self):
        """Advance to the next TFRecord file iterator."""
        from tfrecord.reader import tfrecord_loader

        while self.tfrecord_index < len(self.tfrecord_files):
            tfrecord_path = self.tfrecord_files[self.tfrecord_index]
            self.tfrecord_index += 1
            try:
                self.tfrecord_iterator = iter(
                    tfrecord_loader(tfrecord_path, index_path=None, description=None)
                )
                self.current_tfrecord_file = tfrecord_path
                logger.info("Reading TFRecord file: %s", tfrecord_path)
                self._apply_random_tfrecord_offset(tfrecord_path)
                return
            except Exception as error:
                logger.warning("Failed to open TFRecord file %s: %s", tfrecord_path, error)
                self.tfrecord_iterator = None

        self.tfrecord_iterator = None

    def _apply_random_tfrecord_offset(self, tfrecord_path: str):
        """Skip a random number of TFRecord examples so restarts don't begin on the same frame."""
        if self.tfrecord_iterator is None:
            return

        skip_count = self.randomizer.randint(0, 24)
        skipped = 0
        while skipped < skip_count:
            try:
                next(self.tfrecord_iterator)
                skipped += 1
            except StopIteration:
                break
            except Exception as error:
                logger.warning("Failed while applying random TFRecord offset for %s: %s", tfrecord_path, error)
                break

        if skipped > 0:
            logger.info("Applied random TFRecord start offset: skipped %d frames", skipped)

    def _read_tfrecord_frame(self) -> Optional[np.ndarray]:
        """Read next image frame from TFRecord stream."""
        import numpy as np

        while True:
            if self.tfrecord_iterator is None:
                self._advance_tfrecord_iterator()
                if self.tfrecord_iterator is None:
                    return None

            try:
                example = next(self.tfrecord_iterator)
            except StopIteration:
                self.tfrecord_iterator = None
                continue
            except Exception as error:
                logger.warning("Error reading TFRecord example: %s", error)
                self.tfrecord_iterator = None
                continue

            encoded = example.get('image/encoded') if isinstance(example, dict) else None
            if not isinstance(encoded, (bytes, bytearray)):
                continue

            buffer = np.frombuffer(encoded, dtype=np.uint8)
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            image_name = "unknown"
            if isinstance(example, dict):
                filename_value = example.get('image/filename')
                if isinstance(filename_value, (bytes, bytearray)):
                    try:
                        image_name = filename_value.decode('utf-8', errors='ignore')
                    except Exception:
                        image_name = "unknown"

            file_part = Path(self.current_tfrecord_file).name if self.current_tfrecord_file else "unknown.tfrecord"
            self.last_frame_source = f"tfrecord:{file_part}:{image_name}"

            return frame
    
    def release(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
            logger.info("Camera released")


class ModelManager:
    """
    Manages YOLO model loading (YOLOv5 and YOLOv8) with fallback methods.
    
    Handles different loading approaches for better compatibility across
    different environments and installations.
    """
    
    def __init__(self, model_path: str, confidence: float = 0.8):
        """
        Initialize model manager.
        
        Args:
            model_path: Path to YOLOv5 model weights
            confidence: Detection confidence threshold
        """
        self.model_path = model_path
        self.confidence = confidence
        self.model = None
        self.class_names = {}
        self.model_family = "yolov5"
        self.onnx_session = None
        self.onnx_input_name = None
        self.onnx_input_scale_mode = "auto"
    
    def load_model(self) -> Any:
        """
        Load YOLO model with multiple fallback methods.
        
        Returns:
            Loaded YOLOv5 model object
            
        Raises:
            RuntimeError: If model cannot be loaded
        """
        logger.info("Loading YOLO model from: %s", self.model_path)
        
        if not self._find_model_file():
            raise RuntimeError(f"Model file not found: {self.model_path}")

        # Ultralytics does not load .keras detect models directly in this runtime.
        # If a sibling supported artifact exists, use it automatically.
        if str(self.model_path).lower().endswith('.keras'):
            alternative_model = self._resolve_keras_alternative(self.model_path)
            if alternative_model:
                logger.info("Using supported YOLOv8 artifact instead of .keras: %s", alternative_model)
                self.model_path = alternative_model
            else:
                raise RuntimeError(
                    "The provided model is a .keras file, which is not directly supported by the "
                    "current Ultralytics inference backend. Provide/export one of: .pt, .onnx, "
                    ".tflite, .engine, or SavedModel directory with the same model."
                )

        if self._is_l4_keras_weights_pt(self.model_path):
            raise RuntimeError(
                "The selected .pt file is an L4 Keras-weights container, not a native Ultralytics "
                "YOLO checkpoint. Export/provide a deployable artifact instead: Ultralytics .pt, "
                ".onnx, .tflite, .engine, or TensorFlow SavedModel."
            )

        if self._is_nhwc_onnx_model(self.model_path):
            loading_methods = [self._load_nhwc_onnx_runtime]
        elif self._is_yolov8_model_path(self.model_path):
            loading_methods = [self._load_via_ultralytics]
        else:
            # Choose loading strategy based on model format/path
            loading_methods = [
                self._load_via_yolov5_package,
                self._load_via_local_directory,
                self._load_via_github,
                self._load_fallback_model
            ]
        
        generic_fallback_model = None

        for method in loading_methods:
            try:
                logger.info("Trying model loading method: %s", method.__name__)
                self.model = method()
                if self.model is not None:
                    try:
                        self._configure_model()
                        logger.info("✓ Model loaded successfully via %s", method.__name__)
                        return self.model
                    except RuntimeError as config_error:
                        # Keep a generic model candidate as a last-resort fallback
                        # instead of failing all loading methods.
                        if "does not appear to be hornet-specific" in str(config_error):
                            logger.warning("✗ Model from %s is generic, keeping as fallback", method.__name__)
                            generic_fallback_model = self.model
                            continue
                        raise
                else:
                    logger.warning("✗ Method %s returned None", method.__name__)
            except Exception as e:
                logger.warning("✗ Loading method %s failed: %s", method.__name__, e)
                continue

        if generic_fallback_model is not None:
            self.model = generic_fallback_model
            # Ensure threshold is still applied on fallback model.
            if hasattr(self.model, 'conf'):
                self.model.conf = self.confidence
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            logger.warning(
                "Using generic YOLO model fallback. For hornet detection accuracy, "
                "provide hornet-trained weights with classes like crabro/velutina."
            )
            return self.model
        
        raise RuntimeError("Failed to load model with any method")

    def _resolve_keras_alternative(self, keras_path: str) -> Optional[str]:
        """Find a supported artifact next to a .keras model, if available."""
        model_path = Path(keras_path)
        model_stem = model_path.with_suffix('')

        candidates = [
            f"{model_stem}.pt",
            f"{model_stem}.onnx",
            f"{model_stem}.tflite",
            f"{model_stem}.engine",
            f"{model_stem}_saved_model",
            f"{model_stem}.saved_model",
        ]

        for candidate in candidates:
            candidate_path = Path(candidate)
            if candidate_path.exists():
                return str(candidate_path)

        return None

    def _is_l4_keras_weights_pt(self, model_path: str) -> bool:
        """Detect L4-exported Keras weight containers saved with a .pt extension."""
        if not model_path or not str(model_path).lower().endswith('.pt'):
            return False

        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except Exception:
            return False

        if not isinstance(checkpoint, dict):
            return False

        model_type = str(checkpoint.get('model_type', '')).lower()
        weights = checkpoint.get('weights')
        if model_type != 'yolov8' or not isinstance(weights, dict):
            return False

        sample_keys = list(weights.keys())[:10]
        return any(str(key).startswith('functional_') for key in sample_keys)

    def _is_nhwc_onnx_model(self, model_path: str) -> bool:
        """Detect ONNX models with channel-last input layout not supported by current pipeline."""
        if not model_path or not str(model_path).lower().endswith('.onnx'):
            return False

        try:
            import onnxruntime as ort
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            inputs = session.get_inputs()
            if not inputs:
                return False

            input_shape = inputs[0].shape
            if len(input_shape) != 4:
                return False

            last_dim = input_shape[-1]
            # For symbolic dims, compare string value as well.
            if last_dim == 3 or str(last_dim) == '3':
                return True
        except Exception:
            return False

        return False

    def _is_yolov8_model_path(self, model_path: str) -> bool:
        """Return True if the model path suggests a YOLOv8/Ultralytics export."""
        if not model_path:
            return False

        model_path_lower = str(model_path).lower()
        return (
            "yolov8" in model_path_lower
            or model_path_lower.endswith(".keras")
            or model_path_lower.endswith(".onnx")
            or model_path_lower.endswith(".engine")
            or model_path_lower.endswith(".tflite")
            or model_path_lower.endswith(".saved_model")
        )
    
    def _find_model_file(self) -> bool:
        """
        Find the model file using fallback paths.
        
        Returns:
            bool: True if model file found and updated self.model_path
        """
        import os

        if self.model_path and os.path.exists(self.model_path):
            return True

        # Resolve repository root from this file location:
        # src/vespai/core/detection.py -> repo root is parents[3]
        repo_root = Path(__file__).resolve().parents[3]
        
        # Try alternative paths
        alternative_paths = [
            str(repo_root / "models" / "L4-yolov8_asianhornet_2026-02-25_08-31-37.keras"),
            "/opt/vespai/models/yolov5s-all-data.pt",
            str(repo_root / "models" / "yolov5s-all-data.pt"),
            str(repo_root / "models" / "yolov5s-official.pt"),
            str(repo_root / "yolov5s.pt"),
            str(repo_root / "models" / "yolov5s.pt"),
            "models/yolov5s-all-data.pt", 
            "yolov5s-all-data.pt",
            "yolov5s.pt",
            "models/yolov5s.pt",
            os.path.join(os.getcwd(), "..", "models", "yolov5s-all-data.pt"),
            os.path.join(os.getcwd(), "..", "yolov5s.pt"),
            os.path.join(os.getcwd(), "yolov5s.pt")
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                logger.info("Using alternative model path: %s", path)
                self.model_path = path
                return True
        
        return False

    def _load_via_ultralytics(self):
        """Load model via Ultralytics YOLO (supports YOLOv8 and exported formats)."""
        from ultralytics import YOLO

        self.model_family = "yolov8"
        return YOLO(self.model_path)

    def _load_nhwc_onnx_runtime(self):
        """Load NHWC ONNX model with direct ONNXRuntime backend."""
        import onnxruntime as ort

        providers = ['CPUExecutionProvider']
        self.onnx_session = ort.InferenceSession(self.model_path, providers=providers)
        inputs = self.onnx_session.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX model has no inputs")

        self.onnx_input_name = inputs[0].name
        self.model_family = "onnx_nhwc"

        # Use session object as model marker for existing checks
        self.model = self.onnx_session
        self.class_names = self._load_onnx_class_names()
        return self.onnx_session

    def _load_onnx_class_names(self) -> Dict[int, str]:
        """Load class names from sidecar metadata or return generic labels."""
        model_path = Path(self.model_path)
        metadata_path = model_path.with_name(f"{model_path.stem}_metadata.json")

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as handle:
                    metadata = json.load(handle)

                for key in ('class_names', 'names', 'labels', 'classes'):
                    names = metadata.get(key)
                    if isinstance(names, list):
                        return {index: str(name) for index, name in enumerate(names)}
                    if isinstance(names, dict):
                        normalized: Dict[int, str] = {}
                        for item_key, value in names.items():
                            try:
                                normalized[int(item_key)] = str(value)
                            except Exception:
                                continue
                        if normalized:
                            return normalized
            except Exception as error:
                logger.warning("Failed reading ONNX metadata sidecar: %s", error)

        # Derive class count from ONNX output shape if available.
        if self.onnx_session is not None:
            outputs = self.onnx_session.get_outputs()
            if len(outputs) >= 2 and len(outputs[1].shape) == 3:
                class_dim = outputs[1].shape[-1]
                if isinstance(class_dim, int) and class_dim > 0:
                    return {index: f"class{index}" for index in range(class_dim)}

        return {0: 'class0', 1: 'class1'}
    
    def _load_via_yolov5_package(self):
        """Load model using the yolov5 package."""
        import yolov5
        return yolov5.load(self.model_path, device='cpu')
    
    def _load_via_local_directory(self):
        """Load model from local YOLOv5 directory."""
        import os
        import sys
        import torch

        repo_root = Path(__file__).resolve().parents[3]
        yolo_candidates = [
            repo_root / "models" / "ultralytics_yolov5_master",
            Path(os.getcwd()) / "models" / "ultralytics_yolov5_master",
            Path(os.getcwd()).parent / "models" / "ultralytics_yolov5_master",
        ]

        yolo_dir = next((candidate for candidate in yolo_candidates if candidate.exists()), None)
        if yolo_dir is None:
            raise RuntimeError("Local YOLOv5 directory not found")

        yolo_dir_str = str(yolo_dir)
        if yolo_dir_str not in sys.path:
            sys.path.insert(0, yolo_dir_str)

        return torch.hub.load(yolo_dir_str, 'custom',
                             path=self.model_path,
                             source='local',
                             force_reload=False,
                             _verbose=False)
    
    def _load_via_github(self):
        """Load model from GitHub repository."""
        import torch
        
        try:
            # Try with safe globals first
            import torch.serialization
            with torch.serialization.safe_globals(['models.yolo.DetectionModel']):
                return torch.hub.load('ultralytics/yolov5', 'custom',
                                     path=self.model_path,
                                     force_reload=True,
                                     trust_repo=True,
                                     skip_validation=True,
                                     _verbose=False)
        except Exception as e:
            logger.warning("Safe loading failed, trying direct method: %s", e)
            # Direct torch.load with weights_only=False
            try:
                import torch
                model_data = torch.load(self.model_path, map_location='cpu', weights_only=False)
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False, _verbose=False)
                model.load_state_dict(model_data['model'].state_dict())
                return model
            except Exception as e2:
                logger.warning("Direct loading also failed: %s", e2)
                raise e
    
    def _load_fallback_model(self):
        """Load a standard YOLOv5s model as fallback."""
        import torch
        logger.info("Loading standard YOLOv5s model as fallback")
        return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, _verbose=False)
    
    def _configure_model(self):
        """Configure model after loading."""
        if not self.model:
            return

        if self.model_family == "yolov5" and hasattr(self.model, 'conf'):
            self.model.conf = self.confidence
        
        # Extract class names
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
            logger.info("Model classes: %s", self.class_names)

        if not self._is_hornet_model(self.class_names):
            import os
            allow_generic = os.environ.get('VESPAI_ALLOW_GENERIC_MODEL', '0') == '1'
            message = (
                "Loaded model does not appear to be hornet-specific (expected classes like "
                "'crabro'/'velutina'). Current classes look generic (e.g. COCO)."
            )
            if allow_generic:
                logger.warning("%s Continuing because VESPAI_ALLOW_GENERIC_MODEL=1", message)
            else:
                raise RuntimeError(
                    f"{message} Set VESPAI_ALLOW_GENERIC_MODEL=1 to force generic model, "
                    "or provide hornet-trained weights."
                )
        
        # Log model info
        if hasattr(self.model, 'yaml'):
            logger.debug("Model config: %s", self.model.yaml)

    def _is_hornet_model(self, names: Any) -> bool:
        """Return True when class names appear to represent hornet classes."""
        if not names:
            return False

        if isinstance(names, dict):
            values = [str(value).lower() for value in names.values()]
        else:
            values = [str(value).lower() for value in names]

        joined = " ".join(values)
        has_velutina = 'velutina' in joined
        has_crabro = 'crabro' in joined
        has_vespa = 'vespa' in joined

        # Accept known hornet labeling styles
        if has_velutina and has_crabro:
            return True
        if has_vespa and (has_velutina or has_crabro):
            return True

        return False
    
    def predict(self, frame: np.ndarray):
        """
        Run inference on a frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Model predictions
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        if self.model_family == "onnx_nhwc":
            return self._predict_onnx_nhwc(frame)

        if self.model_family == "yolov8":
            return self.model.predict(source=frame, conf=self.confidence, verbose=False)

        # Convert BGR to RGB for YOLOv5
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.model(rgb_frame)

    def _predict_onnx_nhwc(self, frame: np.ndarray):
        """Run direct ONNXRuntime inference for NHWC YOLOv8-style models."""
        if self.onnx_session is None or self.onnx_input_name is None:
            raise RuntimeError("ONNX session not initialized")

        original_h, original_w = frame.shape[:2]
        safe_h = max(32, (original_h // 32) * 32)
        safe_w = max(32, (original_w // 32) * 32)

        resized_frame = frame
        if safe_h != original_h or safe_w != original_w:
            resized_frame = cv2.resize(frame, (safe_w, safe_h), interpolation=cv2.INTER_LINEAR)

        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        base_input = rgb_frame.astype(np.float32)

        if self.onnx_input_scale_mode == "auto":
            self.onnx_input_scale_mode = self._select_onnx_input_scale_mode(base_input)

        if self.onnx_input_scale_mode == "raw":
            input_tensor = base_input
        else:
            input_tensor = base_input / 255.0

        input_tensor = np.expand_dims(input_tensor, axis=0)  # NHWC

        outputs = self.onnx_session.run(None, {self.onnx_input_name: input_tensor})
        if len(outputs) < 2:
            return {'pred_tuples': []}

        box_output = outputs[0]   # (1, N, 4*reg_max)
        class_output = outputs[1]  # (1, N, num_classes)

        debug_summary = ""
        top_class_id: Optional[int] = None
        top_class_conf: Optional[float] = None
        try:
            class_scores_raw = np.array(class_output)
            if class_scores_raw.size > 0 and class_scores_raw.shape[-1] > 0:
                class_scores = class_scores_raw.reshape(-1, class_scores_raw.shape[-1])
                class_min = float(np.min(class_scores))
                class_max = float(np.max(class_scores))
                if not (class_min >= 0.0 and class_max <= 1.0):
                    class_scores = 1.0 / (1.0 + np.exp(-class_scores))

                per_class_max = np.max(class_scores, axis=0)
                if per_class_max.size > 0:
                    top_class_id = int(np.argmax(per_class_max))
                    top_class_conf = float(per_class_max[top_class_id])
                top_indices = np.argsort(per_class_max)[::-1][:3]
                top_parts: List[str] = []
                for class_id in top_indices:
                    class_id_int = int(class_id)
                    label = str(self.class_names.get(class_id_int, f"class{class_id_int}"))
                    top_parts.append(f"{label}:{float(per_class_max[class_id_int]):.2f}")
                debug_summary = " | ".join(top_parts)
        except Exception:
            debug_summary = ""

        predictions = self._decode_onnx_yolov8_outputs(
            box_output,
            class_output,
            image_height=safe_h,
            image_width=safe_w,
            conf_threshold=self.confidence,
        )
        top_prediction = self._decode_best_onnx_yolov8_prediction(
            box_output,
            class_output,
            image_height=safe_h,
            image_width=safe_w,
        )

        if predictions and (safe_h != original_h or safe_w != original_w):
            scale_x = original_w / float(safe_w)
            scale_y = original_h / float(safe_h)
            scaled_predictions: List[Tuple[float, float, float, float, float, float]] = []
            for x1, y1, x2, y2, conf, cls in predictions:
                scaled_predictions.append((
                    float(np.clip(x1 * scale_x, 0, original_w - 1)),
                    float(np.clip(y1 * scale_y, 0, original_h - 1)),
                    float(np.clip(x2 * scale_x, 0, original_w - 1)),
                    float(np.clip(y2 * scale_y, 0, original_h - 1)),
                    float(conf),
                    float(cls),
                ))
            predictions = scaled_predictions

        if top_prediction and (safe_h != original_h or safe_w != original_w):
            x1, y1, x2, y2, conf, cls = top_prediction
            scale_x = original_w / float(safe_w)
            scale_y = original_h / float(safe_h)
            top_prediction = (
                float(np.clip(x1 * scale_x, 0, original_w - 1)),
                float(np.clip(y1 * scale_y, 0, original_h - 1)),
                float(np.clip(x2 * scale_x, 0, original_w - 1)),
                float(np.clip(y2 * scale_y, 0, original_h - 1)),
                float(conf),
                float(cls),
            )

        return {
            'pred_tuples': predictions,
            'debug_summary': debug_summary,
            'top_class_id': top_class_id,
            'top_class_conf': top_class_conf,
            'top_prediction': top_prediction,
        }

    def _select_onnx_input_scale_mode(self, base_input: np.ndarray) -> str:
        """Choose ONNX input scale mode by probing class-output signal strength."""
        if self.onnx_session is None or self.onnx_input_name is None:
            return "norm"

        try:
            raw_tensor = np.expand_dims(base_input, axis=0)
            norm_tensor = np.expand_dims(base_input / 255.0, axis=0)

            raw_outputs = self.onnx_session.run(None, {self.onnx_input_name: raw_tensor})
            norm_outputs = self.onnx_session.run(None, {self.onnx_input_name: norm_tensor})

            raw_class = np.array(raw_outputs[1]) if len(raw_outputs) > 1 else np.array([])
            norm_class = np.array(norm_outputs[1]) if len(norm_outputs) > 1 else np.array([])

            raw_max = float(np.max(raw_class)) if raw_class.size > 0 else 0.0
            norm_max = float(np.max(norm_class)) if norm_class.size > 0 else 0.0

            if raw_max > (norm_max * 20.0) and raw_max > 0.01:
                logger.info(
                    "ONNX input scaling auto-selected: raw (raw_max=%.6f, norm_max=%.6f)",
                    raw_max,
                    norm_max,
                )
                return "raw"

            logger.info(
                "ONNX input scaling auto-selected: normalized (raw_max=%.6f, norm_max=%.6f)",
                raw_max,
                norm_max,
            )
            return "norm"
        except Exception as error:
            logger.warning("Failed ONNX input scale auto-detection, using normalized input: %s", error)
            return "norm"

    def _decode_onnx_yolov8_outputs(
        self,
        box_output: np.ndarray,
        class_output: np.ndarray,
        image_height: int,
        image_width: int,
        conf_threshold: float,
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """Decode YOLOv8 DFL outputs from ONNXRuntime into xyxy/conf/class tuples."""
        if box_output.ndim != 3 or class_output.ndim != 3:
            return []

        box_output = box_output[0]
        class_output = class_output[0]

        if box_output.shape[0] != class_output.shape[0] or box_output.shape[0] == 0:
            return []

        num_predictions = box_output.shape[0]
        reg_channels = box_output.shape[1]
        if reg_channels % 4 != 0:
            return []

        reg_max = reg_channels // 4
        if reg_max <= 0:
            return []

        anchor_points, stride_values = self._build_yolov8_anchors(image_height, image_width, num_predictions)
        if anchor_points.shape[0] != num_predictions:
            return []

        # Decode DFL distances: [N, 4*reg_max] -> [N, 4]
        dfl = box_output.reshape(num_predictions, 4, reg_max)
        dfl = dfl - np.max(dfl, axis=2, keepdims=True)
        dfl = np.exp(dfl)
        dfl = dfl / (np.sum(dfl, axis=2, keepdims=True) + 1e-9)

        bins = np.arange(reg_max, dtype=np.float32)
        distances = np.sum(dfl * bins[None, None, :], axis=2)

        strides = stride_values.reshape(-1, 1)
        left = distances[:, 0:1] * strides
        top = distances[:, 1:2] * strides
        right = distances[:, 2:3] * strides
        bottom = distances[:, 3:4] * strides

        x_center = anchor_points[:, 0:1] * strides
        y_center = anchor_points[:, 1:2] * strides

        x1 = x_center - left
        y1 = y_center - top
        x2 = x_center + right
        y2 = y_center + bottom

        xyxy = np.concatenate([x1, y1, x2, y2], axis=1)
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, image_width - 1)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, image_height - 1)

        # Some ONNX exports output probabilities directly, others output logits.
        # Only apply sigmoid when values are outside [0, 1].
        class_min = float(np.min(class_output))
        class_max = float(np.max(class_output))
        if class_min >= 0.0 and class_max <= 1.0:
            class_scores = class_output
        else:
            class_scores = 1.0 / (1.0 + np.exp(-class_output))
        best_class = np.argmax(class_scores, axis=1)
        best_conf = class_scores[np.arange(num_predictions), best_class]

        keep = best_conf >= conf_threshold
        if not np.any(keep):
            return []

        xyxy = xyxy[keep]
        confs = best_conf[keep]
        classes = best_class[keep]

        keep_indices = self._nms_xyxy(xyxy, confs, iou_threshold=0.45)

        results: List[Tuple[float, float, float, float, float, float]] = []
        for index in keep_indices:
            box = xyxy[index]
            results.append((
                float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                float(confs[index]), float(classes[index])
            ))
        return results

    def _decode_best_onnx_yolov8_prediction(
        self,
        box_output: np.ndarray,
        class_output: np.ndarray,
        image_height: int,
        image_width: int,
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Return the single highest-confidence ONNX prediction without threshold filtering."""
        predictions = self._decode_onnx_yolov8_outputs(
            box_output,
            class_output,
            image_height=image_height,
            image_width=image_width,
            conf_threshold=0.0,
        )
        if not predictions:
            return None
        return max(predictions, key=lambda item: item[4])

    def _build_yolov8_anchors(self, image_height: int, image_width: int, expected_count: int):
        """Build YOLOv8 anchor points/strides for standard detect heads."""
        anchors: List[np.ndarray] = []
        strides: List[np.ndarray] = []

        for stride in (8, 16, 32):
            grid_h = image_height // stride
            grid_w = image_width // stride
            if grid_h <= 0 or grid_w <= 0:
                continue

            yv, xv = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
            points = np.stack((xv + 0.5, yv + 0.5), axis=-1).reshape(-1, 2).astype(np.float32)
            anchors.append(points)
            strides.append(np.full((points.shape[0],), stride, dtype=np.float32))

        if not anchors:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        anchor_points = np.concatenate(anchors, axis=0)
        stride_values = np.concatenate(strides, axis=0)

        # Guard for non-standard models by clipping/padding to expected size.
        if anchor_points.shape[0] > expected_count:
            anchor_points = anchor_points[:expected_count]
            stride_values = stride_values[:expected_count]
        elif anchor_points.shape[0] < expected_count:
            pad_count = expected_count - anchor_points.shape[0]
            pad_anchor = np.repeat(anchor_points[-1:, :], pad_count, axis=0)
            pad_stride = np.repeat(stride_values[-1:], pad_count, axis=0)
            anchor_points = np.concatenate([anchor_points, pad_anchor], axis=0)
            stride_values = np.concatenate([stride_values, pad_stride], axis=0)

        return anchor_points, stride_values

    def _nms_xyxy(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
        """Simple class-agnostic NMS over xyxy boxes."""
        if boxes.size == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1))
        order = scores.argsort()[::-1]
        keep: List[int] = []

        while order.size > 0:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[current], x1[order[1:]])
            yy1 = np.maximum(y1[current], y1[order[1:]])
            xx2 = np.minimum(x2[current], x2[order[1:]])
            yy2 = np.minimum(y2[current], y2[order[1:]])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            intersection = inter_w * inter_h

            union = areas[current] + areas[order[1:]] - intersection + 1e-9
            iou = intersection / union

            remaining = np.where(iou <= iou_threshold)[0]
            order = order[remaining + 1]

        return keep


class DetectionProcessor:
    """
    Processes detection results and manages statistics.
    
    Handles detection counting, confidence tracking, and frame annotation.
    """
    
    def __init__(self):
        """Initialize detection processor."""
        self.class_names: Dict[int, str] = {}
        self.class_species_map: Dict[int, str] = {}
        self.class_mapping_overridden = False
        self.unmapped_classes_seen = set()
        self.stats = {
            "frame_id": 0,
            "total_bee": 0,
            "total_velutina": 0,
            "total_crabro": 0,
            "total_wasp": 0,
            "total_detections": 0,
            "fps": 0,
            "current_frame_source": "",
            "model_debug_summary": "",
            "last_detection_preview": "",
            "last_detection_preview_frame_id": "",
            "last_detection_time": None,
            "last_bee_time": None,
            "last_velutina_time": None,
            "last_crabro_time": None,
            "last_wasp_time": None,
            "start_time": datetime.datetime.now(),
            "detection_log": deque(maxlen=20),
            "detection_frames": {},
            "inference_timing_recent": deque(maxlen=20),
            "last_inference_ms": 0.0,
            "inference_count": 0,
            "inference_total_ms": 0.0,
            "inference_avg_ms": 0.0,
            "inference_min_ms": 0.0,
            "inference_max_ms": 0.0,
            "confidence_avg": 0,
        }
        
        self.hourly_detections = {hour: {"velutina": 0, "crabro": 0} for hour in range(24)}
        self.current_hour = datetime.datetime.now().hour

    def set_class_names(self, class_names: Any, class_map_override: str = ""):
        """Set model class names and build class-id to species mapping."""
        self.class_names = self._normalize_class_names(class_names)
        self.class_species_map = {}

        for class_id, label in self.class_names.items():
            species = self._map_label_to_species(label)
            if species:
                self.class_species_map[class_id] = species

        override_map = self._parse_class_map_override(class_map_override)
        override_map = self._normalize_override_indices(override_map)
        if override_map:
            self.class_species_map.update(override_map)
            self.class_mapping_overridden = True
            logger.info("Applied class mapping override from VESPAI_CLASS_MAP: %s", override_map)
        else:
            self.class_mapping_overridden = False

        if self.class_species_map:
            logger.info("Resolved hornet class mapping: %s", self.class_species_map)
        elif self.class_names:
            logger.warning(
                "No hornet classes resolved from model labels: %s. "
                "Set VESPAI_CLASS_MAP (e.g. '0:crabro,1:velutina') if needed.",
                self.class_names,
            )

        if self.class_names:
            labels_summary = ", ".join(
                f"{class_id}={label}" for class_id, label in sorted(self.class_names.items())
            )
            self.stats["detection_log"].appendleft({
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "species": "model-info",
                "confidence": "-",
                "frame_id": None,
                "model_label": f"Model labels: {labels_summary}",
                "class_id": -1,
                "velutina_count": 0,
                "crabro_count": 0,
            })
        
    def process_detections(self, 
                          results, 
                          frame: np.ndarray,
                          frame_id: int,
                          confidence_threshold: float = 0.8,
                          log_frame_prediction: bool = False) -> Tuple[int, int, np.ndarray]:
        """
        Process detection results and update statistics.
        
        Args:
            results: YOLOv5/YOLOv8 prediction results
            frame: Original image frame
            frame_id: Current frame ID
            confidence_threshold: Minimum confidence for valid detections
            log_frame_prediction: Log top class for the frame when no thresholded detections are present
            
        Returns:
            Tuple of (asian_hornets, european_hornets, annotated_frame)
        """
        velutina_count = 0  # Asian hornets
        crabro_count = 0    # European hornets
        bee_count = 0
        wasp_count = 0
        annotated_frame = frame.copy()
        detection_entries: List[Dict[str, Any]] = []
        
        # Parse predictions from YOLOv5 or YOLOv8
        predictions = self._extract_predictions(results)
        if predictions:
            
            total_confidence = 0
            confidence_count = 0
            
            for pred in predictions:
                x1, y1, x2, y2, conf, cls = pred
                cls = int(cls)
                confidence = float(conf)
                
                if confidence < confidence_threshold:
                    continue
                
                total_confidence += confidence
                confidence_count += 1

                model_label = self._get_model_label_for_class(cls)
                species = self._resolve_display_category_for_class(cls)
                if species == "velutina":
                    velutina_count += 1
                    color = (0, 0, 255)  # Red for Asian hornets
                    label = f"Velutina {confidence:.2f}"
                elif species == "crabro":
                    crabro_count += 1
                    color = (0, 255, 0)  # Green for European hornets  
                    label = f"Crabro {confidence:.2f}"
                elif species == "bee":
                    bee_count += 1
                    color = (255, 200, 0)
                    label = f"Bee {confidence:.2f}"
                elif species == "wasp":
                    wasp_count += 1
                    color = (255, 140, 0)
                    label = f"Wasp {confidence:.2f}"
                else:
                    color = (0, 165, 255)  # Orange for unmapped class
                    label = f"{model_label} {confidence:.2f}"
                    if cls not in self.unmapped_classes_seen:
                        self.unmapped_classes_seen.add(cls)
                        logger.warning(
                            "Unmapped model class detected: id=%d label='%s'. "
                            "Set VESPAI_CLASS_MAP to map this class if it is hornet-relevant.",
                            cls,
                            model_label,
                        )

                detection_entries.append({
                    "species": species or "other",
                    "model_label": model_label,
                    "confidence": confidence,
                    "class_id": cls,
                })
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Update statistics if detections found
            if detection_entries:
                self._update_detection_stats(bee_count, velutina_count, crabro_count, wasp_count,
                                           frame_id, total_confidence, confidence_count,
                                           detection_entries,
                                           annotated_frame)
            elif log_frame_prediction:
                self._append_frame_prediction_log(results, frame_id, annotated_frame)
        elif log_frame_prediction:
            self._append_frame_prediction_log(results, frame_id, annotated_frame)
        
        return velutina_count, crabro_count, annotated_frame

    def _append_frame_prediction_log(self, results: Any, frame_id: int, frame: np.ndarray):
        """Append a per-frame prediction log entry without changing detection counters."""
        if not isinstance(results, dict):
            return

        class_id_raw = results.get('top_class_id')
        confidence_raw = results.get('top_class_conf')
        if class_id_raw is None or confidence_raw is None:
            return

        try:
            class_id = int(class_id_raw)
            confidence = float(confidence_raw)
        except Exception:
            return

        species = self._resolve_display_category_for_class(class_id) or 'other'
        model_label = self._get_model_label_for_class(class_id)
        confidence_str = f"{(confidence * 100.0):.1f}"

        current_time = datetime.datetime.now()
        detection_key = f"{frame_id}_{current_time.strftime('%H%M%S')}"

        self._increment_category_totals(species, current_time)

        self.stats["detection_log"].append({
            "timestamp": current_time.strftime("%H:%M:%S"),
            "species": species,
            "confidence": confidence_str,
            "frame_id": detection_key,
            "model_label": model_label,
            "class_id": class_id,
            "bee_count": 1 if species == 'bee' else 0,
            "velutina_count": 1 if species == 'velutina' else 0,
            "crabro_count": 1 if species == 'crabro' else 0,
            "wasp_count": 1 if species == 'wasp' else 0,
        })

        self.stats["detection_frames"][detection_key] = frame.copy()
        if len(self.stats["detection_frames"]) > 20:
            oldest_key = min(self.stats["detection_frames"].keys())
            del self.stats["detection_frames"][oldest_key]

        self._update_last_detection_preview(frame, detection_key)

    def _get_model_label_for_class(self, class_id: int) -> str:
        """Return model label for class id when available."""
        if self._has_generic_class_placeholders():
            generic_alias = {
                0: 'Bee',
                1: 'Vespa Crabro',
                2: 'Vespa Velutina',
                3: 'Wasp',
            }
            if class_id in generic_alias:
                return generic_alias[class_id]
        if class_id in self.class_names:
            return str(self.class_names[class_id])
        return f"class{class_id}"

    def _normalize_class_names(self, class_names: Any) -> Dict[int, str]:
        """Normalize model class name structures to {id: label}."""
        if not class_names:
            return {}

        if isinstance(class_names, dict):
            normalized: Dict[int, str] = {}
            for key, value in class_names.items():
                try:
                    normalized[int(key)] = str(value)
                except Exception:
                    continue
            return normalized

        if isinstance(class_names, (list, tuple)):
            return {index: str(value) for index, value in enumerate(class_names)}

        return {}

    def _map_label_to_species(self, label: str) -> Optional[str]:
        """Map a model class label to canonical species key used by VespAI stats/UI."""
        text = str(label).strip().lower().replace('-', ' ').replace('_', ' ')

        velutina_markers = (
            'velutina',
            'asian hornet',
            'asiatic hornet',
            'vespa velutina',
        )
        crabro_markers = (
            'crabro',
            'european hornet',
            'vespa crabro',
        )

        if any(marker in text for marker in velutina_markers):
            return 'velutina'
        if any(marker in text for marker in crabro_markers):
            return 'crabro'

        return None

    def _parse_class_map_override(self, class_map_override: str = "") -> Dict[int, str]:
        """Parse optional class mapping override from VESPAI_CLASS_MAP."""
        import os

        raw_map = (class_map_override or '').strip() or os.environ.get('VESPAI_CLASS_MAP', '').strip()
        if not raw_map:
            return {}

        def normalize_species(value: str) -> Optional[str]:
            mapped = self._map_label_to_species(value)
            if mapped:
                return mapped

            lowered = str(value).strip().lower()
            if lowered in {'velutina', 'crabro'}:
                return lowered
            return None

        parsed: Dict[int, str] = {}

        # JSON format support: {"0":"crabro","1":"velutina"}
        if raw_map.startswith('{'):
            try:
                json_map = json.loads(raw_map)
                if isinstance(json_map, dict):
                    for key, value in json_map.items():
                        try:
                            class_id = int(key)
                        except Exception:
                            continue
                        species = normalize_species(str(value))
                        if species:
                            parsed[class_id] = species
                return parsed
            except Exception as error:
                logger.warning("Invalid JSON in VESPAI_CLASS_MAP: %s", error)
                return {}

        # CSV format support: 0:crabro,1:velutina
        for pair in raw_map.split(','):
            item = pair.strip()
            if not item or ':' not in item:
                continue
            class_id_raw, species_raw = item.split(':', 1)
            try:
                class_id = int(class_id_raw.strip())
            except Exception:
                continue

            species = normalize_species(species_raw.strip())
            if species:
                parsed[class_id] = species

        return parsed

    def _normalize_override_indices(self, override_map: Dict[int, str]) -> Dict[int, str]:
        """Normalize override indices, handling common 1-based label maps."""
        if not override_map or not self.class_names:
            return override_map

        known_ids = set(self.class_names.keys())
        override_ids = set(override_map.keys())

        if override_ids.issubset(known_ids):
            return override_map

        shifted = {class_id - 1: species for class_id, species in override_map.items() if (class_id - 1) >= 0}
        shifted_ids = set(shifted.keys())

        if shifted and shifted_ids.issubset(known_ids):
            logger.warning(
                "Detected 1-based class map override; shifted indices by -1 for runtime model classes: %s -> %s",
                override_map,
                shifted,
            )
            return shifted

        return override_map

    def _has_generic_class_placeholders(self) -> bool:
        """Return True if class names look like class0/class1 placeholders."""
        if not self.class_names:
            return False

        values = [str(value).strip().lower() for value in self.class_names.values()]
        if not values:
            return False

        return all(value.startswith('class') for value in values)

    def _resolve_species_for_class(self, class_id: int) -> Optional[str]:
        """Resolve predicted class id to species key, preferring class-name mapping."""
        if class_id in self.class_species_map:
            return self.class_species_map[class_id]

        if self.class_mapping_overridden:
            return None

        # Legacy fallback for older two-class models without usable names metadata.
        if not self.class_names or self._has_generic_class_placeholders():
            if class_id == 1:
                return 'velutina'
            if class_id == 0:
                return 'crabro'

        return None

    def _resolve_display_category_for_class(self, class_id: int) -> Optional[str]:
        """Resolve class id to dashboard/log category across all four user-facing classes."""
        hornet_species = self._resolve_species_for_class(class_id)
        if hornet_species:
            return hornet_species

        if self._has_generic_class_placeholders():
            if class_id == 0:
                return 'bee'
            if class_id == 3:
                return 'wasp'

        return None

    def _extract_predictions(self, results) -> List[Tuple[float, float, float, float, float, float]]:
        """Normalize YOLOv5/YOLOv8 outputs to a shared prediction tuple format."""
        normalized: List[Tuple[float, float, float, float, float, float]] = []

        # Custom ONNXRuntime backend format
        if isinstance(results, dict) and 'pred_tuples' in results:
            for row in results['pred_tuples']:
                x1, y1, x2, y2, conf, cls = row
                normalized.append((
                    float(x1), float(y1), float(x2), float(y2), float(conf), float(cls)
                ))
            return normalized

        # YOLOv5 format: results.pred[0] tensor, rows [x1, y1, x2, y2, conf, cls]
        if hasattr(results, 'pred') and results.pred is not None and len(results.pred) > 0:
            for row in results.pred[0]:
                x1, y1, x2, y2, conf, cls = row
                normalized.append((
                    float(x1), float(y1), float(x2), float(y2), float(conf), float(cls)
                ))
            return normalized

        # YOLOv8 format: list of Result objects with .boxes
        if isinstance(results, (list, tuple)) and len(results) > 0 and hasattr(results[0], 'boxes'):
            boxes = results[0].boxes
            if boxes is None:
                return normalized

            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, 'xyxy') else []
            conf = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else []
            cls = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else []

            count = min(len(xyxy), len(conf), len(cls))
            for index in range(count):
                x1, y1, x2, y2 = xyxy[index].tolist()
                normalized.append((
                    float(x1), float(y1), float(x2), float(y2), float(conf[index]), float(cls[index])
                ))

        return normalized
    
    def _update_detection_stats(self, 
                               bee: int,
                               velutina: int, 
                               crabro: int, 
                               wasp: int,
                               frame_id: int,
                               total_confidence: float,
                               confidence_count: int,
                               detection_entries: List[Dict[str, Any]],
                               frame: np.ndarray):
        """Update detection statistics and logs."""
        current_time = datetime.datetime.now()
        
        # Update global stats
        self._increment_category_totals('bee', current_time, bee)
        self._increment_category_totals('velutina', current_time, velutina)
        self._increment_category_totals('crabro', current_time, crabro)
        self._increment_category_totals('wasp', current_time, wasp)
        self.stats["frame_id"] = frame_id
        
        # Update hourly stats
        current_hour = current_time.hour
        if current_hour != self.current_hour:
            self.current_hour = current_hour
            
        self.hourly_detections[current_hour]["velutina"] += velutina
        self.hourly_detections[current_hour]["crabro"] += crabro
        
        # Update average confidence
        if confidence_count > 0:
            avg_confidence = (total_confidence / confidence_count) * 100
            self.stats["confidence_avg"] = avg_confidence
        
        # Create detection log entry
        top_detection = max(detection_entries, key=lambda entry: entry.get("confidence", 0.0))
        species = str(top_detection.get("species", "other"))
        top_confidence = float(top_detection.get("confidence", 0.0)) * 100.0
        confidence_str = f"{top_confidence:.1f}"
        detection_key = f"{frame_id}_{current_time.strftime('%H%M%S')}"
        
        log_entry = {
            "timestamp": current_time.strftime("%H:%M:%S"),
            "species": species,
            "confidence": confidence_str,
            "frame_id": detection_key,
            "model_label": str(top_detection.get("model_label", "unknown")),
            "class_id": int(top_detection.get("class_id", -1)),
            "bee_count": bee,
            "velutina_count": velutina,
            "crabro_count": crabro,
            "wasp_count": wasp,
        }
        
        self.stats["detection_log"].append(log_entry)
        
        # Store detection frame
        self.stats["detection_frames"][detection_key] = frame.copy()
        
        # Limit stored frames to prevent memory issues
        if len(self.stats["detection_frames"]) > 20:
            oldest_key = min(self.stats["detection_frames"].keys())
            del self.stats["detection_frames"][oldest_key]

        self._update_last_detection_preview(frame, detection_key)
        
        logger.info(
            "Detection frame %d: %d Velutina, %d Crabro, top label=%s (%.1f%%)",
            frame_id,
            velutina,
            crabro,
            log_entry["model_label"],
            top_confidence,
        )

    def _update_last_detection_preview(self, frame: np.ndarray, frame_id: str):
        """Create a lightweight inline preview for the most recent detection."""
        try:
            preview = frame
            height, width = preview.shape[:2]
            target_width = 320
            if width > target_width:
                target_height = max(1, int(height * (target_width / float(width))))
                preview = cv2.resize(preview, (target_width, target_height), interpolation=cv2.INTER_AREA)

            encoded, buffer = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 55])
            if not encoded:
                return

            self.stats["last_detection_preview"] = (
                'data:image/jpeg;base64,' + base64.b64encode(buffer.tobytes()).decode('ascii')
            )
            self.stats["last_detection_preview_frame_id"] = frame_id
        except Exception:
            pass

    def record_inference_timing(self, frame_id: int, source: str, duration_ms: float):
        """Record recent per-image inference durations for dashboard visualization."""
        duration_ms = round(float(duration_ms), 1)
        label = str(source or f"frame-{frame_id}")
        if ':' in label:
            label = label.split(':')[-1]
        if len(label) > 18:
            label = label[:15] + '...'

        self.stats["last_inference_ms"] = duration_ms
        self.stats["inference_count"] += 1
        self.stats["inference_total_ms"] += duration_ms
        self.stats["inference_avg_ms"] = round(
            self.stats["inference_total_ms"] / max(self.stats["inference_count"], 1),
            1,
        )
        if self.stats["inference_min_ms"] <= 0.0:
            self.stats["inference_min_ms"] = duration_ms
        else:
            self.stats["inference_min_ms"] = min(self.stats["inference_min_ms"], duration_ms)
        self.stats["inference_max_ms"] = max(self.stats["inference_max_ms"], duration_ms)
        self.stats["inference_timing_recent"].append({
            "frame_id": int(frame_id),
            "label": label,
            "duration_ms": duration_ms,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        })

    def _increment_category_totals(self, category: str, current_time: datetime.datetime, amount: int = 1):
        """Increment dashboard totals and timestamps for a display category."""
        if amount <= 0:
            return

        if category == 'bee':
            self.stats["total_bee"] += amount
            self.stats["last_bee_time"] = current_time.strftime("%H:%M:%S")
        elif category == 'velutina':
            self.stats["total_velutina"] += amount
            self.stats["last_velutina_time"] = current_time.strftime("%H:%M:%S")
        elif category == 'crabro':
            self.stats["total_crabro"] += amount
            self.stats["last_crabro_time"] = current_time.strftime("%H:%M:%S")
        elif category == 'wasp':
            self.stats["total_wasp"] += amount
            self.stats["last_wasp_time"] = current_time.strftime("%H:%M:%S")
        else:
            return

        self.stats["total_detections"] += amount
        self.stats["last_detection_time"] = current_time


def parse_resolution(resolution_str: str) -> Tuple[int, int]:
    """
    Parse resolution string into width and height.
    
    Args:
        resolution_str: Resolution string (e.g., "1920x1080", "1080p", "4k")
        
    Returns:
        Tuple of (width, height)
    """
    resolution_map = {
        "4k": (3840, 2160),
        "1080p": (1920, 1080), 
        "720p": (1280, 720)
    }
    
    if resolution_str in resolution_map:
        return resolution_map[resolution_str]
    
    try:
        width, height = map(int, resolution_str.split('x'))
        return width, height
    except:
        logger.warning("Invalid resolution format '%s', using default 1920x1080", resolution_str)
        return 1920, 1080