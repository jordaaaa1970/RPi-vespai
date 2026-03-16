#!/usr/bin/env python3
"""
VespAI Main Application

Modular main application that coordinates all VespAI components for hornet detection.
This replaces the monolithic web_preview.py with a clean, testable architecture.

Author: Jakob Zeise (Zeise Digital)
Version: 1.0
"""

import logging
import sys
import time
import threading
import signal
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, Tuple
from collections import deque

# Core modules
from .core.config import create_config_from_args
from .core.detection import CameraManager, ModelManager, DetectionProcessor
from .sms.lox24 import create_sms_manager_from_env
from .web.routes import register_routes

# External dependencies
try:
    from flask import Flask
    import cv2
    import torch
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install dependencies with: pip install -r requirements.txt")
    sys.exit(1)

class FriendlyLoggerNameFormatter(logging.Formatter):
    """Format logger names using more readable labels in log output."""

    LOGGER_NAME_MAP = {
        'werkzeug': 'web-server',
    }

    def format(self, record: logging.LogRecord) -> str:
        original_name = record.name
        record.name = self.LOGGER_NAME_MAP.get(record.name, record.name)
        try:
            return super().format(record)
        finally:
            record.name = original_name


# Set up logging
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "vespai.log"

log_formatter = FriendlyLoggerNameFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)

file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setFormatter(log_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[stream_handler, file_handler],
)
logger = logging.getLogger(__name__)


class VespAIApplication:
    """
    Main VespAI application that orchestrates all components.
    
    Provides a clean, modular architecture for hornet detection with
    camera management, model inference, web interface, and SMS alerts.
    """
    
    def __init__(self):
        """Initialize the VespAI application."""
        self.config = None
        self.camera_manager = None
        self.model_manager = None
        self.detection_processor = None
        self.sms_manager = None
        self.flask_app = None
        self.web_thread = None
        self.running = False
        self.source_lock = threading.Lock()
        self.current_input_mode = 'camera'
        self.current_dataset_path = ''
        self.dataset_executor = None
        self.dataset_prediction_queue = deque()
        
        # Global state for web interface
        self.web_frame = None
        self.web_lock = threading.Lock()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def initialize(self, args=None):
        """
        Initialize all application components.
        
        Args:
            args: Command line arguments (None for sys.argv)
        """
        logger.info("Initializing VespAI application...")

        # DEBUG: Print all environment variables to verify .env loading
        logger.info("ENVIRONMENT VARIABLES AT STARTUP:")
        for k, v in sorted(os.environ.items()):
            logger.info("%s=%s", k, v)
        
        # Load configuration
        self.config = create_config_from_args(args)
        # DEBUG: Print parsed config values for save_detections and enable_motion_detection
        logger.info("PARSED CONFIG: save_detections=%r, enable_motion_detection=%r", self.config.get('save_detections'), self.config.get('enable_motion_detection'))
        self.config.print_summary()
        
        # Initialize components
        self._initialize_camera()
        self._initialize_model()
        self._initialize_detection_processor()
        self._initialize_sms()
        
        if self.config.get('enable_web'):
            self._initialize_web_interface()
        
        logger.info("VespAI application initialized successfully")
    
    def _initialize_camera(self):
        """Initialize camera manager."""
        logger.info("Initializing camera...")
        resolution = self.config.get_camera_resolution()
        self.camera_manager = CameraManager(
            resolution,
            camera_source=self.config.get('camera_source', 'auto'),
        )
        
        video_file = self.config.get('video_file')
        self.camera_manager.initialize_camera(video_file)
        self.current_input_mode = 'dataset' if video_file else 'camera'
        self.current_dataset_path = video_file or self.config.get('dataset_path', '') or ''

    def get_input_source_state(self) -> Dict[str, Any]:
        """Return current runtime input source state for web API/UI."""
        return {
            'mode': self.current_input_mode,
            'dataset_path': self.current_dataset_path,
        }

    def switch_input_source(self, mode: str, dataset_path: str = '') -> Tuple[bool, str]:
        """Switch live input source between camera and dataset at runtime."""
        mode_normalized = (mode or '').strip().lower()
        if mode_normalized not in ('camera', 'dataset'):
            return False, "Invalid mode. Use 'camera' or 'dataset'."

        target_dataset_path = (dataset_path or '').strip()
        if mode_normalized == 'dataset' and not target_dataset_path:
            target_dataset_path = (
                self.current_dataset_path
                or (self.config.get('video_file') or '').strip()
                or (self.config.get('dataset_path') or '').strip()
                or (os.environ.get('VESPAI_DATASET_PATH') or '').strip()
            )

        if mode_normalized == 'dataset' and not target_dataset_path:
            return False, "Dataset path is required when mode is 'dataset'."

        resolution = self.config.get_camera_resolution()

        with self.source_lock:
            new_manager = CameraManager(
                resolution,
                camera_source=self.config.get('camera_source', 'auto'),
            )
            try:
                new_manager.initialize_camera(target_dataset_path if mode_normalized == 'dataset' else None)
            except Exception as error:
                return False, f"Failed to switch source: {error}"

            old_manager = self.camera_manager
            self.camera_manager = new_manager

            if mode_normalized == 'camera':
                self.current_input_mode = 'camera'
                self.config.set('video_file', None)
                self.detection_processor.stats['current_frame_source'] = ''
            else:
                self.current_input_mode = 'dataset'
                self.current_dataset_path = target_dataset_path
                self.config.set('video_file', target_dataset_path)
                self.config.set('dataset_path', target_dataset_path)
                self.detection_processor.stats['current_frame_source'] = ''

            with self.web_lock:
                self.web_frame = None

            if old_manager:
                try:
                    old_manager.release()
                except Exception:
                    pass

        logger.info("Input source switched to %s", self.current_input_mode)
        return True, "Input source updated"
    
    def _initialize_model(self):
        """Initialize model manager."""
        logger.info("Initializing detection model...")
        model_path = self.config.get('model_path')
        confidence = self.config.get('confidence_threshold')

        class_map = self.config.get('class_map', '')
        if class_map:
            os.environ['VESPAI_CLASS_MAP'] = class_map
        
        self.model_manager = ModelManager(model_path, confidence)
        self.model_manager.load_model()
    
    def _initialize_detection_processor(self):
        """Initialize detection processor."""
        logger.info("Initializing detection processor...")
        self.detection_processor = DetectionProcessor()
        if self.model_manager:
            self.detection_processor.set_class_names(
                self.model_manager.class_names,
                self.config.get('class_map', ''),
            )
    
    def _initialize_sms(self):
        """Initialize SMS manager."""
        if self.config.get('enable_sms'):
            logger.info("Initializing SMS alerts...")
            self.sms_manager = create_sms_manager_from_env()
            if self.sms_manager:
                logger.info("SMS alerts enabled")
            else:
                logger.warning("SMS configuration incomplete - alerts disabled")
        else:
            logger.info("SMS alerts disabled")
    
    def _initialize_web_interface(self):
        """Initialize Flask web interface."""
        logger.info("Initializing web interface...")
        
        # Configure Flask with template and static directories
        import os
        web_dir = os.path.join(os.path.dirname(__file__), 'web')
        template_dir = os.path.join(web_dir, 'templates')
        static_dir = os.path.join(web_dir, 'static')
        
        self.flask_app = Flask(__name__, 
                              template_folder=template_dir,
                              static_folder=static_dir,
                              static_url_path='/static')
        
        # Register web routes
        register_routes(
            self.flask_app,
            self.detection_processor.stats,
            self.detection_processor.hourly_detections,
            self
        )
        
        # Start web server in background thread (matching web_preview.py approach)
        web_config = self.config.get_web_config()
        self.web_thread = threading.Thread(
            target=self._run_web_server,
            args=(web_config['host'], web_config['port']),
            daemon=True  # Use daemon thread like original - auto-dies on main exit
        )
        self.web_thread.start()
        
        # Quick web server startup check
        time.sleep(0.5)
        logger.info("Web interface starting at %s", web_config['public_url'])
    
    def _run_web_server(self, host: str, port: int):
        """Run Flask web server (called in background thread) - simplified like web_preview.py."""
        try:
            # Match web_preview.py parameters exactly
            self.flask_app.run(host=host, port=port, threaded=True, debug=False)
        except Exception as e:
            logger.error("Web server error: %s", e)
    
    def run(self):
        """
        Run the main detection loop.
        
        This is the core application loop that processes camera frames,
        runs detection, handles alerts, and updates the web interface.
        """
        if not self._validate_initialization():
            logger.error("Application not properly initialized")
            return False
        
        logger.info("Starting VespAI detection system...")
        logger.info("Press Ctrl+C to stop")
        
        self.running = True
        frame_count = 0
        fps_start_time = time.time()
        fps_counter = 0
        
        # Add watchdog timer for system health
        last_frame_time = time.time()
        last_stats_update = time.time()
        
        try:
            while self.running:
                loop_start = time.time()
                self._drain_completed_dataset_predictions()
                
                # Watchdog: Detect if system is hanging
                current_time = time.time()
                if current_time - last_frame_time > 30:  # No frame for 30 seconds
                    logger.warning("System appears to be hanging - attempting recovery...")
                    self._attempt_recovery()
                    last_frame_time = current_time
                
                # Read frame from camera with timeout
                try:
                    with self.source_lock:
                        active_camera_manager = self.camera_manager
                        success, frame = active_camera_manager.read_frame()
                        source_exhausted = active_camera_manager.source_exhausted()
                        finite_source = active_camera_manager.is_finite_source()

                    if not success or frame is None:
                        if source_exhausted:
                            logger.info("Input dataset exhausted - switching back to live camera")
                            switched, message = self.switch_input_source('camera')
                            if not switched:
                                logger.error("Failed to switch back to live camera: %s", message)
                                self.running = False
                                break
                            time.sleep(0.2)
                            continue
                        logger.warning("Failed to read frame, retrying...")
                        time.sleep(0.1)
                        continue
                        
                    last_frame_time = current_time
                    
                except Exception as e:
                    logger.error(f"Camera error: {e}")
                    time.sleep(1)
                    continue
                
                frame_count += 1
                fps_counter += 1

                with self.source_lock:
                    current_source = self.camera_manager.get_last_frame_source()
                self.detection_processor.stats['current_frame_source'] = current_source
                
                # Update frame count in stats (for web dashboard)
                self.detection_processor.stats['frame_id'] = frame_count
                
                # Debug logging every 30 frames
                if frame_count % 30 == 0:
                    logger.debug(f"Frame count updated: {frame_count}")
                
                # Update FPS calculation
                if time.time() - fps_start_time >= 1.0:
                    self.detection_processor.stats['fps'] = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()

                if finite_source and self.config.get('enable_web'):
                    try:
                        display_frame = cv2.resize(frame, (480, 270))
                        with self.web_lock:
                            self.web_frame = display_frame.copy()
                    except Exception as e:
                        logger.error(f"Web frame update error: {e}")
                
                if finite_source:
                    dataset_delay = self.config.get('dataset_frame_delay', 0.6)
                    if dataset_delay >= 4.0:
                        velutina_count, crabro_count, annotated_frame = self._run_detection_step(frame, frame_count, finite_source, current_source)

                        if velutina_count > 0 or crabro_count > 0:
                            self._handle_detection(velutina_count, crabro_count, frame_count, annotated_frame)

                        if self.config.get('enable_web'):
                            try:
                                display_frame = cv2.resize(annotated_frame, (480, 270))
                                with self.web_lock:
                                    self.web_frame = display_frame.copy()
                            except Exception as e:
                                logger.error(f"Web frame update error: {e}")
                    else:
                        self._submit_dataset_prediction(frame_count, frame)
                        if len(self.dataset_prediction_queue) > 2:
                            self._drain_completed_dataset_predictions(wait_for_one=True)
                else:
                    velutina_count, crabro_count, annotated_frame = self._run_detection_step(frame, frame_count, finite_source, current_source)

                    # Handle detections
                    if velutina_count > 0 or crabro_count > 0:
                        self._handle_detection(velutina_count, crabro_count, frame_count, annotated_frame)

                    # Update web frame (optimized for Raspberry Pi) with error handling
                    if self.config.get('enable_web'):
                        try:
                            # Reduce resolution for better Raspberry Pi performance
                            display_frame = cv2.resize(annotated_frame, (480, 270))
                            with self.web_lock:
                                self.web_frame = display_frame.copy()
                        except Exception as e:
                            logger.error(f"Web frame update error: {e}")
                
                # Force stats update every 10 seconds to keep web interface alive
                if current_time - last_stats_update > 10:
                    self.detection_processor.stats['last_update'] = current_time
                    last_stats_update = current_time
                
                # Print detection info if enabled
                if self.config.get('print_detections') and (velutina_count > 0 or crabro_count > 0):
                    confidence = self.detection_processor.stats.get('confidence_avg', 0)
                    print(f"Frame {frame_count}: {velutina_count} Velutina, {crabro_count} Crabro "
                          f"(confidence: {confidence:.1f}%)")

                # Print periodic frame progress even when there are no detections.
                if self.config.get('print_detections') and velutina_count == 0 and crabro_count == 0:
                    with self.source_lock:
                        source = self.camera_manager.get_last_frame_source()
                        should_print = self.camera_manager.is_finite_source() or (frame_count % 30 == 0)
                    if should_print:
                        print(f"Frame {frame_count}: processed (no detections) | source: {source}")
                
                # Frame rate control (optimized for Raspberry Pi)
                frame_delay = self.config.get('frame_delay', 0.3)
                with self.source_lock:
                    if self.camera_manager and self.camera_manager.is_finite_source():
                        frame_delay = max(frame_delay, self.config.get('dataset_frame_delay', 0.6))
                elapsed = time.time() - loop_start
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
            self.running = False
        except Exception as e:
            logger.error("Unexpected error in detection loop: %s", e)
            self.running = False
            return False
        finally:
            self._drain_completed_dataset_predictions(wait_for_all=True)
            self._shutdown_dataset_executor()
            # Simple cleanup like web_preview.py
            self._cleanup()
        
        logger.info("VespAI detection system stopped")
        return True

    def _run_detection_step(self, frame, frame_count: int, finite_source: bool, source_label: str = ""):
        """Run one detection step and return counts plus annotated frame."""
        try:
            predict_started = time.time()
            results = self.model_manager.predict(frame)
            inference_ms = (time.time() - predict_started) * 1000.0
            self.detection_processor.record_inference_timing(frame_count, source_label, inference_ms)
            if isinstance(results, dict):
                self.detection_processor.stats['model_debug_summary'] = results.get('debug_summary', '')
            else:
                self.detection_processor.stats['model_debug_summary'] = ''
            return self.detection_processor.process_detections(
                results,
                frame,
                frame_count,
                self.config.get('confidence_threshold'),
                log_frame_prediction=finite_source,
            )
        except Exception as e:
            logger.error(f"Detection error: {e}")
            self.detection_processor.stats['model_debug_summary'] = ''
            return 0, 0, frame.copy()

    def _ensure_dataset_executor(self):
        if self.dataset_executor is None:
            self.dataset_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='vespai-dataset')
        return self.dataset_executor

    def _submit_dataset_prediction(self, frame_count: int, frame):
        executor = self._ensure_dataset_executor()
        with self.source_lock:
            source_label = self.camera_manager.get_last_frame_source() if self.camera_manager else f"frame-{frame_count}"
        future = executor.submit(self._predict_with_timing, frame.copy())
        self.dataset_prediction_queue.append((frame_count, frame.copy(), source_label, future))

    def _predict_with_timing(self, frame):
        predict_started = time.time()
        results = self.model_manager.predict(frame)
        inference_ms = (time.time() - predict_started) * 1000.0
        return results, inference_ms

    def _drain_completed_dataset_predictions(self, wait_for_one: bool = False, wait_for_all: bool = False):
        while self.dataset_prediction_queue:
            frame_count, frame, source_label, future = self.dataset_prediction_queue[0]
            if not wait_for_all and not future.done():
                if wait_for_one:
                    try:
                        future.result()
                    except Exception:
                        pass
                else:
                    break

            self.dataset_prediction_queue.popleft()
            try:
                results, inference_ms = future.result()
            except Exception as error:
                logger.error("Detection error: %s", error)
                self.detection_processor.stats['model_debug_summary'] = ''
                continue

            self.detection_processor.record_inference_timing(frame_count, source_label, inference_ms)

            velutina_count, crabro_count, annotated_frame = self.detection_processor.process_detections(
                results,
                frame,
                frame_count,
                self.config.get('confidence_threshold'),
                log_frame_prediction=True,
            )
            if isinstance(results, dict):
                self.detection_processor.stats['model_debug_summary'] = results.get('debug_summary', '')
            else:
                self.detection_processor.stats['model_debug_summary'] = ''

            if velutina_count > 0 or crabro_count > 0:
                self._handle_detection(velutina_count, crabro_count, frame_count, annotated_frame)

            if not wait_for_all and not wait_for_one:
                continue

    def _shutdown_dataset_executor(self):
        if self.dataset_executor is not None:
            self.dataset_executor.shutdown(wait=False, cancel_futures=False)
            self.dataset_executor = None
        self.dataset_prediction_queue.clear()
    
    def _handle_detection(self, velutina_count: int, crabro_count: int, frame_id: int, frame):
        """
        Handle a detection event with alerts and logging.
        
        Args:
            velutina_count: Number of Asian hornets detected
            crabro_count: Number of European hornets detected  
            frame_id: Current frame ID
            frame: Detection frame with annotations
        """
        # Save detection image if enabled
        if self.config.get('save_detections'):
            self._save_detection_image(frame, frame_id, velutina_count, crabro_count)
        
        # Send SMS alert if configured
        if self.sms_manager:
            self._send_sms_alert(velutina_count, crabro_count, frame_id)
    
    def _save_detection_image(self, frame, frame_id: int, velutina: int, crabro: int):
        """Save detection image to disk."""
        import os
        from datetime import datetime
        
        save_dir = self.config.get('save_directory', 'data/detections')
        os.makedirs(save_dir, exist_ok=True)
        self._prune_saved_detection_images(save_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        species = 'velutina' if velutina > 0 else 'crabro'
        filename = f"{timestamp}_frame{frame_id}_{species}_{velutina}v_{crabro}c.jpg"
        filepath = os.path.join(save_dir, filename)
        
        cv2.imwrite(filepath, frame)
        logger.info("Saved detection image: %s", filepath)

    def _prune_saved_detection_images(self, save_dir: str):
        """Delete saved detection images older than the configured retention period."""
        retention_days = int(self.config.get('detection_retention_days', 21))
        if retention_days <= 0:
            return

        cutoff = time.time() - (retention_days * 24 * 60 * 60)
        deleted_count = 0

        try:
            with os.scandir(save_dir) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                    try:
                        if entry.stat().st_mtime < cutoff:
                            os.remove(entry.path)
                            deleted_count += 1
                    except FileNotFoundError:
                        continue
                    except Exception as error:
                        logger.warning("Failed to prune old detection file %s: %s", entry.path, error)
        except FileNotFoundError:
            return
        except Exception as error:
            logger.warning("Failed to scan detection save directory %s: %s", save_dir, error)
            return

        if deleted_count:
            logger.info(
                "Pruned %d detection image(s) older than %d day(s) from %s",
                deleted_count,
                retention_days,
                save_dir,
            )
    
    def _send_sms_alert(self, velutina_count: int, crabro_count: int, frame_id: int):
        """Send SMS alert for detection."""
        if not self.sms_manager:
            return
        
        # Create frame URL for SMS
        web_config = self.config.get_web_config()
        current_time = time.strftime('%H%M%S')
        detection_key = f"{frame_id}_{current_time}"
        frame_url = f"{web_config['public_url']}/frame/{detection_key}"
        
        # Determine hornet type and create alert
        if velutina_count > 0:
            hornet_type = 'velutina'
            count = velutina_count
        else:
            hornet_type = 'crabro'
            count = crabro_count
        
        confidence = self.detection_processor.stats.get('confidence_avg', 0)
        message = self.sms_manager.create_hornet_alert(hornet_type, count, confidence, frame_url)
        
        # Send alert
        success, status = self.sms_manager.send_alert(message)
        if success:
            logger.info("SMS alert sent: %s", status)
        else:
            logger.warning("SMS alert failed: %s", status)
    
    def _validate_initialization(self) -> bool:
        """Validate that all required components are initialized."""
        if not self.camera_manager:
            logger.error("Camera manager not initialized")
            return False
        
        if not self.model_manager or not self.model_manager.model:
            logger.error("Model manager not initialized") 
            return False
        
        if not self.detection_processor:
            logger.error("Detection processor not initialized")
            return False
        
        return True
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Received signal %d, shutting down...", signum)
        self.running = False
        
        # Force immediate shutdown on second Ctrl+C
        if hasattr(self, '_shutdown_requested'):
            logger.info("Force shutdown requested, terminating immediately...")
            import os
            os._exit(0)
        self._shutdown_requested = True
    
    def _attempt_recovery(self):
        """Attempt to recover from system hang."""
        logger.info("Attempting system recovery...")
        
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Reset camera connection
            if self.camera_manager:
                logger.info("Resetting camera connection...")
                with self.source_lock:
                    self.camera_manager.release()
                    time.sleep(2)
                    self.camera_manager.initialize_camera(self.current_dataset_path if self.current_input_mode == 'dataset' else None)
            
            # Clear any stuck web frames
            with self.web_lock:
                self.web_frame = None
                
            logger.info("Recovery attempt completed")
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
    
    def _cleanup(self):
        """Clean up resources on shutdown - simplified like web_preview.py."""
        logger.info("Cleaning up resources...")
        
        # Release camera (web server will auto-die as daemon thread)
        if self.camera_manager:
            with self.source_lock:
                self.camera_manager.release()
        
        # Final statistics
        if self.detection_processor:
            stats = self.detection_processor.stats
            logger.info("Final statistics:")
            logger.info("  Total frames processed: %d", stats.get('frame_id', 0))
            logger.info("  Total detections: %d", stats.get('total_detections', 0))
            logger.info("  Asian hornets: %d", stats.get('total_velutina', 0))
            logger.info("  European hornets: %d", stats.get('total_crabro', 0))


def main():
    """Main entry point for the VespAI application."""
    app = VespAIApplication()
    
    try:
        app.initialize()
        success = app.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error("Fatal error: %s", e)
        sys.exit(1)


if __name__ == '__main__':
    main()