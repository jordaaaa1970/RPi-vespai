#!/usr/bin/env python3
"""
Integration tests for the modular VespAI application

Tests the complete application flow including initialization,
detection processing, and component integration.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import threading
import time

# Add src to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from src.vespai.main import VespAIApplication


class TestVespAIApplication(unittest.TestCase):
    """Integration tests for VespAI application"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = VespAIApplication()
    
    @patch('src.vespai.main.create_config_from_args')
    @patch('src.vespai.main.CameraManager')
    @patch('src.vespai.main.ModelManager') 
    @patch('src.vespai.main.DetectionProcessor')
    def test_initialization(self, mock_detection, mock_model, mock_camera, mock_config):
        """Test application initialization"""
        # Mock configuration
        mock_config_obj = Mock()
        mock_config_obj.get.side_effect = lambda key, default=None: {
            'enable_web': False,
            'enable_sms': False,
            'video_file': None,
            'model_path': 'test_model.pt',
            'confidence_threshold': 0.8
        }.get(key, default)
        mock_config_obj.get_camera_resolution.return_value = (1920, 1080)
        mock_config_obj.print_summary = Mock()
        mock_config.return_value = mock_config_obj
        
        # Mock components
        mock_camera_instance = Mock()
        mock_camera.return_value = mock_camera_instance
        
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        mock_detection_instance = Mock()
        mock_detection.return_value = mock_detection_instance
        
        # Initialize application
        self.app.initialize()
        
        # Verify initialization
        self.assertIsNotNone(self.app.config)
        self.assertIsNotNone(self.app.camera_manager)
        self.assertIsNotNone(self.app.model_manager)
        self.assertIsNotNone(self.app.detection_processor)
        
        # Verify component calls
        mock_camera_instance.initialize_camera.assert_called_once_with(None)
        mock_model_instance.load_model.assert_called_once()
    
    @patch('src.vespai.main.create_config_from_args')
    @patch('src.vespai.main.Flask')
    @patch('src.vespai.main.register_routes')
    def test_web_interface_initialization(self, mock_register, mock_flask, mock_config):
        """Test web interface initialization"""
        # Mock configuration with web enabled
        mock_config_obj = Mock()
        mock_config_obj.get.side_effect = lambda key, default=None: {
            'enable_web': True,
            'enable_sms': False,
            'video_file': None,
            'model_path': 'test_model.pt',
            'confidence_threshold': 0.8
        }.get(key, default)
        mock_config_obj.get_camera_resolution.return_value = (1920, 1080)
        mock_config_obj.get_web_config.return_value = {
            'enabled': True,
            'host': '0.0.0.0',
            'port': 5000,
            'public_url': 'http://localhost:5000'
        }
        mock_config_obj.print_summary = Mock()
        mock_config.return_value = mock_config_obj
        
        # Mock Flask app
        mock_app = Mock()
        mock_flask.return_value = mock_app
        
        # Mock other components to avoid initialization errors
        with patch('src.vespai.main.CameraManager'), \
             patch('src.vespai.main.ModelManager'), \
             patch('src.vespai.main.DetectionProcessor'):
            
            self.app.initialize()
        
        # Verify web interface setup
        self.assertIsNotNone(self.app.flask_app)
        mock_register.assert_called_once()
        self.assertIsNotNone(self.app.web_thread)
    
    def test_signal_handler(self):
        """Test signal handling for graceful shutdown"""
        self.app.running = True
        
        # Simulate signal
        self.app._signal_handler(2, None)  # SIGINT
        
        self.assertFalse(self.app.running)
    
    @patch('src.vespai.main.create_config_from_args')
    def test_cleanup(self, mock_config):
        """Test application cleanup"""
        # Mock configuration
        mock_config_obj = Mock()
        mock_config_obj.print_summary = Mock()
        mock_config.return_value = mock_config_obj
        
        # Mock camera manager
        mock_camera = Mock()
        self.app.camera_manager = mock_camera
        
        # Mock detection processor
        mock_processor = Mock()
        mock_processor.stats = {
            'frame_id': 100,
            'total_detections': 10,
            'total_velutina': 3,
            'total_crabro': 7
        }
        self.app.detection_processor = mock_processor
        
        # Call cleanup
        self.app._cleanup()
        
        # Verify camera release was called
        mock_camera.release.assert_called_once()
    
    def test_validation_missing_components(self):
        """Test validation with missing components"""
        # Test with no components initialized
        self.assertFalse(self.app._validate_initialization())
        
        # Test with camera but no model
        self.app.camera_manager = Mock()
        self.assertFalse(self.app._validate_initialization())
        
        # Test with camera and model but no detection processor
        self.app.model_manager = Mock()
        self.app.model_manager.model = Mock()
        self.assertFalse(self.app._validate_initialization())
        
        # Test with all components
        self.app.detection_processor = Mock()
        self.assertTrue(self.app._validate_initialization())
    
    @patch('cv2.imwrite')
    @patch('os.makedirs')
    def test_save_detection_image(self, mock_makedirs, mock_imwrite):
        """Test saving detection images"""
        # Mock configuration
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: {
            'save_directory': 'test_detections',
            'detection_retention_days': 21,
        }.get(key, default)
        self.app.config = mock_config
        
        # Create test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Call save method
        self.app._save_detection_image(test_frame, 123, 1, 0)
        
        # Verify directory creation and file write
        mock_makedirs.assert_called_once_with('test_detections', exist_ok=True)
        mock_imwrite.assert_called_once()
    
    @patch('src.vespai.main.create_sms_manager_from_env')
    def test_sms_alert_handling(self, mock_create_sms):
        """Test SMS alert handling"""
        # Mock SMS manager
        mock_sms = Mock()
        mock_sms.create_hornet_alert.return_value = "Test alert message"
        mock_sms.send_alert.return_value = (True, "SMS sent successfully")
        mock_create_sms.return_value = mock_sms
        
        # Mock configuration
        mock_config = Mock()
        mock_config.get_web_config.return_value = {'public_url': 'http://localhost:5000'}
        
        # Set up app state
        self.app.config = mock_config
        self.app.sms_manager = mock_sms
        self.app.detection_processor = Mock()
        self.app.detection_processor.stats = {'confidence_avg': 95.5}
        
        # Call SMS alert method
        self.app._send_sms_alert(1, 0, 456)
        
        # Verify SMS creation and sending
        mock_sms.create_hornet_alert.assert_called_once()
        mock_sms.send_alert.assert_called_once_with("Test alert message")


class TestApplicationFlow(unittest.TestCase):
    """Test complete application flow scenarios"""
    
    @unittest.skip("Complex mock dependencies - functional tests cover this workflow")
    @patch('src.vespai.main.create_config_from_args')
    @patch('src.vespai.main.CameraManager')
    @patch('src.vespai.main.ModelManager')
    @patch('src.vespai.main.DetectionProcessor')
    @patch('src.vespai.main.create_sms_manager_from_env')
    def test_detection_workflow(self, mock_sms, mock_detection, mock_model, mock_camera, mock_config):
        """Test complete detection workflow"""
        # Setup mocks
        app = VespAIApplication()
        
        # Mock configuration
        mock_config_obj = Mock()
        mock_config_obj.get.side_effect = lambda key, default=None: {
            'enable_web': False,
            'enable_sms': True,
            'confidence_threshold': 0.8,
            'frame_delay': 0.01,  # Fast for testing
            'print_detections': False,
            'save_detections': False
        }.get(key, default)
        mock_config_obj.get_camera_resolution.return_value = (640, 480)
        mock_config_obj.print_summary = Mock()
        mock_config.return_value = mock_config_obj
        
        # Mock camera - return a few frames then stop
        mock_camera_instance = Mock()
        frame_data = [
            (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)),
            (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)),
            (False, None)  # Trigger stop
        ]
        mock_camera_instance.read_frame.side_effect = frame_data
        mock_camera.return_value = mock_camera_instance
        
        # Mock model
        mock_model_instance = Mock()
        mock_results = Mock()
        mock_results.pred = [[]]  # Empty predictions for successful processing
        mock_model_instance.predict.return_value = mock_results
        mock_model.return_value = mock_model_instance
        
        # Mock detection processor
        mock_detection_instance = Mock()
        mock_detection_instance.stats = {'fps': 0, 'total_detections': 0}
        mock_detection_instance.process_detections.return_value = (1, 0, np.zeros((480, 640, 3)))
        mock_detection.return_value = mock_detection_instance
        
        # Mock SMS manager
        mock_sms_instance = Mock()
        mock_sms.return_value = mock_sms_instance
        
        # Initialize and run (should stop after processing frames)
        app.initialize()
        
        # Override running state to stop after short time
        def stop_after_delay():
            time.sleep(0.1)
            app.running = False
        
        stop_thread = threading.Thread(target=stop_after_delay)
        stop_thread.start()
        
        # Run application
        result = app.run()
        stop_thread.join()
        
        # Verify workflow executed
        self.assertTrue(result)
        mock_camera_instance.read_frame.assert_called()
        mock_model_instance.predict.assert_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)