#!/usr/bin/env python3
"""Tests for the current Flask web routes."""

import os
import sys
import threading
import unittest
from collections import deque
from datetime import datetime
from unittest.mock import patch

import numpy as np
from flask import Flask

# Add project root to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from src.vespai.web.routes import register_routes


class DummyAppInstance:
    """Minimal application stub for route registration tests."""

    def __init__(self, frame):
        self.web_frame = frame
        self.web_lock = threading.Lock()
        self._mode = 'camera'
        self._dataset_path = ''

    def switch_input_source(self, mode, dataset_path=''):
        if mode not in {'camera', 'dataset'}:
            return False, 'Invalid mode'
        self._mode = mode
        self._dataset_path = dataset_path if mode == 'dataset' else ''
        return True, 'Input source updated'

    def get_input_source_state(self):
        return {
            'mode': self._mode,
            'dataset_path': self._dataset_path,
        }


class TestWebRoutes(unittest.TestCase):
    """Route tests aligned with the current dashboard implementation."""

    def setUp(self):
        web_dir = os.path.join(project_root, 'src', 'vespai', 'web')
        self.app = Flask(
            __name__,
            template_folder=os.path.join(web_dir, 'templates'),
            static_folder=os.path.join(web_dir, 'static'),
            static_url_path='/static',
        )
        self.app.config['TESTING'] = True

        self.stats = {
            'frame_id': 42,
            'total_bee': 1,
            'total_velutina': 3,
            'total_crabro': 7,
            'total_wasp': 2,
            'total_detections': 13,
            'fps': 15.7,
            'last_detection_time': datetime(2026, 3, 7, 12, 30, 45),
            'start_time': datetime(2026, 3, 7, 12, 0, 0),
            'detection_log': deque(maxlen=20),
            'hourly_stats': deque(maxlen=24),
            'detection_frames': {},
            'cpu_temp': 0,
            'cpu_usage': 0,
            'ram_usage': 0,
            'disk_usage': 0,
            'uptime': 0,
            'saved_images': 5,
            'sms_sent': 2,
            'sms_cost': 0.24,
            'confidence_avg': 94.1,
            'inference_timing_recent': deque(maxlen=20),
            'last_sms_time': datetime(2026, 3, 7, 12, 31, 0),
        }
        self.hourly_detections = {hour: {'velutina': 0, 'crabro': 0} for hour in range(24)}
        self.hourly_detections[12]['velutina'] = 3
        self.hourly_detections[12]['crabro'] = 7

        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        self.app_instance = DummyAppInstance(frame)
        register_routes(self.app, self.stats, self.hourly_detections, self.app_instance)
        self.client = self.app.test_client()

    def test_index_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'VespAI Monitor', response.data)
        self.assertIn(b'Live Detection Feed', response.data)
        self.assertIn(b'FR', response.data)

    def test_video_feed_route(self):
        response = self.client.get('/video_feed')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'multipart/x-mixed-replace; boundary=frame')

    def test_current_frame_route(self):
        response = self.client.get('/api/current_frame')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'image/jpeg')

    def test_stats_route(self):
        with patch('psutil.cpu_percent', return_value=25.5), \
             patch('psutil.virtual_memory') as mock_vm, \
             patch('psutil.disk_usage') as mock_disk:
            mock_vm.return_value.percent = 45.2
            mock_disk.return_value.percent = 60.1

            response = self.client.get('/api/stats')

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['total_velutina'], 3)
        self.assertEqual(data['total_crabro'], 7)
        self.assertEqual(data['input_mode'], 'camera')
        self.assertEqual(len(data['hourly_data']), 24)
        self.assertNotIn('detection_frames', data)
        self.assertIn('system_health', data)

    def test_detection_frame_and_page(self):
        frame_id = '42_123045'
        self.stats['detection_frames'][frame_id] = np.zeros((100, 100, 3), dtype=np.uint8)
        self.stats['detection_log'].append({
            'timestamp': '12:30:45',
            'species': 'velutina',
            'confidence': '96.8',
            'frame_id': frame_id,
            'model_label': 'Vespa Velutina',
            'class_id': 2,
        })

        image_response = self.client.get(f'/api/detection_frame/{frame_id}')
        self.assertEqual(image_response.status_code, 200)
        self.assertEqual(image_response.content_type, 'image/jpeg')

        page_response = self.client.get(f'/frame/{frame_id}')
        self.assertEqual(page_response.status_code, 200)
        self.assertIn(b'Detection Frame: 42_123045', page_response.data)
        self.assertIn(b'Vespa Velutina', page_response.data)

    def test_detection_frame_not_found(self):
        response = self.client.get('/api/detection_frame/missing')
        self.assertEqual(response.status_code, 404)

        page_response = self.client.get('/frame/missing')
        self.assertEqual(page_response.status_code, 404)
        self.assertIn(b'Frame missing not found', page_response.data)

    def test_frames_list(self):
        self.stats['detection_frames']['frame1'] = np.zeros((10, 10, 3), dtype=np.uint8)
        self.stats['detection_frames']['frame2'] = np.zeros((10, 10, 3), dtype=np.uint8)

        response = self.client.get('/api/frames')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['total_frames'], 2)
        self.assertIn('frame1', data['available_frames'])
        self.assertIn('frame2', data['available_frames'])

    def test_input_source_update(self):
        response = self.client.post(
            '/api/input_source',
            json={
                'mode': 'dataset',
                'dataset_path': '/tmp/dataset.tfrecord',
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['mode'], 'dataset')
        self.assertEqual(data['dataset_path'], '/tmp/dataset.tfrecord')


if __name__ == '__main__':
    unittest.main(verbosity=2)
