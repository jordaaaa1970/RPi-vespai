#!/usr/bin/env python3
"""
Test suite for VespAI SMS module

Comprehensive tests for the Lox24 SMS client and SMS manager functionality.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json

# Add src to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from src.vespai.sms.lox24 import Lox24SMS, SMSManager, create_sms_manager_from_env


class TestLox24SMS(unittest.TestCase):
    """Test cases for Lox24SMS class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test_api_key"
        self.sender_name = "TestSender"
        self.sms_client = Lox24SMS(self.api_key, self.sender_name)
    
    def test_init_with_simple_api_key(self):
        """Test initialization with simple API key"""
        client = Lox24SMS("simple_key", "TestSender")
        self.assertEqual(client.username, "")
        self.assertEqual(client.password, "simple_key")
        self.assertEqual(client.sender_name, "TestSender")
        self.assertTrue(client.sms_available)
    
    def test_init_with_username_password_format(self):
        """Test initialization with username:password format"""
        client = Lox24SMS("user:pass", "TestSender")
        self.assertEqual(client.username, "user")
        self.assertEqual(client.password, "pass")
    
    @patch('src.vespai.sms.lox24.requests.post')
    def test_send_sms_success(self, mock_post):
        """Test successful SMS sending"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"price": 0.05, "id": "12345"}
        mock_post.return_value = mock_response
        
        success, cost = self.sms_client.send_sms("+491234567890", "Test message")
        
        self.assertTrue(success)
        self.assertEqual(cost, 0.05)
        mock_post.assert_called_once()
    
    @patch('src.vespai.sms.lox24.requests.post')
    def test_send_sms_failure(self, mock_post):
        """Test SMS sending failure"""
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response
        
        success, cost = self.sms_client.send_sms("+491234567890", "Test message")
        
        self.assertFalse(success)
        self.assertEqual(cost, 0.0)
    
    @patch('src.vespai.sms.lox24.requests.post')
    def test_send_sms_network_error(self, mock_post):
        """Test SMS sending with network error"""
        # Mock network exception
        import requests
        mock_post.side_effect = requests.exceptions.RequestException("Network error")
        
        success, cost = self.sms_client.send_sms("+491234567890", "Test message")
        
        self.assertFalse(success)
        self.assertEqual(cost, 0.0)
    
    def test_disabled_sms_client(self):
        """Test SMS client with SMS disabled"""
        client = Lox24SMS("api_key", "TestSender")
        client.sms_available = False
        
        success, cost = client.send_sms("+491234567890", "Test message")
        
        self.assertFalse(success)
        self.assertEqual(cost, 0.0)
    
    def test_extract_cost_from_response(self):
        """Test cost extraction from different response formats"""
        # Test with 'price' field
        response1 = {"price": 0.08}
        cost1 = self.sms_client._extract_cost_from_response(response1)
        self.assertEqual(cost1, 0.08)
        
        # Test with 'cost' field
        response2 = {"cost": "0.12"}
        cost2 = self.sms_client._extract_cost_from_response(response2)
        self.assertEqual(cost2, 0.12)
        
        # Test with no cost field
        response3 = {"id": "12345"}
        cost3 = self.sms_client._extract_cost_from_response(response3)
        self.assertEqual(cost3, 0.0)
    
    def test_handle_error_response(self):
        """Test error response handling"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"
        
        error_msg = self.sms_client._handle_error_response(mock_response)
        
        self.assertIn("Client ID or API key", error_msg)
        self.assertIn("Invalid API key", error_msg)


class TestSMSManager(unittest.TestCase):
    """Test cases for SMSManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test_key"
        self.phone_number = "+491234567890"
        self.manager = SMSManager(self.api_key, self.phone_number)
    
    @patch('src.vespai.sms.lox24.Lox24SMS')
    def test_init_with_valid_config(self, mock_lox24):
        """Test SMS manager initialization with valid configuration"""
        mock_client = Mock()
        mock_lox24.return_value = mock_client
        
        manager = SMSManager("api_key", "+491234567890", delay_minutes=10)
        
        self.assertEqual(manager.phone_number, "+491234567890")
        self.assertEqual(manager.delay_minutes, 10)
        self.assertTrue(manager.enabled)
        self.assertIsNotNone(manager.client)
    
    def test_init_disabled(self):
        """Test SMS manager initialization when disabled"""
        manager = SMSManager("", "", enabled=False)
        
        self.assertFalse(manager.enabled)
        self.assertIsNone(manager.client)
    
    def test_send_alert_disabled(self):
        """Test sending alert when SMS is disabled"""
        manager = SMSManager("", "", enabled=False)
        
        success, message = manager.send_alert("Test alert")
        
        self.assertFalse(success)
        self.assertIn("disabled", message)
    
    def test_send_alert_no_client(self):
        """Test sending alert with no SMS client"""
        manager = SMSManager("", "")
        manager.client = None
        
        success, message = manager.send_alert("Test alert")
        
        self.assertFalse(success)
        self.assertIn("not initialized", message)
    
    @patch('src.vespai.sms.lox24.Lox24SMS')
    def test_send_alert_success(self, mock_lox24):
        """Test successful alert sending"""
        mock_client = Mock()
        mock_client.send_sms.return_value = (True, 0.05)
        mock_lox24.return_value = mock_client
        
        manager = SMSManager("api_key", "+491234567890")
        success, message = manager.send_alert("Test alert")
        
        self.assertTrue(success)
        self.assertIn("successfully", message)
        self.assertIn("0.05", message)
    
    @patch('src.vespai.sms.lox24.Lox24SMS')
    def test_send_alert_rate_limited(self, mock_lox24):
        """Test alert sending with rate limiting"""
        mock_client = Mock()
        mock_lox24.return_value = mock_client
        
        manager = SMSManager("api_key", "+491234567890", delay_minutes=5)
        manager.last_sms_time = datetime.now() - timedelta(minutes=2)
        
        success, message = manager.send_alert("Test alert")
        
        self.assertFalse(success)
        self.assertIn("Rate limited", message)
    
    @patch('src.vespai.sms.lox24.Lox24SMS')
    def test_send_alert_force_override_rate_limit(self, mock_lox24):
        """Test forcing alert to bypass rate limiting"""
        mock_client = Mock()
        mock_client.send_sms.return_value = (True, 0.03)
        mock_lox24.return_value = mock_client
        
        manager = SMSManager("api_key", "+491234567890", delay_minutes=5)
        manager.last_sms_time = datetime.now() - timedelta(minutes=2)
        
        success, message = manager.send_alert("Test alert", force=True)
        
        self.assertTrue(success)
        self.assertIn("successfully", message)
    
    def test_create_velutina_alert(self):
        """Test creating Asian hornet alert message"""
        frame_url = "http://example.com/frame/123"
        
        message = self.manager.create_hornet_alert("velutina", 2, 95.5, frame_url)
        
        self.assertIn("⚠️", message)
        self.assertIn("ALERT", message)
        self.assertIn("Asian Hornet", message)
        self.assertIn("2", message)
        self.assertIn("95.5%", message)
        self.assertIn(frame_url, message)
    
    def test_create_crabro_alert(self):
        """Test creating European hornet alert message"""
        frame_url = "http://example.com/frame/456"
        
        message = self.manager.create_hornet_alert("crabro", 1, 87.2, frame_url)
        
        self.assertIn("ℹ️", message)
        self.assertIn("Info", message)
        self.assertIn("European Hornet", message)
        self.assertIn("1", message)
        self.assertIn("87.2%", message)
        self.assertIn(frame_url, message)
    
    def test_create_alert_without_confidence(self):
        """Test creating alert message without confidence"""
        message = self.manager.create_hornet_alert("crabro", 1, 0, "")
        
        self.assertNotIn("%", message)
        self.assertNotIn("View:", message)


class TestSMSManagerEnvironment(unittest.TestCase):
    """Test environment-based SMS manager creation"""
    
    @patch.dict(os.environ, {
        'LOX24_API_KEY': 'test_key',
        'PHONE_NUMBER': '+491234567890',
        'LOX24_SENDER': 'VespAI',
        'SMS_DELAY_MINUTES': '10',
        'ENABLE_SMS': 'true'
    })
    @patch('src.vespai.sms.lox24.SMSManager')
    def test_create_from_env_success(self, mock_sms_manager):
        """Test successful SMS manager creation from environment"""
        mock_manager = Mock()
        mock_sms_manager.return_value = mock_manager
        
        result = create_sms_manager_from_env()
        
        self.assertIsNotNone(result)
        mock_sms_manager.assert_called_once_with(
            api_key='test_key',
            phone_number='+491234567890',
            sender_name='VespAI',
            delay_minutes=10,
            enabled=True
        )
    
    @patch.dict(os.environ, {}, clear=True)
    def test_create_from_env_no_api_key(self):
        """Test SMS manager creation with no API key"""
        result = create_sms_manager_from_env()
        self.assertIsNone(result)
    
    @patch.dict(os.environ, {'LOX24_API_KEY': 'test_key'})
    def test_create_from_env_no_phone_number(self):
        """Test SMS manager creation with no phone number"""
        result = create_sms_manager_from_env()
        self.assertIsNone(result)
    
    @patch.dict(os.environ, {
        'LOX24_API_KEY': 'test_key',
        'PHONE_NUMBER': '+491234567890',
        'ENABLE_SMS': 'false'
    })
    @patch('src.vespai.sms.lox24.SMSManager')
    def test_create_from_env_disabled(self, mock_sms_manager):
        """Test SMS manager creation when disabled"""
        mock_manager = Mock()
        mock_sms_manager.return_value = mock_manager
        
        result = create_sms_manager_from_env()
        
        mock_sms_manager.assert_called_once_with(
            api_key='test_key',
            phone_number='+491234567890',
            sender_name='VespAI',
            delay_minutes=5,
            enabled=False
        )


class TestSMSManagerIntegration(unittest.TestCase):
    """Integration tests for SMS manager"""
    
    @patch('src.vespai.sms.lox24.requests.post')
    def test_full_sms_workflow(self, mock_post):
        """Test complete SMS sending workflow"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": "msg_12345",
            "price": 0.08,
            "status": "sent"
        }
        mock_post.return_value = mock_response
        
        # Create manager and send alert
        manager = SMSManager("test_key", "+491234567890", delay_minutes=1)
        
        # Send first alert
        success1, msg1 = manager.send_alert("First alert")
        self.assertTrue(success1)
        self.assertIn("successfully", msg1)
        
        # Try to send second alert immediately (should be rate limited)
        success2, msg2 = manager.send_alert("Second alert")
        self.assertFalse(success2)
        self.assertIn("Rate limited", msg2)
        
        # Force send second alert
        success3, msg3 = manager.send_alert("Forced alert", force=True)
        self.assertTrue(success3)
        
        # Verify API was called twice (first + forced)
        self.assertEqual(mock_post.call_count, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)