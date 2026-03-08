#!/usr/bin/env python3
"""
VespAI SMS Module - Lox24 SMS API Integration

This module handles SMS alerts for hornet detections using the Lox24 SMS service.
Provides rate limiting, cost tracking, and comprehensive error handling.

Author: Jakob Zeise (Zeise Digital)
Version: 1.0
"""

import json
import os
import datetime
import logging
import requests
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class Lox24SMS:
    """
    Lox24 SMS API client for sending SMS alerts.
    
    This class handles authentication, message sending, and cost tracking
    for the Lox24 SMS service used in VespAI hornet detection alerts.
    """
    
    def __init__(self, api_key: str, sender_name: str = "VespAI"):
        """
        Initialize the Lox24 SMS client.
        
        Args:
            api_key (str): Lox24 API key or username:password format
            sender_name (str): Sender name that appears on SMS messages
        """
        self.api_key = api_key
        self.sender_name = sender_name
        self.sms_available = True

        # Parse API key format (username:password or token)
        if ":" in api_key:
            self.username, self.password = api_key.split(":", 1)
        else:
            # Use API key as password with empty username
            self.username = ""
            self.password = api_key

    def send_sms(self, to: str, message: str) -> Tuple[bool, float]:
        """
        Send an SMS message via the Lox24 API.
        
        Args:
            to (str): Recipient phone number (e.g., "+491234567890")
            message (str): SMS message content
            
        Returns:
            Tuple[bool, float]: (success, cost) - Success status and message cost in EUR
        """
        if not self.sms_available:
            logger.info(f"[SMS disabled] Would send: {message}")
            return False, 0.0

        url = "https://api.lox24.eu/sms"

        try:
            logger.info(f"Sending SMS to {to}: {message[:50]}...")

            data = {
                'sender_id': self.sender_name,
                'text': message,
                'service_code': "direct",
                'phone': to,
                'delivery_at': 0,
                'is_unicode': True,
                'callback_data': '123456',
                'voice_lang': 'DE'
            }

            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'X-LOX24-AUTH-TOKEN': self.api_key,
            }

            logger.debug("SMS payload: %s", json.dumps(data, indent=4))

            # Send request with 100 second timeout
            response = requests.post(url, headers=headers, json=data, timeout=100)
            
            if response.status_code != 201:  # Created
                error_msg = self._handle_error_response(response)
                logger.error("SMS sending failed: %s", error_msg)
                return False, 0.0
            else:
                logger.info("✓ SMS sent successfully (status: %d)", response.status_code)
                response_data = response.json()
                logger.debug("SMS response: %s", json.dumps(response_data, indent=4))
                
                # Extract cost from response
                cost = self._extract_cost_from_response(response_data)
                return True, cost

        except requests.exceptions.RequestException as e:
            logger.error("SMS request error: %s", e)
            return False, 0.0

    def _handle_error_response(self, response: requests.Response) -> str:
        """
        Handle and format error responses from the Lox24 API.
        
        Args:
            response: The failed HTTP response
            
        Returns:
            str: Formatted error message
        """
        error_messages = {
            400: "Invalid input parameters",
            401: "Client ID or API key is invalid or inactive",
            402: "Insufficient funds in account",
            403: "Account not activated - contact support",
            404: "Resource not found",
            500: "Internal server error - contact LOX24 support",
            502: "Bad gateway - contact LOX24 support", 
            503: "Service unavailable - contact LOX24 support",
            504: "Gateway timeout - contact LOX24 support"
        }
        
        error_msg = error_messages.get(response.status_code, f"Unknown error (status: {response.status_code})")
        
        try:
            response_text = response.text
            logger.debug("Error response body: %s", response_text)
        except:
            response_text = "Unable to read response"
            
        return f"{error_msg}. Response: {response_text}"

    def _extract_cost_from_response(self, response_data: Dict[str, Any]) -> float:
        """
        Extract the SMS cost from the API response.
        
        Args:
            response_data: JSON response from Lox24 API
            
        Returns:
            float: Message cost in EUR, or 0.0 if not found
        """
        # Try different possible cost field names
        cost_fields = ['price', 'cost', 'total_price', 'amount']
        
        for field in cost_fields:
            if field in response_data:
                try:
                    return float(response_data[field])
                except (ValueError, TypeError):
                    continue
                    
        return 0.0


class SMSManager:
    """
    SMS Manager with rate limiting and statistics tracking.
    
    Manages SMS sending with configurable delays, cost tracking,
    and integration with VespAI statistics.
    """
    
    def __init__(self, 
                 api_key: str, 
                 phone_number: str,
                 sender_name: str = "VespAI",
                 delay_minutes: int = 5,
                 enabled: bool = True):
        """
        Initialize the SMS Manager.
        
        Args:
            api_key (str): Lox24 API key
            phone_number (str): Target phone number for alerts
            sender_name (str): SMS sender name
            delay_minutes (int): Minimum delay between SMS messages
            enabled (bool): Whether SMS sending is enabled
        """
        self.phone_number = phone_number
        self.delay_minutes = delay_minutes
        self.enabled = enabled
        self.last_sms_time: Optional[datetime.datetime] = None
        
        # Initialize SMS client
        if api_key and phone_number and enabled:
            self.client = Lox24SMS(api_key, sender_name)
        else:
            self.client = None
            logger.warning("SMS not configured - alerts disabled")
    
    def send_alert(self, message: str, force: bool = False) -> Tuple[bool, str]:
        """
        Send an SMS alert with rate limiting.
        
        Args:
            message (str): Alert message to send
            force (bool): Whether to bypass rate limiting
            
        Returns:
            Tuple[bool, str]: (success, status_message)
        """
        # Check if SMS is enabled and configured
        if not self.enabled:
            return False, "SMS disabled in configuration"
            
        if not self.client:
            return False, "SMS client not initialized"
            
        if not self.phone_number:
            return False, "No phone number configured"
        
        # Check rate limiting
        if not force and self.last_sms_time is not None:
            time_since_last = (datetime.datetime.now() - self.last_sms_time).total_seconds() / 60
            
            if time_since_last < self.delay_minutes:
                remaining = self.delay_minutes - time_since_last
                return False, f"Rate limited - next SMS allowed in {remaining:.1f} minutes"
        
        # Send the SMS
        success, cost = self.client.send_sms(self.phone_number, message)
        
        if success:
            self.last_sms_time = datetime.datetime.now()
            status_msg = f"SMS sent successfully"
            if cost > 0:
                status_msg += f" (cost: {cost:.3f}€)"
            return True, status_msg
        else:
            return False, "Failed to send SMS"
    
    def create_hornet_alert(self, 
                          hornet_type: str, 
                          count: int, 
                          confidence: float,
                          frame_url: str) -> str:
        """
        Create a formatted hornet detection alert message.
        
        Args:
            hornet_type (str): Type of hornet ('velutina' or 'crabro')
            count (int): Number of hornets detected
            confidence (float): Detection confidence percentage
            frame_url (str): URL to view the detection frame
            
        Returns:
            str: Formatted alert message
        """
        time_str = datetime.datetime.now().strftime('%H:%M')
        
        if hornet_type == 'velutina':
            emoji = "⚠️"
            species = "Asian Hornet"
            urgency = "ALERT"
        else:
            emoji = "ℹ️"
            species = "European Hornet"
            urgency = "Info"
            
        plural = "s" if count > 1 else ""
        
        message = f"{emoji} {urgency}: {count} {species}{plural} detected at {time_str}"
        
        if confidence > 0:
            message += f" ({confidence:.1f}% confidence)"
            
        if frame_url:
            message += f". View: {frame_url}"
            
        return message


def create_sms_manager_from_env() -> Optional[SMSManager]:
    """
    Create an SMS manager from environment variables.
    
    Expected environment variables:
    - LOX24_API_KEY: Lox24 API credentials
    - PHONE_NUMBER: Target phone number
    - LOX24_SENDER: SMS sender name (optional)
    - SMS_DELAY_MINUTES: Delay between messages (optional)
    - ENABLE_SMS: Whether SMS is enabled (optional)
    
    Returns:
        SMSManager: Configured SMS manager or None if not properly configured
    """
    api_key = os.getenv("LOX24_API_KEY", "")
    phone_number = os.getenv("PHONE_NUMBER", "")
    sender_name = os.getenv("LOX24_SENDER", "VespAI")
    delay_minutes = int(os.getenv("SMS_DELAY_MINUTES", "5"))
    enabled = os.getenv("ENABLE_SMS", "true").lower() == "true"
    
    if not api_key:
        logger.warning("LOX24_API_KEY not set - SMS alerts disabled")
        return None
        
    if not phone_number:
        logger.warning("PHONE_NUMBER not set - SMS alerts disabled") 
        return None
    
    try:
        return SMSManager(
            api_key=api_key,
            phone_number=phone_number,
            sender_name=sender_name,
            delay_minutes=delay_minutes,
            enabled=enabled
        )
    except Exception as e:
        logger.error("Failed to initialize SMS manager: %s", e)
        return None