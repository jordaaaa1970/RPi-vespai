#!/usr/bin/env python3
"""
VespAI Entry Point

Simple entry point script for running the modular VespAI application.
This script allows running VespAI from the project root directory.

Author: Jakob Zeise (Zeise Digital)
Version: 1.0

Usage:
    python vespai.py --web --confidence 0.8
    python vespai.py --help
"""

import sys
import os

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Import and run the main application
from vespai.main import main

if __name__ == '__main__':
    main()