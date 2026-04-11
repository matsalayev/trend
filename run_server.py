#!/usr/bin/env python3
"""
Trend Robot - Server Entry Point

Usage:
    python run_server.py

Or with custom port:
    SERVER_PORT=8090 python run_server.py
"""

import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trend_robot.server import run_server

if __name__ == "__main__":
    port = int(os.getenv("SERVER_PORT", "8090"))
    print(f"Starting Trend Robot server on port {port}...")
    run_server(port=port)
