#!/usr/bin/env python3
"""
Simple HTTP server for the Document Classifier frontend.

Usage:
    python server.py [port]

Default port: 8080
"""

import http.server
import socketserver
import sys
import os

# Default port
PORT = 8080

# Get port from command line argument if provided
if len(sys.argv) > 1:
    try:
        PORT = int(sys.argv[1])
    except ValueError:
        print(f"Invalid port number: {sys.argv[1]}")
        print("Usage: python server.py [port]")
        sys.exit(1)

# Change to the frontend directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Configure handler
Handler = http.server.SimpleHTTPRequestHandler

# Add CORS headers
class CORSRequestHandler(Handler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

# Start server
with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
    print(f"""
╔════════════════════════════════════════════════════════════╗
║  Document Classifier Frontend Server                       ║
╠════════════════════════════════════════════════════════════╣
║  Server running at: http://localhost:{PORT:<24} ║
║                                                            ║
║  Press Ctrl+C to stop                                      ║
╚════════════════════════════════════════════════════════════╝
""")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        sys.exit(0)

