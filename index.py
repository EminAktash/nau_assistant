from flask import Flask, request
import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your Flask app
from nau_assistant_final import app

# This handles the serverless function entry point
def handler(request, **kwargs):
    return app(request.environ, start_response)