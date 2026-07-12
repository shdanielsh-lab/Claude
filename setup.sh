#!/bin/bash
set -e
apt-get update -qq
apt-get install -y -qq ffmpeg libportaudio2
python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
