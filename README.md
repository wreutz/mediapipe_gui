# Mediapipe GUI

The aim of this project is to provide a stantdalone GUI for different platforms to wrap Google's mediapipe and provide a hassle-free usage of it for people without the need to write programs or code. It captures a live video feed (webcam), mediapipe analyzes the data and outputs the processed landmarks and points of interests via OSC to other interactive applications like TouchDesigner, Max/MSP, Processing or completely different applications and frameworks.

## Requirements

- Python 3.12 with an virtualenv
- class compliant webcam or video capture device

## Installation

1. use your OS's Python or - better - download and install Python from python.org
1. get this repository from Github
1. open a terminal in this folder
1. install a virtualenv for python with `python3.12 -m venv venv`
1. activate the virtualenv: `source venv/bin/activate`
1. update pip: `pip install --upgrade pip`
1. install requirements: `pip install -r requirements.txt``
1. run the app: `python mediapipe_gui.py`
1. have fun

## create a standalone application for your currently running OS platform

1. all the steps from "Installation" section
1. create application bundle with: `pyinstaller mediapipe_gui.spec`
1. copy the 'MediapipeGUI.app' to your /Applications folder

### create disk image for installing the app on another computer

- install homebrew (if not already present on your computer)
- install create-dmg with `brew install create-dmg`
- copy all neccesary files like the app and additional files (e.g. max patch, examples, ...) to ./dist_dmg/
- run `create-dmg --volname "Mediapipe GUI" --hide-extension "MediapipeGUI.app" --window-size 800 400 --window-pos 200 120 --app-drop-link 600 185 MediapipeGUI.dmg dist_dmg/`
