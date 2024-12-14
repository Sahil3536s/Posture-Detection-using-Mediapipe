# Posture-Detection-using-Mediapipe
This repository demonstrates a real-time posture detection system using PoseNet. It employs machine learning models to identify and analyze body poses from video or image inputs. The project aims to encourage proper posture habits and provide a base for further research and development in the field of health tech and fitness tracking

## Overview
This project implements a real-time posture detection system using using [Mediapipe] a machine learning model that estimates human poses from image or video inputs. The application is designed to analyze body posture and provide insights into maintaining proper alignment, making it suitable for use in fitness, ergonomics, and rehabilitation.

## Features
- **Real-Time Posture Detection:** Uses a webcam or video feed to detect body postures.
- **Scalable:** Can be adapted for various use cases, including fitness tracking, yoga pose correction, and workplace ergonomics.
- **Open Source:** Easily extendable and modifiable for custom applications.

## Technologies Used
- **OpenCV:** For image classification.
- **TensorFlow.js:** For running the PoseNet model in the browser.
- **JavaScript/HTML/CSS:** For the front-end user interface.
- **Node.js (optional):** For back-end functionalities.

## How It Works
1. The model processes the input from the webcam or image feed.
2. It identifies key body points (e.g., shoulders, elbows, hips, knees).
3. Using these key points, it calculates angles and distances to determine the user’s posture.
4. Provides feedback if the posture deviates from a standard or pre-defined alignment.


