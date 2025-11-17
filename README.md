# 🚗 Intelligent Parking Detection System
=========================================
Real-time parking slot occupancy detection using computer vision and a CNN classifier.
Slots are manually annotated, scaled to video dimensions, and classified as FREE or OCCUPIED live on video.
***********************************************************************************************************
## 📑 Table of Contents
- [Overview](#overview)
- [Goal of the-project](#goal-of-the-project)
- [Problem Statement](#problem-statement)
- [Project Workflow](#project-workflow)
- [Features](#features)
- [Tools & Technologies](#tools--technologies)
- [Requirements](#requirements)
- [How to Use](#how-to-use)
- [Evaluation Metrics](#evaluation-metrics)
- [Future Improvements](#future-improvements)
- [Author](#author)
************************************************************************************************************
### The system highlights:
	•	🟩 FREE slot
	•	🟥 OCCUPIED slot

⸻

## 🎯 Goal of the Project
	•	Build a real-time parking occupancy detector
	•	Use polygon-based slot annotation for accurate ROI extraction
	•	Classify each slot using a deep learning model
	•	Run efficiently on CPUs & Apple Silicon (Metal acceleration)

⸻

## Problem Statement

> "How can we automatically detect whether a parking slot is occupied or free in real-time using CCTV camera footage and deep learning, ensuring accurate monitoring across varying angles and lighting conditions?"

Challenges include:
	•	Varying lighting
	•	Cars with different shapes/colors
	•	Perspective distortions
	•	Camera shake/focus issues

***********************************************************************************************************************************************************************

🔄 Project Workflow

1. Data Preparation
	•	Capture a “reference frame”
	•	Manually annotate polygons for each parking slot
	•	Scale polygons to match video size

2. Model
	•	CNN classifier trained on:
	•	Occupied
	•	Free
	•	Saved as slot_classifier.h5

3. Inference Pipeline
	1.	Read each video frame
	2.	Crop each polygon region
	3.	Preprocess crop
	4.	Predict occupancy
	5.	Draw red/green overlays
	6.	Display FPS + output



