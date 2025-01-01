# 🏗️🔒 **ConstrucSafe AI**: AI-Powered Safety Detection for Construction Sites

![ConstrucSafe AI Banner](https://via.placeholder.com/800x200.png?text=ConstrucSafe+AI+Banner)

---

## 🚀 **Table of Contents**

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training Details](#model-training-details)
- [Dataset Details](#dataset-details)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 📝 **Introduction**

Welcome to **ConstrucSafe AI**! 🏗️🔒

ConstrucSafe AI is a state-of-the-art application designed to enhance safety on construction sites through real-time object detection. Leveraging the power of **YOLOv8**, this tool detects and highlights safety equipment and potential hazards in images and videos, ensuring a safer working environment.

---

## ✨ **Features**

- **Image Detection**: Upload images to identify safety gear and hazards.
- **Video Detection**: Process and analyze videos to monitor safety compliance in real-time.
- **Customizable Thresholds**: Adjust confidence levels to fine-tune detection accuracy.
- **Downloadable Results**: Save processed images and videos with annotations for record-keeping.
- **User-Friendly Interface**: Intuitive design built with **Streamlit** for seamless user experience.

---

## 🛠️ **Technologies Used**

- **Python 3.10**
- **Streamlit** 🖥️: For building the interactive web application.
- **Ultralytics YOLOv8** 🤖: Advanced object detection model.
- **OpenCV** 📷: Image and video processing.
- **NumPy** 📊: Numerical computations.
- **Pillow** 🖼️: Image handling.
- **FFmpeg** 🎞️: Video encoding and processing.

---

## 📥 **Installation**

Follow these steps to set up ConstrucSafe AI on your local machine:

### 1. **Clone the Repository**

```bash
git clone https://github.com/alphatechlogics/ConstrucSafe.git
cd ConstrucSafe
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Note: Ensure that FFmpeg is installed on your system.

#### Install FFmpeg on Ubuntu

```bash
sudo apt update
sudo apt install ffmpeg
```

Verify FFmpeg Installation

```bash
ffmpeg -version
```

You should see version details if FFmpeg is installed correctly.

## 🎬 Usage

Run the Streamlit application using the following command:

```bash
streamlit run app.py
```

Once the app is running, navigate to http://localhost:8501 in your browser to access the ConstrucSafe AI interface.

### Application Workflow

1. Image Detection:

   - Navigate to the Image Detection tab.
   - Upload an image (.jpg, .jpeg, .png).
   - View the original and annotated images with detected safety equipment and hazards.
   - Download the annotated image for records.

2. Video Detection:

   - Navigate to the Video Detection tab.
   - Upload a video (.mp4, .mov, .avi) within the 200MB limit.
   - The app processes the video, highlighting detected objects frame-by-frame.
   - Once processing is complete, view the annotated video directly in the app.
   - Download the processed video for further analysis.

## 🧠 Model Training Details

The object detection model was trained using YOLOv8 on a custom dataset tailored for construction site safety. Below are the training details:

### Training Environment

- Platform: Google Colab
- Framework: Ultralytics YOLOv8
- Model Architecture: yolov8l.pt (YOLOv8 Large)

### Training Command

```bash
from ultralytics import YOLO

!yolo task=detect mode=train model=yolov8l.pt data='/content/drive/MyDrive/Construction Site Safety.v30-raw-images_latestversion.yolov8/data.yaml' epochs=10
```

### Training Output Highlights

- Total Epochs: 10
- Batch Size: 16
- Image Size: 640x640
- Optimizer: AdamW
- Loss Functions:
  - Box Loss: 7.5
  - Class Loss: 0.5
  - Distribution Focal Loss (DFL): 1.5
- Final Metrics:
  - mAP50: 0.516
  - mAP50-95: 0.333

### Model Summary

- Layers: 365
- Parameters: 43,649,115
- GFLOPs: 165.5

## 📚 Dataset Details

The model was trained on the Construction Site Safety - v30 raw-images_latestversion dataset, curated to enhance safety monitoring on construction sites.

### Dataset Source

[Construction Site Safety Dataset on Roboflow](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)

### Dataset Overview

- Total Images: 717
- Annotation Format: YOLOv8
- Classes: 25 (e.g., Excavator, Gloves, Hardhat, Ladder, Mask, NO-Hardhat, NO-Mask, etc.)
- Pre-processing Applied:
  - Auto-orientation of pixel data with EXIF-orientation stripping.
- Augmentation Techniques: None applied.
  Accessing the Dataset
  The dataset was exported via Roboflow on April 19, 2023.
