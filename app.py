# app.py

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from helper import reencode_video

# Initialize YOLO model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Ensure 'best.pt' is in the same directory or provide the correct path
    return model

model = load_model()

# Define class names
classNames = [
    'Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask',
    'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus',
    'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi',
    'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader'
]

# Define colors for classes
def get_color(class_name):
    if class_name in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
        return (0, 0, 255)  # Red
    elif class_name in ['Hardhat', 'Safety Vest', 'Mask']:
        return (0, 255, 0)  # Green
    else:
        return (255, 0, 0)  # Blue

st.title("Construction Site Safety Detection")
st.write("Upload an image or video, and the model will detect and highlight safety equipment and potential hazards.")

# Create tabs for Image and Video
tabs = st.tabs(["Image Detection", "Video Detection"])

### Image Detection Tab ###
with tabs[0]:
    st.header("Image Detection")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display original image
        image = Image.open(uploaded_image).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Convert the image to a format suitable for OpenCV
        img_array = np.array(image)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Perform detection
        results = model(img)

        # Process results
        annotated_img = img.copy()
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Confidence
                conf = float(box.conf[0])
                # Class Name
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if conf > 0.5:
                    color = get_color(currentClass)

                    # Draw rectangle
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

                    # Prepare label with white background and black text
                    label = f"{currentClass} {conf:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    
                    # Ensure the label does not go above the image
                    y_label = max(y1 - text_height - baseline, 0)

                    # Draw white rectangle for label background
                    cv2.rectangle(
                        annotated_img,
                        (x1, y_label),
                        (x1 + text_width, y1),
                        (255, 255, 255),  # White background
                        -1
                    )
                    
                    # Put black text on the white background
                    cv2.putText(
                        annotated_img,
                        label,
                        (x1, y1 - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),  # Black text
                        1
                    )

        # Convert annotated image back to RGB
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_img)

        # Display annotated image
        st.image(annotated_pil, caption='Detected Image', use_container_width=True)

### Video Detection Tab ###
with tabs[1]:
    st.header("Video Detection")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        # Check file size (optional, e.g., limit to 200MB)
        max_file_size = 200 * 1024 * 1024  # 200MB
        if uploaded_video.size > max_file_size:
            st.error("Uploaded video exceeds the 200MB limit.")
        else:
            # Save the uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
                tmp_input.write(uploaded_video.read())
                input_video_path = tmp_input.name

            # Prepare the output video path
            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            # Open the video file
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                st.error("Error opening video file.")
            else:
                st.write("Processing video...")

                # Create video writer using helper.py
                writer = create_video_writer(cap, output_video_path)

                # Define progress bar
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress_bar = st.progress(0)
                frame_counter = 0

                # Process each frame
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Perform detection
                    results = model(frame)

                    # Process results
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # Bounding Box
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                            # Confidence
                            conf = float(box.conf[0])
                            # Class Name
                            cls = int(box.cls[0])
                            currentClass = classNames[cls]

                            if conf > 0.5:
                                color = get_color(currentClass)

                                # Draw rectangle
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                                # Prepare label with white background and black text
                                label = f"{currentClass} {conf:.2f}"
                                (text_width, text_height), baseline = cv2.getTextSize(
                                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                                )
                                
                                # Ensure the label does not go above the image
                                y_label = max(y1 - text_height - baseline, 0)

                                # Draw white rectangle for label background
                                cv2.rectangle(
                                    frame,
                                    (x1, y_label),
                                    (x1 + text_width, y1),
                                    (255, 255, 255),  # White background
                                    -1
                                )
                                
                                # Put black text on the white background
                                cv2.putText(
                                    frame,
                                    label,
                                    (x1, y1 - baseline),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 0),  # Black text
                                    1
                                )

                    # Write the annotated frame to the output video
                    writer.write(frame)

                    # Update progress bar
                    frame_counter += 1
                    if total_frames > 0:
                        progress = frame_counter / total_frames
                        progress_bar.progress(min(progress, 1.0))

                # Release resources
                cap.release()
                writer.release()
                progress_bar.empty()

                st.write("Re-encoding video for compatibility...")

                # Re-encode the video using FFmpeg to ensure compatibility
                processed_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                reencode_success = reencode_video(output_video_path, processed_video_path)

                if reencode_success:
                    # Read the re-encoded video
                    with open(processed_video_path, 'rb') as f:
                        video_bytes = f.read()

                    # Display the processed video
                    st.video(video_bytes)

                    # Provide a download button for the processed video
                    st.download_button(
                        label="Download Processed Video",
                        data=video_bytes,
                        file_name="Processed_Video.mp4",
                        mime="video/mp4"
                    )

                    # Clean up temporary files
                    os.remove(input_video_path)
                    os.remove(output_video_path)
                    os.remove(processed_video_path)
                else:
                    st.error("Failed to re-encode the video. Please try again.")
                    # Clean up temporary files even if re-encoding fails
                    os.remove(input_video_path)
                    os.remove(output_video_path)
                    os.remove(processed_video_path)
