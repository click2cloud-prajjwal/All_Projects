import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# Load YOLO model from last checkpoint
model = YOLO("D:/Weed_and_Disease_Detector/best.pt")

# Streamlit UI
st.title("Weed Detection in Paddy Fields")
st.write("Upload an image or video to detect weeds using YOLOv11")

# File Upload
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mpeg4", "webm"])

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension in ["jpg", "jpeg", "png"]:
        # Process Image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        st.write("Detecting weeds...")
        results = model.predict(source=np.array(image), conf=0.5, device="cpu")

        # Display results
        for result in results:
            img_with_boxes = result.plot()
            st.image(img_with_boxes, caption="Predicted Image", use_container_width=True)

            # Display detected objects
            st.subheader("Detected Objects:")
            for box in result.boxes.data:
                class_id = int(box[5].item())
                confidence = box[4].item()
                class_name = model.names[class_id]
                st.write(f"{class_name}: {confidence:.2f}")

    elif file_extension in ["mp4", "avi", "mov", "mpeg4", "webm"]:
        st.write("Processing video for weed detection...")

        # Save uploaded video to a temporary file
        temp_input_video = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}")
        temp_input_video.write(uploaded_file.read())
        temp_input_video.close()

        # Open the video with OpenCV
        cap = cv2.VideoCapture(temp_input_video.name)

        if not cap.isOpened():
            st.error("Error loading video. Please try another format or re-upload.")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Debug: Print video properties
        st.write(f"Video Properties - FPS: {fps}, Width: {frame_width}, Height: {frame_height}")

        # Define output video file
        temp_output_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_path = temp_output_video.name

        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Use 'XVID' instead of 'mp4v'
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("End of video reached or error reading frames.")
                break  # End of video

            # Convert frame to RGB (YOLO expects RGB format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run YOLO detection
            results = model.predict(source=frame_rgb, conf=0.5, device="cpu")

            for result in results:
                frame = result.plot()

            # Write frame to output video
            out.write(frame)

        # Release resources
        cap.release()
        out.release()

        # Success Message
        st.success("Video processing completed!")

        # Allow the user to download the processed video
        with open(output_path, "rb") as file:
            video_bytes = file.read()
            st.download_button(label="Download Processed Video", data=video_bytes, file_name="weed_detected.mp4", mime="video/mp4")

        # Clean up temporary files
        os.remove(temp_input_video.name)
        os.remove(output_path)


