import os
import cv2 as cv
import numpy as np
import mediapipe as mp
import subprocess
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.detected_face import visualize

def cam_video(output_path="./slice_video_folder", min_detect_duration=5):
    cap = cv.VideoCapture(0)

    base_options = python.BaseOptions(model_asset_path='./blaze_face_short_range.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    if not cap.isOpened():
        print("Error: Cannot open video")
        exit()

    fps = cap.get(cv.CAP_PROP_FPS)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    #vid_output = cv.VideoWriter('./output.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, frame_size)
    fourcc = cv.VideoWriter_fourcc('M', 'P', '4', 'V')  # Codec for output video

    folder_num = 1
    while os.path.exists(f"{output_path}/cam_{folder_num}"):
        folder_num += 1

    output_folder = f"{output_path}/cam_{folder_num}"
    os.makedirs(output_folder, exist_ok=True)

    start_time = None
    face_detected = False
    segment_index = 1
    detected_duration = 0  # Duration face detected (in seconds)
    out_writer = None  # To write the video segments

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or failed to capture frame.")
            break

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        try:
            detection_result = detector.detect(mp_image)
        except Exception as e:
            print(f"Detection error: {e}")
            continue

        has_faces = len(detection_result.detections) > 0

        if has_faces:
            if not face_detected:
                start_time = cap.get(cv.CAP_PROP_POS_MSEC) / 1000
                detected_duration = 0
                face_detected = True

                # Start writing the new segment
                segment_filename = os.path.join(output_folder, f"segment_{segment_index}.avi")
                out_writer = cv.VideoWriter(segment_filename, fourcc, fps, frame_size)

            detected_duration += 1 / fps
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x_min, y_min = int(bbox.origin_x), int(bbox.origin_y)
                x_max, y_max = x_min + int(bbox.width), y_min + int(bbox.height)
                cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Write frame to video
            if out_writer:
                out_writer.write(frame)

        else:
            if face_detected:
                if detected_duration >= min_detect_duration:
                    print(f"Saved segment: segment_{segment_index}.mp4")
                    segment_index += 1
                face_detected = False
                detected_duration = 0

                # Release writer for the current segment
                if out_writer:
                    out_writer.release()
                    out_writer = None

        # Display the frame with bounding boxes
        cv.imshow("Live Detection", frame)

        # Break if 'q' is pressed
        if cv.waitKey(1) == ord('q'):
            break

    # Handle last segment if still recording
    if face_detected and detected_duration >= min_detect_duration:
        print(f"Saved segment: segment_{segment_index}.mp4")
        if out_writer:
            out_writer.release()

    cap.release()
    cv.destroyAllWindows()
    print(f"All face-detected segments saved in: {output_folder}")  

def slice_video(input_path="./source_videos", output_path = "./slice_video_folder", min_detect_duration = 5):
    
    VIDEO_FILES = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith((".mp4", ".avi", ".mov", ".webm"))]
    
    if not VIDEO_FILES:
        print(f"Not Found VIDEO FILES")
        return
    
    for file in VIDEO_FILES:
        cap = cv.VideoCapture(file) # 동영상
        
        base_options = python.BaseOptions(model_asset_path='./blaze_face_short_range.tflite')
        options = vision.FaceDetectorOptions(base_options=base_options)
        detector = vision.FaceDetector.create_from_options(options)
        
        
        if not cap.isOpened():
            print("Error: Cannot open video")
            exit()

        fps = cap.get(cv.CAP_PROP_FPS)
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_size = (frame_width, frame_height)
        duration = frame_count / fps

         # 자르기 관련 변수 초기화
        start_time = None
        face_detected = False
        segment_index = 1
        detected_duration = 0  # 얼굴 검출 지속 시간 (초)
        
        output_folder = f"{output_path}/{os.path.basename(file).split(".")[0]}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        while cap.isOpened():
            frame_position = cap.get(cv.CAP_PROP_POS_MSEC) / 1000
            ret, frame = cap.read()
            if not ret:
                print("Video failed to read frame")
                break
            
            try:
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                detection_result = detector.detect(mp_image)
            except Exception:
                continue
        
            has_faces = len(detection_result.detections) > 0

            if has_faces:
                if not face_detected:
                    start_time = frame_position
                    detected_duration = 0
                    face_detected = True
                detected_duration += 1 / fps
            else :
                if face_detected and start_time is not None:
                    if detected_duration >= min_detect_duration:
                        end_time = frame_position
                        output_segment = os.path.join(output_folder, f"{os.path.basename(file).split('.')[0]}_{segment_index}.mp4")
                        command = ["ffmpeg", "-i", file,
                            "-ss", str(start_time), "-to", str(end_time),
                            "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "128k", output_segment]
                        subprocess.run(command, check=True)
                        print(f"Saved segment : {output_segment}")
                        segment_index += 1
                    face_detected = False
            
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x_min, y_min = int(bbox.origin_x), int(bbox.origin_y)
                x_max, y_max = x_min + int(bbox.width), y_min + int(bbox.height)
                cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Break if 'q' is pressed
            if cv.waitKey(1) == ord('q'):
                break
        if face_detected and start_time is not None and detected_duration >= min_detect_duration:
            end_time = duration
            output_segment = os.path.join(output_folder, f"{os.path.basename(file).split('.')[0]}_segment_{segment_index}.mp4")
            command = [
                "ffmpeg", "-i", file,
                "-ss", str(start_time), "-to", str(end_time),
                "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k", output_segment
            ]

            subprocess.run(command, check=True)
            print(f"Saved segment: {output_segment}")

        # Release resources
        cap.release()
        cv.destroyAllWindows()

        print(f"Slicing complete for video: {file}")

input_path = "./source_videos"
output_path = "./slice_video_folder"
cam_video(output_path, min_detect_duration=5)
# slice_video(input_path, output_path, min_detect_duration=5)
