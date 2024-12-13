import os
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.detected_face_landmark import draw_landmarks_on_image, plot_face_blendshapes_bar_graph
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


def draw_landmark(input_path="./croped_folder", output_path="./landmark_videos"):
    # 폴더 내 동영상 파일 차례로 접근
    #VIDEO_FILES = os.listdir(input_path)
    
    video_folders = [os.path.join(input_path, folder) for folder in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, folder))]

    if not video_folders:
        print(f"Not Fonund video_folders")
        return
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    base_options = python.BaseOptions(model_asset_path = './face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    version = 1

    for folder in video_folders:

        # 2. 각 폴더 내의 동영상 파일 순회
        video_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith((".mp4", ".avi", ".mov", ".webm"))]

        if not video_files:
            print(f"Not Fonund video_folders")
            continue
    
        output_folder = f"{output_path}/{os.path.basename(folder)}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file in video_files:
            cap = cv.VideoCapture(file) # 동영상
            
            if not cap.isOpened():
                print("Error: Cannot open video")
                exit()

            fps = cap.get(cv.CAP_PROP_FPS)
            frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            frame_size = (frame_width, frame_height)

            while cap.isOpened():
                
                ret, frame = cap.read()
                if not ret:
                    print("Video failed to read frame_read")
                    break
                
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                detection_result = detector.detect(mp_frame)

                key = cv.waitKey(1)
                if key == ord('1'):
                    version = 1
                elif key == ord('2'):
                    version = 2
                elif key == ord('3'):
                    version = 3
                elif key == ord('4'): 
                    version = 4

                annotated_frame = draw_landmarks_on_image(cv.cvtColor(frame, cv.COLOR_BGR2RGB), detection_result, version)
                annotated_cv_frame = cv.cvtColor(annotated_frame, cv.COLOR_RGB2BGR)

                cv.imshow("Visualizing face landmark", annotated_cv_frame)             

                if key == ord('q'):
                    break

            cap.release()
            cv.destroyAllWindows()

input_path = './croped_folder'
draw_landmark(input_path)