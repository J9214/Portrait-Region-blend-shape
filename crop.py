import os
import cv2 as cv
import numpy as np
import mediapipe as mp
import subprocess
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.detected_face import visualize

def crop_video(input_path="./slice_video_folder", output_path = "./croped_folder"):
    # 폴더 내 동영상 파일 차례로 접근
    #VIDEO_FILES = os.listdir(input_path)
    
    video_folders = [os.path.join(input_path, folder) for folder in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, folder))]

    if not video_folders:
        print(f"Not Fonund video_folders")
        return
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

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

            base_options = python.BaseOptions(model_asset_path='./blaze_face_short_range.tflite')
            options = vision.FaceDetectorOptions(base_options=base_options)
            detector = vision.FaceDetector.create_from_options(options)

            fps = cap.get(cv.CAP_PROP_FPS)
            frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            frame_size = (frame_width, frame_height)

            # 자르기 관련 변수 초기화
            min_x, min_y, max_x, max_y = frame_height, frame_width, 0, 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Video failed to read frame_read")
                    break
                
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                detection_result = detector.detect(mp_image)            

                for detection in detection_result.detections:
                    bbox = detection.bounding_box
                    x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
                    min_x, max_x, min_y, max_y = min(min_x, x), max(max_x, x + w), min(min_y, y), max(max_y, y + h)            
                
                if cv.waitKey(1) == ord('q'):
                    break

            min_x, max_x, min_y, max_y = max(min_x, 0), min(max_x, frame_width), max(min_y, 0), min(max_y, frame_height)

            cap.release()

            #### portrait region 영역 crop

            vid_output = cv.VideoWriter(f'./{output_folder}/{os.path.basename(file).split(".")[0]}_crop.mp4',  cv.VideoWriter_fourcc(*'avc1'), fps, (256,256))
            cap = cv.VideoCapture(file)

            if not cap.isOpened():
                print("Error: Cannot open video")
                exit()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Video failed to read frame_crop")
                    break

                portrait_region = frame[min_y:max_y, min_x:max_x]

                # 크롭된 영역을 256x256으로 리사이즈
                resized_frame = cv.resize(portrait_region, (256, 256))
                vid_output.write(resized_frame)        

                if cv.waitKey(1) == ord('q'):
                    break
            
            if cv.waitKey(1) == ord('q'):
                break
            
            cap.release()
            vid_output.release()

            print(f"Croped complete for video: {file}")
        cv.destroyAllWindows()

input_path = "./slice_video_folder"
output_path = "./croped_folder"
crop_video(input_path, output_path)
