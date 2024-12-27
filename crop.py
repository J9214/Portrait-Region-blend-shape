import os
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import argparse

def crop_video(input_path, output_path):
    def process_folder(folder_path, output_folder):
        video_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith((".mp4", ".avi", ".mov", ".webm"))]
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        base_options = python.BaseOptions(model_asset_path='./blaze_face_short_range.tflite')
        options = vision.FaceDetectorOptions(base_options=base_options)
        detector = vision.FaceDetector.create_from_options(options)

        for file in video_files:
            file_path = os.path.abspath(file)
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            cap = cv.VideoCapture(file_path)

            if not cap.isOpened():
                print(f"Error: Cannot open video {file_path}")
                continue

            fps = cap.get(cv.CAP_PROP_FPS)
            frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

            # 자르기 관련 변수 초기화
            min_x, min_y, max_x, max_y = frame_height, frame_width, 0, 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print(f"Finished processing video: {file_path}")
                    break

                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                detection_result = detector.detect(mp_image)

                for detection in detection_result.detections:
                    bbox = detection.bounding_box
                    x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
                    min_x, max_x, min_y, max_y = min(min_x, x), max(max_x, x + w), min(min_y, y), max(max_y, y + h)

            if max_y - min_y > max_x - min_x:
                diff = max_y - min_y - max_x + min_x
                max_x = max_x + diff // 2
                min_x = min_x - diff // 2
            else:
                diff = max_x - min_x - max_y + min_y
                max_y = max_y + diff // 2
                min_y = min_y - diff // 2

            min_x, max_x, min_y, max_y = max(min_x, 0), min(max_x, frame_width), max(min_y, 0), min(max_y, frame_height)
            cap.release()

            # portrait region 영역 crop
            output_file_path = os.path.join(output_folder, f"{os.path.basename(file).split('.')[0]}_crop.mp4")
            vid_output = cv.VideoWriter(output_file_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (256, 256))

            cap = cv.VideoCapture(file_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                portrait_region = frame[min_y:max_y, min_x:max_x]

                # 크롭된 영역을 256x256으로 리사이즈
                resized_frame = cv.resize(portrait_region, (256, 256))
                vid_output.write(resized_frame)

            cap.release()
            vid_output.release()

            print(f"Cropped complete for video: {file_path}")

    def recursive_process(folder_path, output_path):
        for root, dirs, files in os.walk(folder_path):
            relative_path = os.path.relpath(root, folder_path)
            current_output_folder = os.path.join(output_path, relative_path)
            process_folder(root, current_output_folder)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if os.path.isdir(input_path):
        base_folder_name = os.path.basename(os.path.normpath(input_path))
        final_output_path = os.path.join(output_path, base_folder_name)
        if not os.path.exists(final_output_path):
            os.makedirs(final_output_path)
        recursive_process(input_path, final_output_path)
    else:
        base_folder_name = os.path.basename(os.path.dirname(input_path))
        final_output_path = os.path.join(output_path, base_folder_name)
        if not os.path.exists(final_output_path):
            os.makedirs(final_output_path)
        process_folder(os.path.dirname(input_path), final_output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="./slice_videos")
    parser.add_argument("-o", "--output", default="./cropped_videos")

    args = parser.parse_args()

    crop_video(input_path=args.input, output_path=args.output)

if __name__ == "__main__":
    main()
