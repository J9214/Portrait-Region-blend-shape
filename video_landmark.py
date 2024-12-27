import os
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.detected_face_landmark import draw_landmarks_on_image, plot_face_blendshapes_bar_graph
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import argparse

def save_metadata(file_path, video_length, frame_size, fps, version):
    """Save metadata for a given video."""
    metadata_file = f"{os.path.splitext(file_path)[0]}.txt"

    metadata_content = (
        f"Video File: {os.path.basename(file_path)}\n"
        f"Video Length: {video_length:.2f} seconds\n"
        f"Frame Size: {frame_size[0]}x{frame_size[1]}\n"
        f"FPS: {fps}\n"
        f"Landmark Version: {version}\n"
    )

    with open(metadata_file, "w") as file:
        file.write(metadata_content)

    print(f"Metadata saved to {metadata_file}")

def draw_landmark(input_path, output_path, version):
    def process_folder(folder_path, output_base_folder):
        video_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith((".mp4", ".avi", ".mov", ".webm"))]

        if not video_files:
            return  # 비디오 파일이 없는 폴더는 스킵

        # 현재 폴더의 이름을 가져와서 버전 폴더 경로 생성
        folder_name = os.path.basename(folder_path)
        version_folder = os.path.join(output_base_folder, folder_name, f"version_{version}")
        
        if not os.path.exists(version_folder):
            os.makedirs(version_folder)

        base_options = python.BaseOptions(model_asset_path='./face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        detector = vision.FaceLandmarker.create_from_options(options)

        for file in video_files:
            cap = cv.VideoCapture(file)
            
            if not cap.isOpened():
                print(f"Error: Cannot open video {file}")
                continue

            fps = cap.get(cv.CAP_PROP_FPS)
            frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            frame_size = (frame_width, frame_height)
            video_length = cap.get(cv.CAP_PROP_FRAME_COUNT) / fps

            video_name = os.path.splitext(os.path.basename(file))[0]
            output_file_path = os.path.join(version_folder, f"{video_name}_landmarks_v{version}.mp4")
            vid_output = cv.VideoWriter(output_file_path, cv.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print(f"Finished processing video: {file}")
                    break

                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                detection_result = detector.detect(mp_frame)

                annotated_frame = draw_landmarks_on_image(cv.cvtColor(frame, cv.COLOR_BGR2RGB), detection_result, version)
                annotated_cv_frame = cv.cvtColor(annotated_frame, cv.COLOR_RGB2BGR)

                vid_output.write(annotated_cv_frame)

                if cv.waitKey(1) == ord('q'):
                    print("Interrupted by user.")
                    break

            cap.release()
            vid_output.release()
            save_metadata(output_file_path, video_length, frame_size, fps, version)

    def recursive_process(folder_path, output_path):
        for root, _, _ in os.walk(folder_path):
            process_folder(root, output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    recursive_process(input_path, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="./cropped_videos")
    parser.add_argument("-o", "--output", default="./landmark_videos")
    parser.add_argument("-v", "--version", type=int, default=1)

    args = parser.parse_args()

    draw_landmark(input_path=args.input, output_path=args.output, version=args.version)

if __name__ == "__main__":
    main()
