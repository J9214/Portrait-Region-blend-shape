import download
import slicer
import crop
import video_landmark
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", action="store_true")
    parser.add_argument("-v", "--version", type=int, default=1)

    args = parser.parse_args()

    csv_path = "metadata.csv"
    download_output_dir = "./source_videos"
    video_path = f"{download_output_dir}"
    sliced_output_path = "./slice_videos"
    cropped_output_path = "./cropped_videos"
    landmark_output_path = "./landmark_videos"

    if args.camera:
        slicer.cam_video(output_path=sliced_output_path, min_detect_duration=5)
    else:
        download.downloads_video_from_csv(csv_path, download_output_dir)
        print("Video download completed.")
        
        slicer.slice_video(video_path, output_path=sliced_output_path, min_detect_duration=5)
        print("Video slicing completed.")
    

    
    crop.crop_video(sliced_output_path, output_path=cropped_output_path)
    print("Video cropping completed.")
    
    video_landmark.draw_landmark(cropped_output_path, output_path=landmark_output_path, version=args.version)
    print("Landmark drawing completed.")

    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()
