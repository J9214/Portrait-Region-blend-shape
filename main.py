import download
import slicer
import crop
import video_landmark

def main():
    # 1. 비디오 다운로드
    csv_path = "metadata.csv"
    download_output_dir = "./source_videos"
    download.downloads_video_from_csv(csv_path, download_output_dir)

    # 2. 비디오 자르기
    video_path = f"{download_output_dir}"
    sliced_output_path = "./slice_video_folder"
    slicer.slice_video(video_path, output_path=sliced_output_path, min_detect_duration=5)

    # 3. 비디오 크롭
    cropped_output_path = "./croped_video.mp4"
    crop.crop_video(sliced_output_path, output_path=cropped_output_path)

    # 4. 랜드마크 그리기
    video_landmark.draw_landmark(cropped_output_path)

    print("모든 작업이 완료되었습니다!")

# if __name__ == "__main__":
#     print("Script is starting...")
#     main()
#     print("Script has finished execution.")
