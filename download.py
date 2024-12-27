import os
import subprocess
import csv
import argparse

def download_video(video_id, output_folder):
    try:
        command = [
            "yt-dlp",
            f"https://www.youtube.com/watch?v={video_id}",
            "-o", f"{output_folder}/{video_id}s.%(ext)s",
            "--recode-video", "mp4"
        ]
        subprocess.run(command, check=True)
        print(f"동영상 다운로드 완료: {video_id}")
    except subprocess.CalledProcessError as e:
        print(f"동영상 다운로드 실패: {video_id}, Error: {e}")

def downloads_video_from_csv(csv_file, output_folder):
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        with open(csv_file, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                video_id = row.get("video_id")  

                # 이미 동영상이 있는 경우 continue
                if os.path.isfile(os.path.join(output_folder, f"{video_id}.mp4")):
                    print(f"already have {video_id}.mp4")
                    continue
                download_video(video_id, output_folder)

    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {csv_file}")
    except Exception as e:
        print(f"오류 발생 : {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="./metadata.csv")
    parser.add_argument("-o", "--output", default="./source_videos")

    args = parser.parse_args()

    downloads_video_from_csv(csv_file=args.input, output_folder=args.output)

if __name__ == "__main__":
    main()