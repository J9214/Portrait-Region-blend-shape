MediaPipe를 이용한 얼굴 랜드마크 비디오 데이터셋 구축
이 프로젝트는 동영상 다운로드, 구간 슬라이싱, 크롭, 그리고 얼굴 랜드마크 추출을 자동화하는 도구입니다. MediaPipe 라이브러리를 사용하여 얼굴 랜드마크를 추출하고, 이를 바탕으로 비디오 데이터셋을 생성합니다.

🚀 기능 설명
동영상 다운로드: 유튜브 영상에서 동영상을 다운로드합니다.

구간 슬라이싱: 비디오에서 관심 있는 구간을 슬라이싱합니다.

크롭: 얼굴이 포함된 영역을 잘라냅니다.

Face Landmark 추가: 비디오에서 얼굴 랜드마크를 추출하여 표시합니다.

🖥️ 실행 방법
metadata.csv 파일에 원하는 유튜브 영상의 video_id를 추가합니다.

Final 폴더로 이동한 후 터미널에서 아래 명령어를 실행하여 전체 프로세스를 시작합니다.

bash
복사
python ./Final.py
📁 프로젝트 구성
metadata.csv: 다운로드할 유튜브 동영상의 video_id가 저장되어 있습니다.

main.py: download.py, slicer.py, crop.py, video_landmark.py가 순차적으로 실행되는 파일입니다.

download.py: metadata.csv에서 video_id를 가져와서 yt-dlp로 mp4 파일을 다운로드합니다. 이미 동영상이 있으면 다운로드를 생략합니다.

slicer.py: 비디오 내에서 바운딩 박스가 원하는 시간 이상으로 감지되면 그 구간을 ffmpeg로 슬라이싱하여 영상을 따로 저장합니다.

crop.py: 슬라이싱한 영상에서 얼굴이 포함된 영역을 크롭하여 잘라냅니다.

video_landmark.py: 크롭한 영상에 대해 MediaPipe를 사용하여 얼굴 랜드마크를 그립니다.

🔧 세부 기능 설명
1. 다운로드 (download.py)
metadata.csv의 video_id를 기반으로 유튜브 영상을 다운로드합니다.

이미 다운로드된 동영상은 생략됩니다.

실행 방법:

bash
복사
python download.py -i {csv_file}
기본값은 metadata.csv입니다.

2. 슬라이싱 (slicer.py)
비디오에서 얼굴을 포함한 구간을 ffmpeg를 사용하여 슬라이싱합니다.

실행 방법:

비디오 파일을 지정하여 슬라이싱:

bash
복사
python slicer.py -i {input_video.mp4 or input_path} -o {output_path}
카메라를 사용하여 실시간으로 슬라이싱:

bash
복사
python slicer.py --camera
3. 크롭 (crop.py)
슬라이싱한 영상에서 얼굴 영역을 포함한 부분을 잘라냅니다.

실행 방법:

bash
복사
python crop.py -i {input_video.mp4 or input_path} -o {output_path}
4. Face Landmark 추가 (video_landmark.py)
크롭된 영상에서 MediaPipe를 사용하여 얼굴 랜드마크를 그립니다.

실행 방법:

bash
복사
python video_landmark.py -i {input_video.mp4 or input_path} -v {version_number} -o {output_path}
조작키:

1: MediaPipe 기본 얼굴 랜드마크 그리기

2: 테셀레이션 없이 그리기

3: 얼굴 윤곽(FACE_OVAL) 없이 그리기

4: 검은 배경에 얼굴 윤곽 없이 그리기

5. 전체 프로세스 실행 (sequential.py)
모든 단계를 한번에 실행합니다.

실행 방법:

비디오 파일을 처리:

bash
복사
python sequential.py
카메라를 사용:

bash
복사
python sequential.py -c or --camera
🧰 기타 파일
detected_face_landmark.py: 랜드마크를 그려주는 파일입니다.

detected_face.py: 얼굴을 감지하고 바운딩 박스를 그려주는 파일입니다.

blaze_face_short_range.tflite: 실시간 얼굴 감지 모델입니다.

