MediaPipe를 이용한 얼굴 랜드마크 비디오 데이터셋 구축

# 기능 설명
동영상 다운로드, 구간 슬라이싱, 크롭, face landmark 추가를 자동화 하는 프로젝트

# 실행방법
원하는 유튜브 영상의 watch?v= 이후의 video_id를 metadata파일에 추가
Final 폴더로 이동 후 터미널에 python ./Final.py 으로 실행
 
# 프로젝트 구성
- metadata.csv
-- 다운로드 하려는 유튜브 동영상의 video_id가 저장되어 있다.'

- main.py
-- download.py, slicer.py, crop.py, video_landmark.py가 순차적으로 실행되는 파일

- download.py
-- metadata.csv의 video id를 가져와서 yt-dlp로 mp4 파일로 다운로드 함. 이 과정에서 이미 동영상이 있는 경우 생략한다.
-- source_videos 폴더가 없는 경우 생성하여 내부에 {video_id}.mp4 이름으로 저장된다.
-- 실행 : python download.py -i {csv_file}
--- input을 주지 않는 경우 기본 값이 metadata.csv이다.

- slicer.py
-- 영상 내에서 bounding box가 원하는 시간 이상으로 검출될 경우 그 구간을 ffmpeg를 통해 슬라이싱하여 영상을 따로 저장한다.
-- slice_video_folder 폴더가 없는 경우 생성하여 내부에 video_id 별로 폴더를 생성하여 내부에 저장한다.
-- 실행 : 
## 동영상 파일 사용
    python slicer.py -i {input_video.mp4 or input_path} -o {output_path}
## 카메라 사용
    python slicer.py --camera
--- input은 폴더와 파일 둘 다 입력 가능하다.


- crop.py
-- 슬라이싱 한 영상에서 바운딩 박스가 생기는 영역을 모두 포함한 portrait region을 계산 해, 영역을 잘라낸다.
-- croped_folder 폴더가 없는 경우 생성하여 내부에 video_id 별로 폴더를 생성하여 내부에 저장한다.
-- 
-- 실행 : python crop.py -i {input_video.mp4 or input_path} -o {output_path}
--- input은 폴더와 파일 둘 다 입력 가능하다.

- video_landmark.py
-- croped_folder 내부에 있는 폴더, 파일을 순차적으로 실행시키며 face landmark를 띄워준다.
-- 조작키
--- 1번 : MediaPipe의 기본 얼굴 랜드마크 그리기 방식
--- 2번 : TESSELATION 없이 그리는 방식
--- 3번 : FACE_OVAL 없이 그리는 방식
--- 4번 : 검은색 배경에 FACE_OVAL 없이 그리는 방식
-- 실행: python video_landmark.py -i {input_video.mp4 or input_path} -v {version_number} -o {output_path}
 
- sequential.py
-- 전체 프로세스를 한번에 실행시킨다
-- 실행 : python sequential.py {-c or --camera}
-- 인자 -c or --camera를 줄 경우 카메라를 사용하고, 주지 않을 경우 metadata.csv의 영상을 처리하게 된다.

- utils
-- detected_face_landmark.py : landmark를 그려주는 파일
-- detected_face.py : 얼굴 감지 및 바운딩 박스를 그려주는 파일

- blaze_face_short_range.tflite
-- 실시간 얼굴 감지 모델


