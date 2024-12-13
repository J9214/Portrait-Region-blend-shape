import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.detected_face import visualize

#IMAGE_FILE = './brother-988180-1280.jpg'
video_path = './output.mp4'
output_path = './mediaPipe_output'

#img = cv.imread(IMAGE_FILE)
# cv.imshow("example", img)
# cv.waitKey(1000)
# cap = cv.VideoCapture(video_path) # 동영상
cap = cv.VideoCapture(0) # 웹 캠

# STEP 2: Create an ObjectDetector object.
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


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video failed to read frame")
        break

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    detection_result = detector.detect(mp_image)

    for detection in detection_result.detections:
        bbox = detection.bounding_box
        x_min, y_min = int(bbox.origin_x), int(bbox.origin_y)
        x_max, y_max = x_min + int(bbox.width), y_min + int(bbox.height)
        cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the frame with annotations
    cv.imshow("Face Detection", frame)

    # Break if 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
# # STEP 3: Load the input image.
# image = mp.Image.create_from_file(IMAGE_FILE)

# # STEP 4: Detect objects in the input image.
# detection_result = detector.detect(image)

# # STEP 5: Print the information of the Detect objects.
# for detection in detection_result.detections:
#     print(detection.bounding_box)
#     print(detection.keypoints)
#     print(detection.categories)

# # STEP 6: Visualization with OpenCV
# image_copy = np.copy(image.numpy_view())
# annotated_image = visualize(image=image_copy, detection_result=detection_result)
# annotated_cv_image = cv.c
# vtColor(annotated_image, cv.COLOR_RGB2BGR)
# cv.imshow("Object Detection", annotated_cv_image)
# cv.waitKey(10000)
# cv.imwrite("annotated_image.jpg", annotated_cv_image)
# cv.destroyAllWindows()