import os
import cv2
from glob import glob
from mtcnn import MTCNN

# 이미지 파일 경로 설정
image_path = "/home/alpaco/REAL_LAST/effnet/data/dfdc"
image_paths = glob(image_path+'/*.jpg')

# 얼굴 이미지를 저장할 폴더 경로
output_folder = "/home/alpaco/REAL_LAST/effnet/data/dfdc/ori_crop"
# os.makedirs(output_folder, exist_ok=True)

c_crop = 0
for ip in image_paths:
    # 이미지 불러오기
    image = cv2.imread(ip)

    # MTCNN 디텍터 초기화
    detector = MTCNN()

    # 얼굴 탐지
    faces = detector.detect_faces(image)

    # 탐지된 얼굴들을 크롭하고 저장
    
    for idx, face in enumerate(faces):
        x, y, w, h = face['box']
        
        # 얼굴 이미지를 크롭
        face_image = image[y:y+h, x:x+w]

        # 파일 이름 생성 (인덱스 추가)
        filename = f'{c_crop:05d}.jpg'

        # 파일 경로 생성 및 저장
        output_file_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_file_path, face_image)
        c_crop += 1

print("얼굴 크롭 및 저장이 완료되었습니다.")
