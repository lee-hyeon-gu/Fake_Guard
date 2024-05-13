from pytorch_grad_cam import GradCAM, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2
import torch
from torch import nn
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import timm
import os, sys
import torch.optim as optim
from mtcnn import MTCNN

import argparse
from pathlib import Path


detector = MTCNN()

def preprocess_image(
    img: np.ndarray, mean=[
        0.5, 0.5, 0.5], std=[
            0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)

class EffNet(nn.Module):
    def __init__(self, backbone, n_out, is_sigmoid):
        super(EffNet, self).__init__()
        self.model = timm.create_model(model_name=backbone, pretrained=True)
        self.model.classifier = nn.LazyLinear(n_out)
        self.is_sigmoid = is_sigmoid

    def forward(self, x):
        x = self.model(x)
        if self.is_sigmoid:
            x = nn.Sigmoid()(x)
        return x
    
model = EffNet(backbone='efficientnet_b0', n_out=1, is_sigmoid=True)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
checkpoint = torch.load("/home/alpaco/REAL_LAST/effnet/results/train/20240428_182740/model.pt")
model.load_state_dict(checkpoint['model'])
model.eval()

layer = model.model.conv_head

if sys.argv[1][-3:]=='mp4':
    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("비디오를 열 수 없습니다.")
        exit()

    # 첫 번째 프레임 읽기
    ret, frame = cap.read()

    # print(frame.shape)

    cv2.imwrite('/'.join(sys.argv[1].split('/')[:-1])+'/thumbnail.jpg',frame)
    # 비디오 캡처 객체 해제
    cap.release()
    faces = detector.detect_faces(frame)

else:
    faces = detector.detect_faces(sys.argv[1])

# print(len(faces), len(faces[0]), len(faces[0][0]), len(faces[0][0][0]))

for idx, face in enumerate(faces):

    x, y, w, h = face['box']
    margin_pct = 1.2
    margin = int(max(w, h) * margin_pct)
    cx = x + w // 2
    cy = y + h // 2
    x1 = max(0, cx - margin // 2)
    y1 = max(0, cy - margin // 2)
    x2 = min(frame.shape[1], cx + margin // 2)
    y2 = min(frame.shape[0], cy + margin // 2)
    
    face_image = frame[y1:y2, x1:x2]

rgb_img = cv2.resize(face_image, (342, 342))
rgb_img = np.float32(rgb_img) / 255

input_tensor = preprocess_image(rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

# 모델 예측 수행
with torch.no_grad():
    output = model(input_tensor)
    output.item()

targets = [ClassifierOutputTarget(0)]

# cam 생성 및 처리
cam = GradCAM(model=model, target_layers=[layer])
cam.batch_size = 1
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
# display(Image.fromarray(visualization))
bgr_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)


# 이미지 크기
height, width = bgr_image.shape[:2]

# 하단에 추가할 여분의 공간 크기 설정
extra_space = 50

# 하단에 하얀색 공간 추가
new_height = height + extra_space
new_image = cv2.copyMakeBorder(bgr_image, 0, extra_space, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

print(output.item())
# 텍스트 추가할 위치 계산
text = str(output.item()*100)[:5]+'%'
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
text_x = (width - text_size[0]) // 2
text_y = height + (extra_space + text_size[1]) // 2

# 이미지에 텍스트 추가
cv2.putText(new_image, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

# 이미지 저장
cv2.imwrite('/'.join(sys.argv[1].split('/')[:-1])+'/image_xai.jpg', new_image)
