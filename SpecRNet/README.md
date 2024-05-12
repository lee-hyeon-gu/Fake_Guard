# Audio Part
WaveFake 논문을 기반으로 하여 KSS데이터로 사전학습된 Parallel WaveGAN 모델을 이용해
Fake 데이터를 GAN, TTS로 생성하였으며 이를 1:1로 학습함

SpecRNet 에서 사용한 방식을 대부분 채용, 16Khz 리샘플링, 4초 슬라이싱 or 패딩, SOX를 사용한 침묵 제거 후

LFCC를 생성하여 학습함

## PS
1 : 2 학습 또는 KSS, JSUT, LJSpeech 각각의 원본과 생성 데이터를 모두 학습시킨 사전훈련 모델을 생성하였으나 WaveFake에서 각 생성모델에 서로 다른 결과를 보인 것처럼
개별 훈련 모델이 대부분 더 뛰어난 결과를 보임

# Train
Model 폴더 내의 train_model.py 는 WaveFake 모델의 Train 과 trainer를 모두 사용하도록 변형됨

# Pre-trained Model

- 원본 : GAN생성음성 학습 GAN_model.pth

- 원본 : TTS생성음성 학습 TTS_model.pth

# Evaluation

mp4 파일이 위치한 폴더를 지정, 폴더 내 모든 .mp4 파일을 List하여 영상의 음성을 추출하고
학습된 방식과 같게 전처리 후 모델의 출력을 비교하여
모델의 출력 중 가장 유력한 Label(REAL, GAN, TTS) 과 확률을 Sigmoid를 사용하여 나타내며
Xgrad-CAM을 사용한 XAI와 함께 Plot에 표시됨

```bash
python Audio.py --d mp4 dir
```

# Reference

## WaveFake
Github : https://github.com/RUB-SysSec/WaveFake/tree/main
arxiv : https://arxiv.org/abs/2111.02813

## Parallel WaveGAN
Github : https://github.com/kan-bayashi/ParallelWaveGAN
arxiv : https://arxiv.org/abs/1910.11480

## SpecRNet
Github : https://github.com/piotrkawa/specrnet
arxiv : https://arxiv.org/abs/2210.06105

## KSS (Korean Single Speaker Speech) Dataset
https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset