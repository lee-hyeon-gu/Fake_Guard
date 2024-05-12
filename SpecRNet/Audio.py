import time
import os
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchaudio
import torchaudio.sox_effects
import argparse

from pytorch_grad_cam import GradCAM, ScoreCAM, XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from lfcc import LFCC
from model.model import SpecRNet

from model import config

def slicing(waveform, samplerate):
    duration = samplerate * 4
    waveform_sqz = waveform.squeeze(0)
    waveform_len = waveform_sqz.shape[0]
    start = 0
    slice = []
    index = 1

    for i in range(duration, waveform_len+1, duration):

        if waveform_sqz[start:i].shape[0] >= duration:
            slice.append([index,start,i])
            start = i
            index += 1
        else:
            start = i
            index += 1

    return slice


def find_mp4_files(path_to_dir: Union[Path, str]) -> Optional[List[Path]]:
    """Find all mp4 files in the directory and its subtree.

    Args:
        path_to_dir: Path top directory.
    Returns:
        List containing Path objects or None (nothing found).
    """
    paths = list(sorted(Path(path_to_dir).glob("**/*.mp4")))

    return paths
import subprocess
def extract_wav(line):
    dir = os.path.dirname(line)
    id, _ = os.path.splitext(os.path.basename(line))
    output_path = f'{dir}/'
    dst = f'{dir}/{id}.wav'
    
    #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@2",dst)

    os.makedirs(output_path, exist_ok=True)

    if os.path.exists(dst):
        pass
    else:
        cmd = f'ffmpeg -i {line} -vn -acodec pcm_s16le -ac 1 -y {dst}'
        
        #subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        #print("hweiqheowdioqwn",cmd)
        os.popen(cmd)

    return dst

def resample_norm(line, sample_rate=16000):
    
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(
        line, [["rate", f"{sample_rate}"]], normalize=True)

    return waveform, sample_rate

def resample_norm_long(line, sr, sample_rate=16000):
    
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
        line, effects = [["rate", f"{sample_rate}"]], sample_rate=sr)
    
    max_val = waveform.abs().max()

    waveform = waveform / max_val

    return waveform, sample_rate

        
def trim(waveform, sample_rate, 
         SOX_SILENCE = [["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
                    ]#[침묵제거명령, 검출시작지점, 최소지속시간, 임계값DB%, 검출끝지점, 최소지속시간 , 임계값DB%]
        ):
    waveform_trim, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate, SOX_SILENCE)
    return waveform_trim, sample_rate

def pad(waveform, sample_rate):
        
        duration = 64600
        waveform_sqz = waveform.squeeze(0)
        waveform_len = waveform_sqz.shape[0]

        if waveform_len > duration:
            return waveform_sqz[:duration], sample_rate
        elif waveform_len < duration:
            num_repeats = int(duration / waveform_len) + 1
            padded_wavefrom = torch.tile(waveform_sqz, (1, num_repeats))[:, :duration][0]
            return padded_wavefrom, sample_rate

def lfcc_gen(input, sample_rate):
    lfcc = LFCC(sample_rate=sample_rate, n_lfcc=80, 
                speckwargs={'n_fft' : 512,
                            'hop_length' : 160, 
                            'win_length': 400,
                            }
                )
    lfcc_feature = lfcc(input)
    return lfcc_feature

def XAI_GAN(input_G,GAN):

    layer_G = GAN.block0

    xgrad_G = XGradCAM(model=GAN, target_layers=[layer_G])

    xgrad_G.batch_size = 1

    xgrad_cam_G = xgrad_G(input_tensor=input_G.unsqueeze(0), targets=[ClassifierOutputTarget(0)])
    xgrad_cam_G = xgrad_cam_G[0, :]

    return xgrad_cam_G

def XAI_TTS(input_T,TTS):

    layer_T = TTS.block2

    xgrad_T = XGradCAM(model=TTS, target_layers=[layer_T])

    xgrad_T.batch_size = 1

    xgrad_cam_T = xgrad_T(input_tensor=input_T.unsqueeze(0), targets=[ClassifierOutputTarget(0)])
    xgrad_cam_T = xgrad_cam_T[0, :]

    return xgrad_cam_T


def save_plot(lfcc, cam, output, dpi=600):
    #matplotlib.use('Agg')
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    im = ax.imshow(lfcc.squeeze(), aspect='auto', origin='lower', vmin=-35, vmax=45, cmap=sns.diverging_palette(220, 20, l=40, s=120, as_cmap=True))
    plt.colorbar(im, format='%+0.4f')
    
    plt.title(output)
    plt.xlabel('Time')
    plt.ylabel('LFCC')
    cam = np.where( cam > 0, cam, np.nan)
    plt.imshow(cam.squeeze(), aspect='auto', origin='lower', alpha=1, vmin= -0.1, vmax= 1, cmap='inferno')
    
    plt.tight_layout()
    plt.show()

    png_name = f'Audio_xai.png'
    
    plt.savefig(root+"/"+png_name, dpi=dpi, bbox_inches='tight')

    return png_name

def save_LFCC(line):

    dir = os.path.dirname(line)
    id, _ = os.path.splitext(os.path.basename(line))
    output_path = f'{dir}'
    dst = f'{output_path}/{id}.pt'
    
    wav_path = extract_wav(line)

    waveform, sample_rate = resample_norm(wav_path, sample_rate=16000)
    
    trim_waveform, sample_rate = trim(waveform, sample_rate)

    padded_waveform, sample_rate = pad(trim_waveform, sample_rate)

    features = lfcc_gen(padded_waveform.unsqueeze(0), sample_rate)

    dir = os.path.dirname(line)
    id, _ = os.path.splitext(os.path.basename(line))
    
    torch.save(features, dst)
    return features, id

def save_LFCC_long(input, sr, idx):

    dir = os.path.dirname(line)
    id, _ = os.path.splitext(os.path.basename(line))
    output_path = f'{dir}'
    dst = f'{output_path}/{id}_{idx}.pt'

    waveform, sample_rate = resample_norm_long(input, sr, sample_rate=16000)
    
    trim_waveform, sample_rate = trim(waveform, sample_rate)
    padded_waveform, sample_rate = pad(trim_waveform, sample_rate)

    features = lfcc_gen(padded_waveform.unsqueeze(0), sample_rate)
    
    torch.save(features, dst)

    return features, id

def LFCC_pred(input):

    GAN_ckpt = torch.load('./GAN_model.pth')
    TTS_ckpt = torch.load('./TTS_model.pth')
    
    specrnet_config = config.get_specrnet_config(input_channels=1)

    TTS = SpecRNet(specrnet_config, device='cpu')
    TTS.load_state_dict(TTS_ckpt)
    specrnet_config = config.get_specrnet_config(input_channels=1)

    specrnet_config = config.get_specrnet_config(input_channels=1)
    GAN = SpecRNet(specrnet_config, device='cpu')
    GAN.load_state_dict(GAN_ckpt)

    GAN.eval()
    TTS.eval()

    gan_out = GAN(input.unsqueeze(0))
    tts_out = TTS(input.unsqueeze(0))

    gan_sig = torch.sigmoid(gan_out)
    tts_sig = torch.sigmoid(tts_out)

    gan_dct = (gan_sig > 0.5).long()
    tts_dct = (tts_sig > 0.5).long()

    return gan_sig.item(), tts_sig.item(), gan_dct.item(), tts_dct.item(), GAN, TTS




if __name__ == "__main__":
    # 인자값을 받을 수 있는 인스턴스 생성
    
    parser = argparse.ArgumentParser(description='Sound fake detection')

    # 입력받을 인자값 등록
    parser.add_argument('--d', required=True, help='mp4 dir')

    # 입력받은 인자값을 args에 저장 (type: namespace)
    args = parser.parse_args()

    # 입력받은 인자값 출력
    print(args.d)

    root = args.d

    lines = find_mp4_files(root)

    line = lines[0]

    print(line)

    wav = extract_wav(line)
    time.sleep(10)
    dir = os.path.dirname(line)
    id, _ = os.path.splitext(os.path.basename(line))

    waveform, samplerate = torchaudio.load(wav)
    sr = 16000

    duration = samplerate * 4

    waveform_sqz = waveform.squeeze(0)
    waveform_len = waveform_sqz.shape[0]

    if waveform_len < (duration * 2.5):

        input, id = save_LFCC(line)

        gan_sig, tts_sig, gan_dct, tts_dct, GAN, TTS = LFCC_pred(input)

        gan_percentage = gan_sig * 100 
        tts_percentage = tts_sig * 100


        if gan_dct == 0:
            gan_out = f'GAN {gan_percentage:.2f}%'
            print(f'{gan_out}')
            xgrad_cam = XAI_GAN(input, GAN)
            img = save_plot(input, xgrad_cam, gan_out)
        else:
            gan_out = f'REAL {gan_percentage:.2f}%'
            print(f'{gan_out}')
            xgrad_cam = XAI_GAN(input, GAN)
            img = save_plot(input, xgrad_cam, gan_out)
        if tts_dct == 0:
            tts_out = f'TTS {tts_percentage:.2f}'
            print(f'{tts_out}')
            xgrad_cam = XAI_TTS(input, TTS)
            img = save_plot(input, xgrad_cam, gan_out)
        else:
            tts_out = f'REAL {tts_percentage:.2f}'
            print(f'{tts_out}')
            xgrad_cam_T = XAI_TTS(input, TTS)
            img = save_plot(input, xgrad_cam,gan_out)

        print(f'Save Plot!, {img}')
    else:

        slice = slicing(waveform, sr)
        score = []

        high_r = 0.0
        high_g = 0.0
        high_t = 0.0

        list_r = []
        list_g = []
        list_t = []

        for idx, start, end  in slice:
            input = save_LFCC_long(waveform_sqz[start:end].unsqueeze(0), sr, idx)
            input = input[0]

            gan_sig, tts_sig, gan_dct, tts_dct, GAN, TTS = LFCC_pred(input)

            if gan_dct == 0:
                high_g += gan_sig
                list_g.append(gan_sig)
            else:
                high_r += gan_sig
                list_r.append(gan_sig)
            if tts_dct == 0:
                high_t += tts_sig
                list_t.append(tts_sig)
            else:
                high_r += tts_sig
                list_r.append(tts_sig)

        score.append(high_r / len(list_r))
        score.append(high_g / len(list_g))
        score.append(high_t / len(list_t))
    
        high_score = max(score)
        result = score.index(high_score)

        list_r = sorted(list_r, reverse=True)
        list_g = sorted(list_g, reverse=True)
        list_t = sorted(list_t, reverse=True)

        if result == 0 :
            percentage = list_r[0] * 100
            p_out = f'REAL : {percentage:.2f}%'
            xgrad_cam_G = XAI_GAN(input, GAN)
            img = save_plot(input, xgrad_cam_G, p_out)
            print(p_out)
            print(f'Save Plot!, {img}')

        elif result == 1:
            percentage = list_g[0] * 100
            p_out = f'GAN : {percentage:.2f}%'
            xgrad_cam_G = XAI_GAN(input, GAN)
            g_png = save_plot(input, xgrad_cam_G, p_out)
            print(p_out)
            print(f'Save Plot!, {g_png}')

        elif result == 2:
            percentage = list_t[0] * 100
            p_out = f'TTS : {percentage:.2f}%'
            xgrad_cam_T = XAI_TTS(input, TTS)
            img = save_plot(input, xgrad_cam_T, p_out)
            print(p_out)
            print(f'Save Plot!, {img}')