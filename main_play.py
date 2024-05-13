import pandas as pd 
import subprocess
import matplotlib.pyplot as plt
import re
import numpy as np
from pathlib import Path
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--video_path',default="/home/alpaco/REAL_LAST/fastwebapi_for_deef/static/16600_171224_2_0023/16600_171224_2_0023.mp4")
    return parser
        
    # 메인 플레이
def main(args):
    video_path = args.video_path
    
    tall_max=0
    effnet_value=0
    specRnet  =0
    
    sorce_path = "/".join(video_path.split("/")[:-1])
    
    #tall
    #tall 추론 파일 경로 파일 만들기 (강제 덮어쓰기 진행)
    with open('/home/alpaco/REAL_LAST/TALL4Deepfake/cdf_test_fold.txt', 'w') as output_file:
        output_file.write(f"{video_path} 1 400 {0}\n")
        output_file.write(f"{video_path} 1 400 {0}\n")
        output_file.write(f"{video_path} 1 400 {0}\n")
        output_file.write(f"{video_path} 1 400 {0}\n")
        output_file.write(f"{video_path} 1 400 {0}\n")
        output_file.write(f"{video_path} 1 400 {0}\n") #  6번
            
    tall_command ="/home/alpaco/REAL_LAST/TALL4Deepfake/test.sh"
    tall_result = subprocess.run(tall_command, shell=True, capture_output=True, text=True)

    pattrn_t = r'probabilities \[(.*?)\]'
    ntall = np.array(re.findall(pattrn_t, tall_result.stdout))
    tall_list = [float(a.split(', ')[1]) for a in ntall]

    plt.figure(1)
    plt.plot(tall_list, 'o-')
    plt.ylim(0, 1)
    plt.xlabel('Index')
    plt.ylabel('Probability')

    # 각 데이터 포인트에 tall_list 값을 텍스트로 표시합니다.
    for i, txt in enumerate(tall_list):
        plt.text(i, tall_list[i], f'{txt:.2f}%', ha='center', va='top', fontsize=15)

    plt.savefig(sorce_path+'/tall_graph.png')

    #effnet
    ef_grad_command ='python /home/alpaco/REAL_LAST/effnet/predict_onepick.py '+video_path
    ef_grad_result = subprocess.run(ef_grad_command, shell=True, capture_output=True, check=True, text=True)

    #specRnet
    sp_command =f'python /home/alpaco/REAL_LAST/Audio.py --d {"/".join(video_path.split("/")[:-1])}'
    sp_result = subprocess.run(sp_command, shell=True, capture_output=True, text=True)

    tall_max = max(tall_list)
    effnet_value = float(ef_grad_result.stdout.split('\n')[-2])
    specRnet = float(sp_result.stdout.split()[-4][:-1])

    predics_max = max([tall_max, effnet_value, specRnet])
    
    if tall_max>0.5 or effnet_value> 0.5 or specRnet> 0.5:
        print('Fake')
        return 'Fake'
    else:
        print('True')
        return 'True'

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
