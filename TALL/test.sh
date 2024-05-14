CUDA_VISIBLE_DEVICES=1 python /home/alpaco/REAL_LAST/TALL4Deepfake/test.py  --dataset ffpp \
 --input_size 112 --opt adamw --lr 1e-4 --epochs 30 --sched cosine --thumbnail_rows 2 --disable_scaleup \
 --pretrained --warmup-epochs 5 --model TALL_SWIN --use_pyav pyav \
 --hpe_to_token --initial_checkpoint /home/alpaco/REAL_LAST/TALL4Deepfake/my_models/model_best.pth --eval --num_crops 5 --num_clips 8 \
 2>&1 | tee ./output/test_ffpp_`date +'%m_%d-%H_%M'`.log