# TRAIN MODEL

# input device id
device_id=$1
minp=$2

CUDA_VISIBLE_DEVICES=$device_id python main.py \
    --image_path /data1/yifan/ecg/data/ukb/ecg_npy \
    --k_fold 2 \
    --neptune 1 \
    --batch_size 320 \
    --lr 0.00005 \
    --seed 1 \
    --max_epochs 100 \
    --w2 1.0 \
    --w3 0.5 \
    --w4 1.0 \
    --minp $minp \
    --temperature 0.3 \
