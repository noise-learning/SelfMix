
results_dir=./save_model/
mkdir -p $results_dir

CUDA_VISIBLE_DEVICES=0 python main.py \
    --results_dir $results_dir \
    --train_path ./data/train_idn.csv \
    --test_path ./data/test.csv \
    --seed 8 \
    --noise_ratio 0.4 \
    --noise_type idn  \
    --warmup_epoch 1 \
    --wamup_samples 5000 \
    --lambda_r 0.3 \
    --lambda_p 0.0 \
    --class_regularization > $results_dir/train.log
