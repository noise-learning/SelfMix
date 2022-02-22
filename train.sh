
results_dir=./save_model/trec/0.4asym
mkdir -p $results_dir

CUDA_VISIBLE_DEVICES=0 python main.py \
    --results_dir $results_dir \
    --train_path ./data/trec/train.csv \
    --test_path ./data/trec/test.csv \
    --seed 8 \
    --noise_ratio 0.4 \
    --noise_type asym  \
    --lambda_r 0.3 \
    --lambda_p 0.2 > $results_dir/train.log
