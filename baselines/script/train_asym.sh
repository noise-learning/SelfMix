gpu_device=2

pretrain_model_dir=data/pre_train_models/

# data=trec
# train_path=data/trec/train_n.csv
# test_path=data/trec/test_n.csv

# data=imdb45k
# train_path=data/IMDB/train.csv
# test_path=data/IMDB/test.csv

data=agnews
train_path=data/ag_news/ag_news_csv/train_a.csv
test_path=data/ag_news/ag_news_csv/test_c.csv

model=cl

noise_type=asym
lr=1e-5
arr=(0.4 0.3 0.2 0.1)
arr2=(8 16 32 64 128)

for noise_ratio in ${arr[@]};do
    for seed in ${arr2[@]};do

        log_path=data/save_model/$data/${noise_ratio}${noise_type}
        mkdir -p $log_path

        CUDA_VISIBLE_DEVICES=$gpu_device python train_${model}.py \
            --pretrain_model_dir $pretrain_model_dir --bert_type bert-base-uncased \
            --train_path $train_path \
            --test_path $test_path \
            --fix_data 0 \
            --seed $seed \
            --noise_ratio $noise_ratio \
            --noise_type $noise_type \
            --learning_rate $lr \
            --epoch 6 \
            > $log_path/${seed}.log
    done
done