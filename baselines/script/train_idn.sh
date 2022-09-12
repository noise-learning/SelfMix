gpu_device=5

pretrain_model_dir=data/pre_train_models/

data=agnews
train_path=data/instance_dependent_noise/ag_news/train_a
test_path=data/ag_news/ag_news_csv/test_c.csv

# data=trec
# train_path=data/instance_dependent_noise/trec/train_n
# test_path=data/trec/test_n.csv

# data=imdb
# train_path=data/instance_dependent_noise/imdb/train
# test_path=data/IMDB/test.csv

model=elr

noise_type=idn
lr=1e-5
arr=(0.4)
arr2=(8 16 32 64 128)

for noise_ratio in ${arr[@]};do
    for seed in ${arr2[@]};do

        log_path=data/save_model/${model}/$data/${noise_ratio}${noise_type}
        mkdir -p $log_path

        CUDA_VISIBLE_DEVICES=$gpu_device python train_${model}.py \
            --pretrain_model_dir $pretrain_model_dir \
            --train_path ${train_path}${noise_ratio}.csv \
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