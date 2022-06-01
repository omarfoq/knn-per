cd ../..


echo "=> generate data"

cd data/femnist || exit

rm -r all_data
python generate_data.py \
    --s_frac 0.1 \
    --tr_frac 0.8 \
    --train_tasks_frac 0.7 \
    --seed 12345


cd ../..


echo "Train base model | FedAvg"

echo "Run FedAvg lr=0.03"
python train.py \
    femnist \
    --n_rounds 200 \
    --sampling_rate 1.0 \
    --bz 128 \
    --lr 0.03 \
    --local_steps 1 \
    --lr_scheduler multi_step \
    --log_freq 15 \
    --device cuda \
    --optimizer sgd \
    --logs_dir logs/femnist/FedAvg \
    --chkpts_dir chkpts/femnist_fedavg \
    --seed 1234  \
    --verbose 1
