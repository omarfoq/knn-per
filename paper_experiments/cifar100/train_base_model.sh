cd ../..

echo "=> generate data"

cd data/cifar100 || exit

rm -r all_data
python generate_data.py \
    --n_tasks 200 \
    --pachinko_allocation_split \
    --n_components -1 \
    --alpha 0.3 \
    --beta 10 \
    --s_frac 1.0 \
    --test_tasks_frac 0.0 \
    --seed 12345

cd ../..


echo "Train base model | FedAvg"

echo "Run FedAvg lr=0.07"
python train.py \
    cifar100 \
    --aggregator_type centralized \
    --client_type normal \
    --sampling_rate 1. \
    --local_steps 1 \
    --n_rounds 200 \
    --bz 256 \
    --optimizer sgd \
    --lr_scheduler constant \
    --lr 0.07 \
    --log_freq 10 \
    --device cuda \
    --logs_dir logs_09_11_2021/cifar100/FedAvg \
    --chkpts_dir chkpts_09_11_2021/cifar100_fedavg\
    --seed 1234  \
    --verbose 1
