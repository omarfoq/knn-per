cd ../..


echo "=> generate data"

cd data/cifar10 || exit

rm -r all_data
python generate_data.py \
    --n_tasks 200 \
    --by_labels_split \
    --n_components -1 \
    --alpha 0.3 \
    --s_frac 1.0 \
    --test_tasks_frac 0.0 \
    --seed 12345

cd ../..


echo "Train base model | FedAvg"

echo "Run FedAvg lr=0.03"
python train.py \
    cifar10 \
    --n_rounds 200 \
    --bz 128 \
    --lr 0.03 \
    --lr_scheduler multi_step \
    --log_freq 10 \
    --device cuda \
    --optimizer sgd \
    --logs_root logs/cifar10/FedAvg \
    --save_path chkpts/cifar10_fedavg \
    --seed 1234  \
    --verbose 1
