cd ../..


echo "=> generate data"

cd data/shakespeare || exit

rm -r all_data
python generate_data.py \
    --s_frac 1.0 \
    --tr_frac 0.8 \
    --seed 12345

cd ../..


echo "Train base model | FedAvg"

echo "Run FedAvg lr=0.03"
python train.py \
    shakespeare \
    FedAvg \
    --n_rounds 200 \
    --bz 128 \
    --lr 0.1 \
    --lr_scheduler constant \
    --log_freq 10 \
    --device cuda \
    --optimizer sgd \
    --logs_root logs/shakespeare/FedAvg \
    --save_path chkpts/shakespeare_fedavg \
    --seed 1234  \
    --verbose 1
