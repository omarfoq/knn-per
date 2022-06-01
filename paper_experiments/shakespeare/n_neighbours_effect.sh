cd ../..

CHKPTS_PATH="chkpts/shakespeare.pt"

echo "=> generate data"

cd data/cifar100 || exit

rm -r all_data

python generate_data.py \
    --s_frac 1.0 \
    --tr_frac 0.8 \
    --seed 12345

cd ../..

for k in 1 3 5 7 10
do

  echo "Experiment with k = $k"

  python eval_knnper.py \
    shakespeare \
    random \
    $CHKPTS_PATH \
    $k \
    --capacities_grid_resolution 0.01 \
    --weights_grid_resolution 0.01 \
    --bz 256 \
    --device cuda \
    --verbose 1 \
    --results_dir "results/shakespeare/n_neighbors_$k" \
    --seed 12345

  echo "-----------------------------------------"

done
