cd ../..



for ALPHA in 0.1 0.3 0.5 0.7 1.0
do
  echo "Experiment with alpha = ${ALPHA}"

  echo "=> generate data"

  cd data/cifar100 || exit
  rm -r all_data

  python generate_data.py \
    --n_tasks 200 \
    --pachinko_allocation_split\
    --n_components -1 \
    --alpha 0.3 \
    --beta 10 \
    --s_frac 1.0 \
    --test_tasks_frac 0.0 \
    --seed 12345

    cd ../..

    echo "=> gather scores"

    python eval_knnper.py \
      cifar100 \
      random \
      "chkpts/cifar100-fedavg-alpha-${ALPHA}.pt" \
      7 \
      --capacities_grid_resolution 0.01 \
      --weights_grid_resolution 0.01 \
      --bz 256 \
      --device cuda \
      --verbose 1 \
      --results_dir "results_base_model_effect_0.3/cifar100/n_neighbors_7_alpha_${ALPHA}" \
      --seed 12345

    echo "=> make plots"
    python make_plots.py results_/cifar100/n_neighbors_7_alpha_${ALPHA} \
      --save_path plots/cifar100/n_neighbors_7_alpha_${ALPHA}.png

    echo "-----------------------------------------"
done
