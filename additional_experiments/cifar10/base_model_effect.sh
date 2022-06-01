cd ../..

for ALPHA in 0.1 0.5 0.7 1.0
do

  echo "Experiment with alpha = ${ALPHA}"

  cd data/cifar10 || exit
    rm -r all_data

    python generate_data.py \
    --n_tasks 200 \
    --by_labels_split \
    --n_components -1 \
    --alpha $ALPHA \
    --s_frac 1.0 \
    --test_tasks_frac 0.0 \
    --seed 12345

  cd ../..

  echo "-----------------------------------------"

  for FILE_NAME in chkpts_list_cifar10/*.pth;
  do

    python eval_knnper.py \
      cifar10 \
      random \
      "${FILE_NAME}" \
      10 \
      --capacities_grid_resolution 1.0 \
      --weights_grid_resolution 0.1 \
      --bz 256 \
      --device cuda \
      --verbose 0 \
      --results_dir $"results_base_model_effect_alpha_${ALPHA}/cifar10/${FILE_NAME}" \
      --seed 12345

    echo "-----------------------------------------"

  done

  echo "###################################################"

done