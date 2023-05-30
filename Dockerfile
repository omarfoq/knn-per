# syntax = docker/dockerfile:experimental
FROM continuumio/miniconda3
SHELL ["/bin/bash","-l", "-c"]

WORKDIR /root/
#RUN apt update && apt install -y libtorch-dev
# Install Miniconda
RUN /opt/conda/bin/conda init bash && \
    /opt/conda/bin/conda config --add channels pytorch && \ 
    /opt/conda/bin/conda install -c pytorch faiss-cpu  numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

# =================================

# Add conda bin to path
ENV PATH /opt/conda/bin:$PATH

SHELL ["conda", "run", "/bin/bash", "-c"]
WORKDIR /app
COPY . /app
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

WORKDIR /app/data/femnist
RUN mkdir -p intermediate/data_as_tensor_by_writer
RUN ls
RUN python generate_data.py --s_frac 0.2 --tr_frac 0.8 --seed 12345

WORKDIR /app/data/shakespeare
RUN mkdir -p raw_data/by_play_and_character
RUN python generate_data.py  --s_frac 0.2 --tr_frac 0.8 --seed 12345

WORKDIR /app
RUN ln -s /opt/conda/lib/libmkl_intel_lp64.so.2 /opt/conda/lib/libmkl_intel_lp64.so.1 && ln -s /opt/conda/lib/libmkl_gnu_thread.so.2 /opt/conda/lib/libmkl_gnu_thread.so.1 && ln -s /opt/conda/lib/libmkl_core.so.2 /opt/conda/lib/libmkl_core.so.1
RUN python train.py femnist --aggregator_type centralized --n_rounds 100 --bz 128 --lr 0.05 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --logs_dir ./logs --chkpts_dir ./chkpts/cifar10_fedavg --verbose 1
CMD [python, eval_knnper.py, femnist, random, chkpts, 20, --capacities_grid_resolution, 0.01, --weights_grid_resolution, 0.01, --bz, 256, --device, cuda, --verbose, 1, --result_dir, results/, --seed, 12345]

