#!/usr/bin/env bash
set -xeuo pipefail

conda install -c conda-forge python=3.9 pygame pymunk
pip install stable-baselines3 sb3-contrib imitation optuna shimmy protobuf==3.20.3

git clone https://github.com/Replicable-MARL/MARLlib.git
cd MARLlib || exit
pip install -r requirements.txt
python marllib/patch/add_patch.py -y
pip install .
pip install numpy==1.21