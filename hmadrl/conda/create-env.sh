#!/usr/bin/env bash
set -xeuo pipefail

conda_exec="micromamba"

$conda_exec install -c conda-forge python=3.9 pygame pymunk
pip install -U setuptools==65.5.0 pip==21
pip install stable-baselines3 sb3-contrib imitation optuna shimmy minari protobuf==3.20.3

git clone https://github.com/Replicable-MARL/MARLlib.git
cd MARLlib || exit
pip install -r requirements.txt
python marllib/patch/add_patch.py -y
pip install .
pip install numpy==1.21
pip install -U pettingzoo supersuit
# pip install --force-reinstall torch==1.9.0 --index-url https://download.pytorch.org/whl/cpu
