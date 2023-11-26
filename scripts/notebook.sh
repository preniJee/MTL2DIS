#!/bin/bash

cd ..
source venv/bin/activate

gpu=3
SESSION_NAME="notebook_${gpu}"

screen -dmS "$SESSION_NAME" bash -c "CUDA_VISIBLE_DEVICES=$gpu jupyter-lab --no-browser"