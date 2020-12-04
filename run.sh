#!/bin/sh
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=stylegan2 python Model.py stylegan2-ffhq-config-f.pkl
