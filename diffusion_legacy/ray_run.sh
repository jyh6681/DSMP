#!/bin/bash

CUDA_VISIBLE_DEVICES=0,2,3,4,5 python3 ScatterMP.py
CUDA_VISIBLE_DEVICES=6,7 python3 DiffusionMP.py
CUDA_VISIBLE_DEVICES=0,2,3,4,5 python3 ScatterMP_ODE.py