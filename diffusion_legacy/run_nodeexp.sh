#CUDA_VISIBLE_DEVICES=7  python3 new_graphlevel_diffusion.py arg_train --dataname QM9 -b 20000 --lr 0.005 -e 100 --if_gat False
CUDA_VISIBLE_DEVICES=7,3 python3 Node_DiffusionMP_noabs.py ray_train


#CUDA_VISIBLE_DEVICES=1  python3 new_graphlevel_diffusion.py arg_train --dataname QM9 -b 20000 --lr 0.005 -e 100 --if_gat True