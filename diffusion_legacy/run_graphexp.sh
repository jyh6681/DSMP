#CUDA_VISIBLE_DEVICES=7  python3 new_graphlevel_diffusion.py arg_train --dataname QM9 -b 20000 --lr 0.005 -e 100 --if_gat False
CUDA_VISIBLE_DEVICES=7,3,0,6,5,4 python3 new_graphlevel_diffusion_noabs.py ray_train -a 4.11_listscatter_morepara_this


#CUDA_VISIBLE_DEVICES=1  python3 new_graphlevel_diffusion.py arg_train --dataname QM9 -b 20000 --lr 0.005 -e 100 --if_gat True