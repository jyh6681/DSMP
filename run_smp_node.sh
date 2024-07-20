CUDA_VISIBLE_DEVICES=2,1,3,4,6,7 python3 smp_graphlevel_noabs.py ray_train -b 6 --dataname COLLAB --filename 23.10.31 
# CUDA_VISIBLE_DEVICES=2 python3 ./src/smp_graphlevel_noabs.py arg_train --filename batch32 -b 2 --dataname COLLAB
# CUDA_VISIBLE_DEVICES=5 python3 src/depth_vs_accuracy.py --lr 1e-4 --wd 1e-4 --batch_size 32 --epochs 200 --dataname PROTEINS