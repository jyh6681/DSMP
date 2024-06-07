CUDA_VISIBLE_DEVICES=5 python new_graphlevel_diffusion.py --filename batch32 -b 32
#CUDA_VISIBLE_DEVICES=5 python new_graphlevel_diffusion.py --filename batch32 -b 32 --if_linear False
CUDA_VISIBLE_DEVICES=5 python new_graphlevel_diffusion.py --filename batch32 -b 32 --if_gat False

CUDA_VISIBLE_DEVICES=4 python new_graphlevel_diffusion.py --filename batch32 -b 32 --data PROTEINS
#CUDA_VISIBLE_DEVICES=5 python new_graphlevel_diffusion.py --filename batch32 -b 32 --if_linear False --data PROTEINS
CUDA_VISIBLE_DEVICES=4 python new_graphlevel_diffusion.py --filename batch32_nogat -b 32 --if_gat False --data PROTEINS