# TODO: setting 1
CUDA_VISIBLE_DEVICES=0 python main.py --dir_data='./datasets' \
               --n_GPUs=1 \
               --model='blindsr' \
               --scale='4' \
               --blur_type='iso_gaussian' \
               --noise=0.0 \
               --sig_min=0.2 \
               --sig_max=4.0

# TODO: setting 2
#CUDA_VISIBLE_DEVICES=0 python main.py --dir_data='./datasets' \
#               --n_GPUs=1 \
#               --model='blindsr' \
#               --scale='4' \
#               --blur_type='aniso_gaussian' \
#               --noise=25.0 \
#               --sig_min=0.2 \
#               --sig_max=4.0

# TODO: setting 3
#CUDA_VISIBLE_DEVICES=0 python main.py --dir_data='./datasets' \
#               --n_GPUs=1 \
#               --model='blindsr' \
#               --scale='4'