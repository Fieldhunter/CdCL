# setting 1
for benchmark in Set5 Set14 B100 Urban100
do
  for sig in 1.2 2.4 3.6
  do
    CUDA_VISIBLE_DEVICES=0 python test.py --test_only \
                   --dir_data='./datasets' \
                   --n_GPUs=1 \
                   --data_test=$benchmark \
                   --model='blindsr' \
                   --scale='4' \
                   --resume=700 \
                   --blur_type='iso_gaussian' \
                   --noise=0.0 \
                   --sig=$sig
  done
done

# setting 2
#theta=(0.0 10.0 30.0 45.0 90.0 120.0 135.0 165.0 180.0)
#lambda_1=(2.0 2.0 3.5 3.5 3.5 4.0 4.0 4.0 4.0)
#lambda_2=(0.5 1.0 1.5 2.0 2.0 1.5 2.0 3.0 4.0)
#for noise in 5.0 10.0
#do
#  for i in {0..8}
#  do
#  CUDA_VISIBLE_DEVICES=0 python test.py --test_only \
#                 --dir_data='./datasets' \
#                 --n_GPUs=1 \
#                 --data_test='Set14' \
#                 --model='blindsr' \
#                 --scale='4' \
#                 --resume=700 \
#                 --blur_type='aniso_gaussian' \
#                 --noise=$noise \
#                 --theta=${theta[$i]} \
#                 --lambda_1=${lambda_1[$i]} \
#                 --lambda_2=${lambda_2[$i]}
#  done
#done

# setting 3
#for benchmark in Set14 B100 Urban100
#do
#  for jpg in 0.0 60.0
#  do
#    for nos in 0.0 20.0
#    do
#      for sig in 0.0 2.0
#      do
#        CUDA_VISIBLE_DEVICES=0 python test.py --test_only \
#                   --dir_data='./datasets' \
#                   --n_GPUs=1 \
#                   --data_test=$benchmark \
#                   --model='blindsr' \
#                   --scale='4' \
#                   --resume=700 \
#                   --blur_type='iso_gaussian' \
#                   --noise=$nos \
#                   --jpeg=$jpg \
#                   --sig=$sig
#      done
#    done
#  done
#done
