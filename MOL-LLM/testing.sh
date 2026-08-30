GPU=0

###testing
CUDA_VISIBLE_DEVICES=$GPU python main.py --main 'test' --data_home_folder "/Users/divyam/Course/Project Arbeit/MOL-LLM/MOL-LLM/Extrap_data/" \
--svd_model_folder "FINAL_AllMix/999999_params.pth" \
--test_batch_size 50 --IC_per_eq 1 \
--dataset_workers 0 --epochs 1 --steps_per_epoch 1 --t_len 128 --seed 345 --t_end 5 --dataset both \
--board 

##Extrapolation 
# CUDA_VISIBLE_DEVICES=$GPU python main.py --main 'extrapolate' --data_home_folder /home/elisa/code/icon-gen/Extrap_data/ \
# --svd_model_folder "FINAL_AllMix/999999_params.pth" \
# --test_batch_size 10 --IC_per_eq 1 \
# --dataset_workers 1 --epochs 1 --steps_per_epoch 1 --t_len 128 --seed 345 --t_end 5

