the-well-download --base-path "/Users/divyam/Course/Project Arbeit" --dataset active_matter

cd the_well/benchmark
python3 train.py experiment=unet_classic server=local data=turbulent_radiative_layer_2D
python3 train_multi_pde.py experiment=unet_classic_conditioned server=local data=turbulent_radiative_layer_2D

epoch: 1
auto_resume: true/false