#!/bin/bash

# source only
python noun.py --method Srconly --dset office-home --s 0 --t 1 --gpu_id 0 --bs 36 --seed 0 --output_dir results/Srconly/seed0/ --num_iterations 10000

# dann
python noun.py --method DANN --dset office-home --s 0 --t 1 --gpu_id 0 --bs 36 --seed 0 --output_dir results/DANN/seed0/ --num_iterations 10000

# dann-[f, p]
python noun.py --method NOUN --dset office-home --cond_feat p --s 0 --t 1 --gpu_id 0 --bs 36 --seed 0 --output_dir results/DANN_fp/seed0/ --num_iterations 10000

# noun
python noun.py --method NOUN --dset office-home --cond_feat p_norm --s 0 --t 1 --gpu_id 0 --bs 36 --seed 0 --output_dir results/NOUN/seed0/ --num_iterations 10000

# noun+e
python noun.py --method NOUN --dset office-home --cond_feat p_norm --e True --s 0 --t 1 --gpu_id 0 --bs 36 --seed 0 --output_dir results/NOUN_E/seed0/ --num_iterations 10000

# pronoun
python noun.py --method NOUN --dset office-home --cond_feat ema_ctr_norm --s 0 --t 1 --gpu_id 0 --norm_factor 3 --bs 36 --seed 0 --output_dir results/PRONOUN/seed0/ --num_iterations 10000
