# for more info, see https://arxiv.org/abs/2403.03942

# 1) save average activation
python save_avg_activations.py save_avg_activation --initialize_from /path/to/model --data_dir ./data/tokenized/task

# 2) prune
for target_sparsity in 0.3 0.4 0.5 0.6 0.7; do 
    python prune.py --target_sparsity $target_sparsity --max_steps 5000 --sparsity_warmup_steps 1000 --model_name /path/to/model/ --data_dir ./data/tokenized/task --output_dir /path/to/output --bsz 4 --save_steps 5000
done

# stack together masks
python aggregate_pruning.py main --super_dir /path/to/output