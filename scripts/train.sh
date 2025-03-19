# Note: This script pretrains on wikitext for easy downloading.
# To reproduce paper results, download and tokenize c4.

# pre-pretrain
python train.py --model_name EleutherAI/pythia-160m --data_dir ./data/tokenized/shuff_dyck --output_dir ./output/shuff_dyck/pythia-160m --save_steps 500 --max_steps 500 --reinit True

# pretrain
# an epoch of wikitext is roughly 1k steps
python train.py --model_name EleutherAI/pythia-160m --data_dir ./data/tokenized/wikitext --output_dir ./output/wikitext/pythia-160m --save_steps 3000 --max_steps 3000 --reinit True

# pretrain w shuff_dyck
python train.py --model_name ./output/shuff_dyck/pythia-160m/checkpoint-500 --data_dir ./data/tokenized/wikitext --output_dir ./output/wikitext/pythia-160m/shuff_dyck --save_steps 3000 --max_steps 3000 --reinit False 