# for this script, you should see separation on wikitext between pre-pretraining with shuff_dyck and vanilla pretraining within 3k steps or so.

# pre-pretrain
python train.py --model_name EleutherAI/pythia-160m --data_dir ./data/tokenized/shuff_dyck --output_dir ./output/shuff_dyck/pythia-160m --save_steps 500 --max_steps 500 --reinit True

# pretrain
# an epoch of wikitext is roughly 1k steps
# python train.py --model_name EleutherAI/pythia-160m --data_dir ./data/tokenized/wikitext --output_dir ./output/wikitext/pythia-160m --save_steps 2000 --max_steps 10000 --reinit True
python train.py --model_name EleutherAI/pythia-160m --data_dir ./data/tokenized/c4 --output_dir ./output/c4/pythia-160m --save_steps 2000 --max_steps 10000 --reinit True

# pretrain w shuff_dyck
# python train.py --model_name ./output/shuff_dyck/pythia-160m/checkpoint-500 --data_dir ./data/tokenized/wikitext --output_dir ./output/wikitext/pythia-160m/shuff_dyck --save_steps 2000 --max_steps 10000 --reinit False
python train.py --model_name ./output/shuff_dyck/c4/checkpoint-500 --data_dir ./data/tokenized/c4 --output_dir ./output/c4/pythia-160m/shuff_dyck --save_steps 2000 --max_steps 10000 --reinit False