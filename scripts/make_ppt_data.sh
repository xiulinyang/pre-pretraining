python -m src.grammar generate_shuff_dyck --file_dir ./data/shuff_dyck 

python -m src.utils cache_data --dataset_name data/shuff_dyck/dyck_sequences_cross_serial_64_0.5.txt --out_dir ./data/tokenized/shuff_dyck 

python -m src.utils cache_data --dataset_name wikitext --out_dir ./data/tokenized/wikitext 