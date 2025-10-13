# Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases

Increase pre-training token efficiency by pre-pretraining on formal languages first!

![Loss Curves](assets/loss_curves.png)


<a href="https://arxiv.org/abs/2502.19249" style="font-size: 20px;">arXiv</a>

<a href="https://wandb.ai/myhu/pre-pretraining" style="font-size: 20px;">wandb logs</a>

Trained Pythia 160M Models:
- [Trained on Shuffle-Dyck only, 500 steps](https://huggingface.co/michahu8/pythia-160m-shuffle-dyck-500steps)
- [C4, no pre-pretraining](https://huggingface.co/michahu8/pythia-160m-c4-only-10k)
- [500 steps Shuffle-Dyck --> C4](https://huggingface.co/michahu8/pythia-160m-sd-500-c4-10k)


## Changelog

**Oct 2025**: 
- Added `uv` and explicit download support for c4 dataset, removed `unsloth` as a dependency.
- Released models, checkpoints, and wandb logs for c4 training. (This is not a strict reproduction of the paper results--just a sample run for folks who've been asking for model checkpoints 😊)


## Installing dependencies
```bash
git clone https://github.com/michahu/pre-pretraining.git
cd pre-pretraining/
pip install -r requirements.txt
```

## Usage
For a minimal reproduction:
```bash
bash scripts/make_ppt_data.sh
bash scripts/train.sh
```

For this minimal reproduction on wikitext, you should see the following:
1. The pre-pretrained model loss starts off higher, then catches up to no pre-pretraining by around 3k steps.
2. After 3k steps, pre-pretraining is better.
   
To reproduce our full experiments, change `wikitext` to `allenai/c4` in the scripts above (download time will be slightly longer).

This is how the repo is organized:
- Formal language pre-pretraining data generation: `src/grammar.py`
- Training: `train.py`
- Pruning: `modeling_ppt_neox.py` and `prune.py`
- Evals: `eval_checkpoint.py`
- Ablations data generation: `src/utils.py`


## Citation
```
@inproceedings{hu-etal-2025-circuits,
    title = "Between Circuits and {C}homsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases",
    author = "Hu, Michael Y.  and
      Petty, Jackson  and
      Shi, Chuan  and
      Merrill, William  and
      Linzen, Tal",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.478/",
    doi = "10.18653/v1/2025.acl-long.478",
    pages = "9691--9709",
    ISBN = "979-8-89176-251-0",
}
```

## Contact
Feel free to create an issue or email Michael Hu (<myh2014@nyu.edu>)
