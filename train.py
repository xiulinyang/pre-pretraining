from typing import List

import datasets
import fire
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


class SaveAtStepsCallback(TrainerCallback):
    """Custom callback to save model at specific training steps."""

    def __init__(self, save_steps: List[int], output_dir: str):
        """
        Args:
            save_steps: List of steps at which to save the model
            output_dir: Base directory for saving checkpoints
        """
        self.save_steps = sorted(save_steps)  # Sort steps in ascending order
        self.output_dir = output_dir

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each step."""
        if state.global_step in self.save_steps:
            # Create a subdirectory for this specific step
            checkpoint_dir = f"{self.output_dir}/checkpoint-{state.global_step}"
            kwargs["model"].save_pretrained(checkpoint_dir)

            # If you're using a tokenizer, you might want to save it too
            if "tokenizer" in kwargs:
                kwargs["tokenizer"].save_pretrained(checkpoint_dir)

            print(f"Saved model at step {state.global_step}")


def load_c4_dataset(start_file=0, end_file=10):
    """Loads a portion of the C4 dataset.

    Args:
        start_file: The starting file number (inclusive).
        end_file: The ending file number (exclusive).

    Returns:
        A Hugging Face Dataset object.
    """

    data_files = []
    for i in range(start_file, end_file):
        # Format the file number with leading zeros
        file_number = str(i).zfill(5)  # Pad with zeros to 4 digits
        data_files.append(
            f"/vast/work/public/ml-datasets/c4/en/c4-train.{file_number}-of-01024.json"
        )

    dataset = datasets.load_dataset("json", data_files=data_files)
    return dataset


def main(
    data_dir="./data/tokenized/depth9_train",
    model_name="EleutherAI/pythia-160m",
    reinit=False,
    max_seq_length=2048,
    gradient_accumulation_steps=1,
    max_steps=10000,
    bsz=32,
    warmup_steps=500,
    logging_steps=5,
    save_steps=250,
    output_dir="output",
    seed=3407,
    report_to="wandb",
    lr=1e-3,
    min_lr_rate=0.1,
    override_packing=False,
    use_callback=False,
):
    print(locals())
    set_seed(seed)

    if "pythia-1b" in model_name:
        c4_max = 11  # longer training --> more data
    else:
        c4_max = 1

    callback = SaveAtStepsCallback(
        save_steps=list(range(0, 4000, 100)) + list(range(4000, 10000, 1000)),
        output_dir=output_dir,
    )

    if data_dir == "c4":
        # if you happen to have C4 locally in json format
        dataset = load_c4_dataset(0, c4_max)
    else:
        # otherwise, pretokenize
        dataset = datasets.load_from_disk(data_dir)

    if "train" in dataset:
        dataset = dataset["train"]

    if reinit:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    packing = "c4" in data_dir

    training_args = SFTConfig(
        per_device_train_batch_size=bsz,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        output_dir=output_dir,
        seed=seed,
        report_to=report_to,
        learning_rate=lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": min_lr_rate},
        packing=packing if not override_packing else False,
        max_length=max_seq_length,
        bf16=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    if use_callback:
        trainer.add_callback(callback)

    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
