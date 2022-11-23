import os
import logging, argparse
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader

from accelerate import Accelerator
from arguments import EvaluationArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed


class ConstantLengthDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        dataset,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    break
                try:
                    buffer.append(next(iterator)["content"])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    more_examples = False
                    break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids)


def create_dataloader(dataset_name, args, tokenizer):
    ds_kwargs = {"streaming": True}
    if ".jsonl" in dataset_name:
        valid_data = load_dataset(
            "json", data_files=dataset_name, split="train", **ds_kwargs
        )
    else:
        valid_data = load_dataset(dataset_name, split="train", **ds_kwargs)

    valid_dataset = ConstantLengthDataset(
        tokenizer, valid_data, seq_length=args.seq_length
    )
    eval_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)
    return eval_dataloader


def evaluate(args, model, eval_dataloader, accelerator):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(args.batch_size)
        losses.append(accelerator.gather(loss))

    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()


def evaluate_all():
    # Setup Accelerator
    accelerator = Accelerator()

    # Parse configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--load_model_dir", type=str, required=True)
    parser.add_argument("--load_data_dir", type=str, required=True)

    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--start_ckpt", type=str, default="gpt2-medium")
    parser.add_argument("--tokenizer_ckpt", type=str, default="codeparrot/codeparrot")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=3)
    args = parser.parse_args()
    set_seed(args.seed)

    model_paths = {
        task_name: os.path.join(args.load_model_dir, f"last_{task_name}")
        for task_name in args.tasks
    }
    model_paths["zeroshot"] = args.start_ckpt

    data_paths = {
        task_name: os.path.join(args.load_data_dir, f"{task_name}_val.jsonl")
        for task_name in args.tasks
    }
    print(model_paths)
    print(data_paths)

    f = open(
        f'{os.path.join(args.load_model_dir, "eval_results.txt")}', "w", buffering=1
    )
    f.write(f"TrainTask\tEvalTast\tEvalLoss\tEvalPerplexity\n")

    # Load and prepare tokenizer and all dataloders
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_ckpt)
    eval_dataloaders = {
        task_name: accelerator.prepare(create_dataloader(dataset_name, args, tokenizer))
        for task_name, dataset_name in data_paths.items()
    }

    evol_perplexity = np.zeros((len(args.tasks), len(args.tasks)))
    evol_loss = np.zeros((len(args.tasks), len(args.tasks)))

    for ti, train_task in enumerate(args.tasks):
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_paths[train_task])
        model = accelerator.prepare(model)

        for ei, eval_task in enumerate(args.tasks):
            print(f"Train Task: {train_task}, Eval Task: {eval_task}")
            eval_loss, perplexity = evaluate(
                args, model, eval_dataloaders[eval_task], accelerator
            )
            print(f"loss/eval: {eval_loss:.4f}, perplexity: {perplexity:.4f}")
            evol_perplexity[ti, ei] = perplexity
            evol_loss[ti, ei] = eval_loss
            f.write(f"{train_task}\t{eval_task}\t{eval_loss:4f}\t{perplexity:.4f}\n")
    f.close()


if __name__ == "__main__":
    evaluate_all()


# export CUDA_VISIBLE_DEVICES=3; python scripts/validation_loss.py --tasks zeroshot other web --load_model_dir /nas-ssd/prateek/projects/codeCL/codeparrot/saved_models/test_all --load_data_dir /nas-ssd/prateek/projects/codeCL/data/bigquery
