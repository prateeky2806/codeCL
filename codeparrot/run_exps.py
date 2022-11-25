import os, sys
import argparse
import copy


def get_train_commands(args):
    commands = []
    for i, task in enumerate(args.tasks, start=1):
        config = copy.deepcopy(vars(args))
        del config["tasks"]
        cmd = f"sh ~/.bashrc; conda activate codeparrot; cd {config['workdir']}; "
        cmd += (
            f"accelerate launch "
            if args.accelerate
            else f"export CUDA_VISIBLE_DEVICES={config['gpu']}; python "
        )
        cmd += os.path.join(config["workdir"], "scripts", "codeparrot_training.py")

        del config["gpu"]
        del config["accelerate"]
        del config["workdir"]
        del config["only_cmd"]

        config["dataset_name_train"] = (
            os.path.join(config["dataset_dir"], f"{task}_train.jsonl")
            if config["dataset_dir"] is not None
            and config["dataset_name_train"] is None
            else config["dataset_name_train"]
        )
        config["dataset_name_valid"] = (
            os.path.join(config["dataset_dir"], f"{task}_val.jsonl")
            if config["dataset_dir"] is not None
            and config["dataset_name_valid"] is None
            else config["dataset_name_valid"]
        )
        del config["dataset_dir"]
        for k, v in config.items():
            if k == "model_ckpt" and i > 1:
                cmd += f" --{k} {os.path.join(args.save_dir, 'best')} "
            else:
                cmd += f" --{k} {v} "
        print(cmd)
        commands.append(cmd)
    return commands


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--accelerate", action="store_true")
    parser.add_argument("--only_cmd", action="store_true")

    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--dataset_name_train", type=str, default=None)
    parser.add_argument("--dataset_name_valid", type=str, default=None)
    parser.add_argument("--tokenizer_ckpt", type=str, default="codeparrot/codeparrot")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_warmup_steps", type=int, default=2000)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_train_steps", type=int, default=150000)
    parser.add_argument("--save_checkpoint_steps", type=int, default=1500)
    parser.add_argument("--save_dir", type=str, default="./saved_results/test")
    parser.add_argument("--gpu", type=int, default=3)
    args = parser.parse_args()

    commands = get_train_commands(args)
    if not args.only_cmd:
        for cmd in commands:
            print(f"\n\n\n\n\n\n\n\n\n{cmd}")
            os.system(cmd)


# python run_exps.py --model_ckpt=gpt2-medium --dataset_dir=/nas-ssd/prateek/projects/codeCL/data/bigquery --workdir=/nas-ssd/prateek/projects/codeCL/repos/codeparrot --tasks other networking database --gpu 3 --max_train_steps 100 --num_warmup_steps 10 --save_checkpoint_steps 100 --save_dir ./saved_models/test_all1
