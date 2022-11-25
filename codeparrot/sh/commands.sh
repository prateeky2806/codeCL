### FUll Pretraining.
# GPT small
python run_exps.py --model_ckpt=gpt2 --accelerate --lr_scheduler_type linear --dataset_dir=/nas-hdd/prateek/data/package_level --workdir=/nas-ssd/prateek/projects/codeCL/codeparrot --tasks other --max_train_steps 50000 --train_batch_size 12 --valid_batch_size 24 --num_warmup_steps 1000 --save_checkpoint_steps 2500 --save_dir /nas-hdd/prateek/models/package_level/pretrain_gptsmall
# GPT medium
python run_exps.py --model_ckpt=gpt2-medium --accelerate --lr_scheduler_type linear --dataset_dir=/nas-hdd/prateek/data/package_level --workdir=/nas-ssd/prateek/projects/codeCL/codeparrot --tasks other --max_train_steps 50000 --num_warmup_steps 1000 --save_checkpoint_steps 2500 --save_dir /nas-hdd/prateek/models/package_level/pretrain2
# GPT Large
python run_exps.py --model_ckpt=gpt2-large --accelerate --lr_scheduler_type linear --dataset_dir=/nas-hdd/prateek/data/package_level --workdir=/nas-ssd/prateek/projects/codeCL/codeparrot --tasks other --max_train_steps 50000 --train_batch_size 2 --valid_batch_size 4 --num_warmup_steps 1000 --save_checkpoint_steps 2500 --save_dir /nas-hdd/prateek/models/package_level/pretrain_gptlarge