

python run_exps.py --model_ckpt=gpt2-medium \
--dataset_dir=/nas-hdd/prateek/data/package_level \
--workdir=/nas-ssd/prateek/projects/codeCL/codeparrot \
--tasks other web science networking database gui \
--max_train_steps 30000 --num_warmup_steps 1000 --save_checkpoint_steps 1500 \
--save_dir ./saved_models/test_all1 \
--accelerate


