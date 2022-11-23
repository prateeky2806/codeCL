WORKDIR="/nas-ssd/prateek/projects/codeCL/repos/codeparrot"
export PYTHONPATH=$WORKDIR

GPU = ${1}
RUN_FN = ${WORKDIR}/scripts/codeparrot_training.py
MODEL_TAG = ${2}
TOKENIZER = ${3}
SAVE_DIR = ${4}
TRAIN_DATA_FILE = ${5}
VAL_DATA_FILE = ${6}
MAX_STEPS = ${7}
LR = ${8}
TRAIN_BS = ${9}
VAL_BS = ${10}
WARMUP = ${11}
CKPT_STEPS = ${12}

export CUDA_VISIBLE_DEVICES=${GPU}; python ${RUN_FN} \
--model_ckpt=${MODEL_TAG} --tokenizer_ckpt=${TOKENIZER} \
--train_batch_size=${TRAIN_BS} --valid_batch_size=${VAL_BS}
--learning_rate=${LR} --num_warmup_steps=${WARMUP} --gradient_accumulation=1 --gradient_checkpointing=False \
--max_train_steps=${MAX_STEPS} --save_checkpoint_steps=${CKPT_STEPS} \
--save_dir=${SAVE_DIR} --dataset_name_train=${TRAIN_DATA_FILE} --dataset_name_valid=${VAL_DATA_FILE}



# export CUDA_VISIBLE_DEVICES=3; python scripts/codeparrot_training.py \
# --model_ckpt=gpt2-medium --train_batch_size=4 --valid_batch_size=8
# --learning_rate=5e-4 --num_warmup_steps=2000 --gradient_accumulation=1 --save_dir=./clone_model/gpt-medium
# --gradient_checkpointing=False --max_train_steps=150000 --save_checkpoint_steps=1000
# --dataset_name_train=/nas-ssd/prateek/projects/codeCL/data/bigquery/other_train.jsonl
# --dataset_name_valid=/nas-ssd/prateek/projects/codeCL/data/bigquery/other_val.jsonl