# OUTCOME: fine-tune BART-base seq2seq language model on punctuator dataset (punctuation + capitalization)

echo "Cloning Huggingface Transformers repo..."
git clone https://github.com/huggingface/transformers.git

echo "Installing packages..."
cd transformers
pip install -e .
pip install -r examples/requirements.txt
cd examples/seq2seq

echo "Copying punctuator dataset to directory punct_dataset..."
DIR_DATA=punct_dataset
BUCKET=tony_punctuator_2

mkdir $DIR_DATA
gsutil cp gs://$BUCKET/*.txt ./$DIR_DATA

# need to rename files to match huggingface input
cd $DIR_DATA
mv train_nopunc.txt train.source
mv train.txt train.target
mv valid_nopunc.txt val.source
mv valid.txt val.target
mv test_nopunc.txt test.source
mv test.txt test.target
cd ..

echo "Fine-tuning BART language model on punctuator dataset..."
DATA_PATH=${PWD}/$DIR_DATA
DIR_RESULTS=punct_results
LM=facebook/bart-base

mkdir $DIR_RESULTS

# full list of command line options here: https://github.com/huggingface/transformers/blob/master/examples/seq2seq/finetune.py
# default learning rate: 3e-5, can adjust with option --learning_rate=$learning_rate
# GPU usage: optionally, can prepend CUDA_AVAILABLE_DEVICES and add --gpus=$num_gpus
# no hangup: prepend nohup to command
# TODO: if registered with wandb, can prepend WANDB_PROJECT='$proj_name' and add --logger wandb option
./finetune.sh \
    --data_dir $DATA_PATH \
    --train_batch_size=1 \
    --eval_batch_size=1 \
    --output_dir=$DIR_RESULTS \
    --num_train_epochs 1 \
    --model_name_or_path $LM \
    --max_target_length=160 \
    --val_max_target_length=160 \
    --test_max_target_length=160 \
    --fp16
    
# results will be in $DIR_RESULTS
# training for 1 epoch takes ~4-5 hours on a 16GB NVIDIA Tesla V100 GPU
# TODO: sometimes model will truncate output. Setting the targ_length longer can help, but still not perfect. Will look into this
