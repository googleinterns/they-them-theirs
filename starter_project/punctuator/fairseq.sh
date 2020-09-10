UTCOME: train a word-level seq2seq model using Transformer and fairseq toolkit

echo "Cloning Facebook's fairseq repo..."
git clone https://github.com/pytorch/fairseq.git

echo "Installing packages..."
cd fairseq
pip install --editable ./
pip install fastBPE sacremoses subword_nmt

cd examples/translation

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git


echo "Copying punctuator dataset to directory punct_dataset..."
DIR_DATA=punct_dataset
BUCKET=tony_punctuator_2

mkdir $DIR_DATA
gsutil cp gs://$BUCKET/*.txt ./$DIR_DATA

# need to rename files to match huggingface input
cd $DIR_DATA
mv train_nopunc.txt train.source
mv train.txt train.target
mv valid_nopunc.txt valid.source
mv valid.txt valid.target
mv test_nopunc.txt test.source
mv test.txt test.target


cd ../../..


echo "Preprocessing / binarizing the data..."
# perl script uses multi-threading to speed up this process
TEXT=examples/translation/$DIR_DATA
DIR_TOKENIZED=punct_dataset.tokenized.source-target
fairseq-preprocess --source-lang source --target-lang target \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/$DIR_TOKENIZED \
    --workers 20
    

echo "Training seq2seq Transformer model..."
# can add more GPUs to CUDA_VISIBLE_DEVICES if more GPUs are available
# training objective is to maximize BLEU score
CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/$DIR_TOKENIZED \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
   
   
echo "Evaluating trained model..."
fairseq-generate data-bin/$DIR_TOKENIZED \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe

