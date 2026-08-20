root_dir=~/new_work/speech_ppl

echo "Running CosyVoice scoring..."
python $root_dir/TASTE-SpokenLM/STAGE1_TRAIN/CosyVoice/cozyvoice_ppl.py \
    --dataset_dir $root_dir/speechocean762/WAVE/ \
    --annotation_dir $root_dir/src/scores_enhanced.json \
    --alignments_file $root_dir/src/mfa/phone_extraction.json
