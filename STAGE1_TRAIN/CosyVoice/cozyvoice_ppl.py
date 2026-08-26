import argparse
from cosyvoice.cli.cosyvoice import CosyVoice  
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
import torchaudio
from modelscope import snapshot_download
import torch
import torch.nn.functional as F
from cosyvoice.utils.common import IGNORE_ID
import numpy as np
import json
import os
from tqdm import tqdm
import time
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher
from wordfreq import word_frequency
import math
from sklearn.metrics import roc_auc_score
import onnxruntime
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import scipy.stats
from sklearn.metrics import roc_auc_score
import csv

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

cosyvoice = CosyVoice('/home/ubuntu/speech_ppl/pretrained_models/CosyVoice-300M')
frontend = cosyvoice.frontend
print(frontend.device)

FRAMERATE = int(np.array(cosyvoice.model.flow.input_frame_rate))
TOKEN_DURATION =  1 / FRAMERATE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8

print(torch.cuda.is_available())
print(onnxruntime.get_available_providers())

def parse_human_annotations(filename):
    human_scores = {}
    with open(filename) as json_data:
        data = json.load(json_data)
        for audio_file in data:
            value = data[audio_file]
            human_scores[audio_file] = {
                "filename" : audio_file,
                "accuracy" : value["accuracy"],
                "fluency" : value["fluency"],
                "prosodic" : value["prosodic"],
                "completeness" : value["completeness"],
                "words" : value["words"],
                "text" : value["text"]
            }
    return human_scores

def is_overlapping(a_start, a_end, b_start, b_end):
    if (a_end >= b_start and a_start <= b_end):
        return True
    else:
        return False

def strip_stress(phone_label):
    if phone_label[-1].isdigit():
        return phone_label[:-1]
    else:
        return phone_label
    
def process_speechocean(input_dataset):

        audio_file_info = []
        spk_count = 0
        ignored_speakers = ["1076"]
        removed = []

        pbar = tqdm(sorted(os.listdir(input_dataset)))
        files = 0
        for spk_dir in pbar:
            # if files >= 5:
            #     break
            spk_count += 1
            speaker = spk_dir[7:None]
            pbar.set_description(f"Processing speaker: {speaker}")
            spk_dir_path = os.path.join(input_dataset, spk_dir)
            for audio_file in os.listdir(spk_dir_path):
                audio_path = os.path.join(spk_dir_path, audio_file)
                filename = os.path.basename(audio_path)[0:9]
                audio_file_info.append({
                    "speaker" : speaker,
                    "filename" : filename,
                    "path" : audio_path
                })
            # files += 1
        
        for i in range(len(audio_file_info) - 1, -1, -1):
            if audio_file_info[i]["speaker"] in ignored_speakers:
                removed.append(audio_file_info.pop(i))
                          

        return {
            "processed" : audio_file_info,
            "ignored" : removed,
            "spk_count" : spk_count - len(ignored_speakers)
        }

def collate_batch(samples):
    """samples = list of dicts, each with text_token, speech_token, embedding (all unbatched, single-utterance tensors)"""
    
    text_tokens = [s['text_token'].squeeze(0) for s in samples]      # remove existing batch dim of 1
    speech_tokens = [s['speech_token'].squeeze(0) for s in samples]
    embeddings = [s['embedding'].squeeze(0) for s in samples]

    text_token_lens = torch.tensor([t.size(0) for t in text_tokens])
    speech_token_lens = torch.tensor([s.size(0) for s in speech_tokens])

    batch = {
        'text_token': pad_sequence(text_tokens, batch_first=True, padding_value=0),
        'text_token_len': text_token_lens,
        'speech_token': pad_sequence(speech_tokens, batch_first=True, padding_value=0),
        'speech_token_len': speech_token_lens,
        'embedding': torch.stack(embeddings),
    }
    return batch

def process_batch(sample_buffer, cosyvoice, device):
    batch_info = []
    batch = collate_batch(sample_buffer)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.inference_mode():
        logits, lm_target = cosyvoice.model.llm.forward(batch, device)
        logits_reshaped = logits.transpose(1, 2)

        surprisal_per_position = F.cross_entropy(
            logits_reshaped,
            lm_target.long(),
            ignore_index=IGNORE_ID,
            reduction='none'
        )

    mask = (lm_target != IGNORE_ID)

    for i, sample_meta in enumerate(sample_buffer):
        # for each utterance:
        row_mask = mask[i]
        row_surprisal = surprisal_per_position[i][row_mask]

        utterance_ppl = row_surprisal.mean().exp().item()
        timestamps_start = [(t * TOKEN_DURATION) for t in range(len(row_surprisal))]
        timestamps_end = [((t + 1) * TOKEN_DURATION) for t in range(len(row_surprisal))]
        timestamped_surprisal = list(zip(row_surprisal.tolist(), timestamps_start, timestamps_end))

        sample_meta.update({'timestamped_surprisal' : timestamped_surprisal})
        batch_info.append(sample_meta)

    return batch_info

def get_losses(dataset, labels_dict, alignments_path, limit=None):
    '''
    dataset         : dataset object with speaker, filename, and path
    labels_dict     : dictionary of human annotated information, sorted by filename 
    alignments_path : json file of phone-level alignment boundaries
    granularity     : phone/word/utterance level
    pooling         : pooling method (max/mean/std)
    norm_dict      : dict for normalization
    '''

    ppl_results = []
    file_count = 0
    lim = limit if limit != None else len(dataset)

    # prepare and finalize alignments and dataset

    with open(alignments_path, 'r') as f:
            alignment_list = json.load(f)
    
    dataset_cleaned = []

    for sample in dataset:
        for idx in range(len(alignment_list) -1, -1, -1):
            if sample['filename'] == alignment_list[idx]['audio_id']:
                dataset_cleaned.append(sample)
            if alignment_list[idx]['speaker'] == '1076':
                alignment_list.pop(idx)
                
    if len(dataset_cleaned) != len(alignment_list):
        raise Exception(f"Length mismatch between alignments ({len(alignment_list)}) and dataset ({len(dataset_cleaned)})")

    pbar = tqdm(dataset_cleaned)
    sample_buffer = []

    for sample in pbar:
        if file_count >= lim:
            break

        # info
        speaker = sample["speaker"]
        file_path = sample["path"]
        filename = sample["filename"]
        pbar.set_description(f"Getting per phone losses for file: {filename}")

        human_annotation_obj = labels_dict.get(filename)

        # sample pre-processing
        canonical_text = human_annotation_obj['text']
        wav = load_wav(file_path, 16000)
        text_token, text_token_len = frontend._extract_text_token(canonical_text)
        speech_token, speech_token_len = frontend._extract_speech_token(wav)
        embedding = frontend._extract_spk_embedding(wav)

        #print(f"load_wav: {t1-t0:.3f}s | text_token: {t2-t1:.3f}s | speech_token: {t3-t2:.3f}s | embedding: {t4-t3:.3f}s")

        # get losses
        sample_buffer.append({
            'text_token': text_token,
            'speech_token': speech_token,
            'embedding': embedding,
            'speaker' : speaker,
            'file_path' : file_path,
            'filename' : filename,
            'text' : human_annotation_obj['text']
        })      

        if len(sample_buffer) == BATCH_SIZE:
            batch_losses = process_batch(sample_buffer, cosyvoice, device)                  # inference and get losses for batch
            for sample in batch_losses:
                for loss in sample['timestamped_surprisal']:
                    ppl_results.append({
                        'speaker' : sample['speaker'],
                        'file_path' : sample['file_path'],
                        'filename' : sample['filename'],
                        'ppl_loss' : loss[0],
                        'start' : loss[1],
                        'end' : loss[2]
                    })
            sample_buffer = []

        file_count += 1

    if len(sample_buffer) > 0:
        batch_losses = process_batch(sample_buffer, cosyvoice, device)
        for sample in batch_losses:
            for idx, loss in enumerate(sample['timestamped_surprisal']):
                ppl_results.append({
                    'speaker' : sample['speaker'],
                    'file_path' : sample['file_path'],
                    'filename' : sample['filename'],
                    'token_id' : idx,
                    'ppl_loss' : loss[0],
                    'start' : loss[1],
                    'end' : loss[2]
                })    

    print(f"FILES RECORDED: {file_count}")

    return {
        "results" : ppl_results,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--alignments_file", required=True)
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--output_dir")

    args = parser.parse_args()

    # get labels to compare to
    score_labels = args.annotation_dir
    human_scores = parse_human_annotations(score_labels)

    # process dataset
    input_dataset = args.dataset_dir

    processed = process_speechocean(input_dataset)
    processed_dataset = processed["processed"]
    ignored_samples = processed["ignored"]
    spk_count = processed["spk_count"]
    print(f"Processed {len(processed_dataset)} samples.")

    NORM_DICT_DIR = f"{args.root_dir}/src/metrics/result_dicts"
    MODEL_TYPE = "COSYVOICE"
    MODEL_NAME = f"COSYVOICE"
    OUTPUT_DIR = args.output_dir
    csv_path = f"{OUTPUT_DIR}/{MODEL_TYPE}_{MODEL_NAME}_per_token_losses.csv"   

    results = get_losses(
        dataset=processed_dataset, 
        labels_dict=human_scores, 
        alignments_path=args.alignments_file, 
        limit=None,
        )

    ppl_results = results["results"]

    with open(csv_path, "w") as f:
        fieldnames = ppl_results[0].keys()
        dict_writer = csv.DictWriter(f, fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(ppl_results)
    
    