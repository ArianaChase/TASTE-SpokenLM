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

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

cosyvoice = CosyVoice('/home/u5504709/new_work/speech_ppl/pretrained_models/CosyVoice-300M-SFT')
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

        for spk_dir in pbar:
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

def aggregate_batch(batch_losses, granularity, pooling, norm_dict):

    nan_count = 0
    batch_results = []

    auc_threshold = 0.5 if granularity == "phone" else 3

    for sample in batch_losses:
        human_scores = sample['human_scores']
        alignments = sample['alignments']
        losses_with_timestamps = sample['timestamped_surprisal']

        if granularity != 'utterance':
            for i in range(0, len(alignments)):
                current_alignment = alignments[i]

                a_start = current_alignment["start"]
                a_end = current_alignment["end"]
                losses = []

                for loss_item in losses_with_timestamps:
                    t_start = loss_item[1]
                    t_end = loss_item[2]
                    if is_overlapping(a_start, a_end, t_start, t_end):
                        losses.append(loss_item[0])
                
                # pooling
                loss_pooled = np.nan
                
                if pooling == "mean":
                    loss_pooled = np.mean(losses) if len(losses) > 0 else np.nan
                elif pooling == "max":
                    loss_pooled = np.max(losses) if len(losses) > 0 else np.nan
                elif pooling == "std":
                    loss_pooled = np.std(losses) if len(losses) > 1 else np.nan
                else:
                    raise Exception("No pooling method specified.")
                
                if np.isnan(loss_pooled):
                    nan_count += 1

                # normalization
                if granularity == "phone":
                    # z-score normalization
                    phone_label = strip_stress(alignments[i]['label']) # type: ignore
                    p_mean = norm_dict[phone_label]['mean'] # type: ignore
                    p_std = norm_dict[phone_label]['std'] # type: ignore
                    loss_pooled_norm = ((loss_pooled - p_mean) / p_std) if p_std > 0 else np.nan
                else :
                    word = alignments[i]['label']
                    freq = word_frequency(word, 'en')
                    neg_log_freq = -math.log(freq) if freq > 0 else np.nan  # guard against unknown words
                    w_mean = None
                    w_std = None

                    for bucket, item in norm_dict.items():
                        s = item['freq_range']
                        clean_s = s.strip("()[]")
                        left_str, right_str = clean_s.split(",")
                        left = float(left_str)
                        right = float(right_str)
                        interval = pd.Interval(left, right, closed="right")

                        if neg_log_freq in interval:
                            w_mean = item['mean']
                            w_std = item['std']

                    if w_mean != None and w_std != None and w_std > 0:
                        loss_pooled_norm = (loss_pooled - w_mean) / w_std
                    else:
                        loss_pooled_norm = np.nan

                batch_results.append({
                    "speaker" : sample['speaker'],
                    "filename" : sample['filename'],
                    "label" : current_alignment['label'],
                    'auc_label' : 1 if human_scores[i]['accuracy'] > auc_threshold else 0,
                    "ppl_loss" : -loss_pooled,
                    "ppl_loss_norm" : -loss_pooled_norm,
                    "human_score": human_scores[i]['accuracy']
                })

        else:
            losses = []
            for loss_item in losses_with_timestamps:
                losses.append(loss_item[0])
            
            # pooling
            loss_pooled = np.nan
            
            if pooling == "mean":
                loss_pooled = np.mean(losses) if len(losses) > 0 else np.nan
            elif pooling == "max":
                loss_pooled = np.max(losses) if len(losses) > 0 else np.nan
            elif pooling == "std":
                loss_pooled = np.std(losses) if len(losses) > 1 else np.nan
            else:
                raise Exception("No pooling method specified.")
            
            if np.isnan(loss_pooled):
                nan_count += 1

            batch_results.append({
                "speaker" : sample['speaker'],
                "filename" : sample['filename'],
                "label" : sample["text"],
                'auc_label' : 1 if human_scores > auc_threshold else 0,
                "ppl_loss" : -loss_pooled,
                "ppl_loss_norm" : np.nan,
                "human_score": human_scores
            })
            
    return batch_results, nan_count

def get_losses(dataset, labels_dict, alignments_path, granularity, pooling, norm_dict, limit=None):
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
    error_log = []
    nan_count = 0
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
        utterance_ppl_info = []
        pbar.set_description(f"Getting per phone losses for file: {filename}")

        # external preparation
        alignment_obj = next((item for item in alignment_list if item.get('audio_id') == filename), None)
        phone_alignments = alignment_obj["phone_alignment"] # type: ignore list of phone objects {start, end, label}
        word_alignments = alignment_obj["word_alignment"] # type: ignore list of word objects {start, end, label}
        alignments = None

        human_annotation_obj = labels_dict.get(filename)
        phone_scores = []
        word_scores = []
        utterance_score = human_annotation_obj["accuracy"]

        for word_obj in human_annotation_obj["words"]:
            for i in range(0, len(word_obj["phones"])):
                phone_scores.append({
                    "phone" : word_obj["phones"][i], 
                    "accuracy" : word_obj["phones-accuracy"][i]
                }) 
            word_scores.append({
                "word" : word_obj["text"],
                "accuracy" : word_obj["accuracy"],
                "stress" : word_obj["stress"] # unused for now
            })

        if granularity == "phone":      
            phone_alignments_labels = [item['label'] for item in phone_alignments]
            phone_scores_labels = [item['phone'] for item in phone_scores]

            matcher = SequenceMatcher(None, phone_scores_labels, phone_alignments_labels)
            opcodes = matcher.get_opcodes()
            has_error = False
            matched_alignments = []
            matched_scores = []
            for tag, a_idx1, a_idx2, b_idx1, b_idx2 in opcodes:
                if tag == "equal":
                    matched_scores.extend(phone_scores[a_idx1:a_idx2])
                    matched_alignments.extend(phone_alignments[b_idx1:b_idx2])
            phone_alignments = matched_alignments
            phone_scores = matched_scores

            if len(phone_alignments) != len(phone_scores):
                error_log.append(f"Alignment mismatch at file {filename}. {len(phone_alignments)} alignments but {len(phone_scores)} scores.")
                error_log.append(f"{[item['label'] for item in phone_alignments]}\n{[item['phone'] for item in phone_scores]}")

            human_scores = phone_scores
            alignments = phone_alignments

        elif granularity == "word":
            if len(word_scores) != len(word_alignments):
                raise Exception("Human word annotations cannot be aligned with word alignments.")
            human_scores = word_scores
            alignments = word_alignments

        elif granularity == "utterance":
            human_scores = utterance_score
        else:
            raise Exception("Invalid granularity")

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
            'human_scores' : human_scores, # depends on granularity
            'alignments' : alignments,     # depends on grnularity
            'text' : human_annotation_obj['text']
        })      

        if len(sample_buffer) == BATCH_SIZE:
            batch_losses = process_batch(sample_buffer, cosyvoice, device)                  # inference and get losses for batch
            batch_results, batch_nan_count = aggregate_batch(batch_losses, granularity, pooling, norm_dict)  # pool and norm per granularity
            nan_count += batch_nan_count
            ppl_results += batch_results
            sample_buffer = []

        file_count += 1

    if len(sample_buffer) > 0:
        batch_losses = process_batch(sample_buffer, cosyvoice, device)
        batch_results, batch_nan_count = aggregate_batch(batch_losses, granularity, pooling, norm_dict)
        nan_count += batch_nan_count
        ppl_results.extend(batch_results)
    
    print(f"FILES RECORDED: {file_count}")

    return {
        "results" : ppl_results,
        "nan_count" : nan_count
    }

def append_to_sheet(
    row_data,
    spreadsheet_name="ICASSP 2026 Experiment Results",
    worksheet_name="main",
    service_account_file="/home/u5504709/new_work/speech_ppl/src/service_account.json"
):
    # Authenticate
    creds = Credentials.from_service_account_file(
        service_account_file,
        scopes=SCOPES
    )

    client = gspread.authorize(creds)

    # Open sheet
    spreadsheet = client.open(spreadsheet_name)
    worksheet = spreadsheet.worksheet(worksheet_name)

    # Append row
    worksheet.append_row(row_data)

    print("Spreadsheet updated successfully.")

def per_phone_auc(results):

    phone_auc_dict = {}

    for result in results:
        if result['label'] not in list(phone_auc_dict.keys()):
            phone_auc_dict[result['label']] = {
                'auc_labels' : [result['auc_label']],
                'ppl_losses' : [result['ppl_loss']],
                'ppl_norm_losses' : [result['ppl_loss_norm']],
            }
        else:
            phone_auc_dict[result['label']]['auc_labels'].append(result['auc_label'])
            phone_auc_dict[result['label']]['ppl_losses'].append(result['ppl_loss'])
            phone_auc_dict[result['label']]['ppl_norm_losses'].append(result['ppl_loss_norm'])
    
    roc_auc_scores = []
    roc_auc_scores_norm = []

    for phone, item in phone_auc_dict.items():
        df = pd.DataFrame(item)
        df = df.dropna(axis=0, subset=['ppl_losses', 'auc_labels'])

        y_true = df['auc_labels']
        y_score = df['ppl_losses']

        if len(np.unique(y_true)) != 1 and len(y_score) >= 1:
            auc = roc_auc_score(y_true, y_score)
            roc_auc_scores.append(auc)

        df = pd.DataFrame(item)
        df = df.dropna(axis=0, subset=['ppl_norm_losses', 'auc_labels'])
        y_true_norm = df['auc_labels']
        y_score_norm = df['ppl_norm_losses']

        if len(np.unique(y_true_norm)) != 1 and len(y_score_norm) >= 1:
            auc = roc_auc_score(y_true_norm, y_score_norm) # because 0 - 1 is akin to big loss - small loss
            roc_auc_scores_norm.append(auc)

    return {
        'auc' : np.nanmean(roc_auc_scores) if len(roc_auc_scores) >= 1 else "n/a",
        'auc_norm' : np.nanmean(roc_auc_scores_norm) if len(roc_auc_scores_norm) >= 1 else "n/a"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--alignments_file", required=True)

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

    NORM_DICT_DIR = f"/home/u5504709/new_work/speech_ppl/src/gslm/tools/result_dicts"
    MODEL_TYPE = "COSYVOICE"
    MODEL_NAME = f"COSYVOICE"
    granularity = "phone"
    pool = "mean"

    for granularity in ["phone", "word", "utterance"]:
        for pool in ['mean', 'max', 'std']:

            if granularity == "phone" or granularity == "word":
                with open(f"{NORM_DICT_DIR}/{MODEL_NAME}_{granularity}_{pool}_norm.json", "r") as f:
                    norm_dict = json.load(f)
            else:
                norm_dict = None    

            results = get_losses(
                dataset=processed_dataset, 
                labels_dict=human_scores, 
                alignments_path=args.alignments_file, 
                granularity=granularity,
                pooling=pool,
                norm_dict=norm_dict,
                limit=None,
                )

            ppl_results = results["results"]
            nan_percent = (results["nan_count"] / len(ppl_results)) * 100
            
            # correlate
            df = pd.DataFrame(ppl_results)
            df.dropna(axis=0, subset=df.columns.drop('ppl_loss_norm'), inplace=True)
            x = df["ppl_loss"]
            y = df["human_score"]
            pcc = scipy.stats.pearsonr(x, y)

            # auc
            y_score = df["ppl_loss"]
            y_true = df["auc_label"]
            if len(np.unique(y_true)) != 1:
                auc = roc_auc_score(y_true, y_score)
            else:
                auc = "n/a"

            if granularity == "phone" or granularity == "word":
                df_norm = pd.DataFrame(ppl_results)
                df_norm.dropna(axis=0, inplace=True)
                x_norm = df_norm["ppl_loss_norm"]
                y_norm = df_norm["human_score"]
                pcc_norm = scipy.stats.pearsonr(x_norm, y_norm)
                pcc_norm_stats = pcc_norm.statistic
                pcc_norm_pvalue = pcc_norm.pvalue

                y_score_norm = df_norm["ppl_loss_norm"]
                y_true_norm = df_norm["auc_label"]
                if len(np.unique(y_true_norm)) != 1:
                    auc_norm = roc_auc_score(y_true_norm, y_score_norm)
                else:
                    auc_norm = "n/a"
            
            else:
                pcc_norm_stats = "n/a"
                pcc_norm_pvalue = "n/a"
                auc_norm = "n/a"

            if granularity == "phone":
                per_phone_auc_result = per_phone_auc(ppl_results)
            else:
                per_phone_auc_result = {
                    "auc" : "n/a",
                    "auc_norm" : "n/a"
                }
                
            # Record in CSV
            append_to_sheet([MODEL_TYPE, MODEL_NAME, granularity, pool, pcc.statistic, pcc.pvalue, pcc_norm_stats, pcc_norm_pvalue, auc, per_phone_auc_result['auc'], auc_norm, per_phone_auc_result['auc_norm'], f"{nan_percent:2f}" + "%", len(df)])
            
    

