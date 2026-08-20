import argparse
from cosyvoice.cli.cosyvoice import CosyVoice  
from cosyvoice.utils.file_utils import load_wav, load_wav_array
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

cosyvoice = CosyVoice('/home/u5504709/new_work/speech_ppl/pretrained_models/CosyVoice-300M-SFT')
frontend = cosyvoice.frontend
print(frontend.device)

FRAMERATE = int(np.array(cosyvoice.model.flow.input_frame_rate))
TOKEN_DURATION =  1 / FRAMERATE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8

print(torch.cuda.is_available())
print(onnxruntime.get_available_providers())

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
    
def process_librispeech(input_dataset):

    audio_file_info = []

    for sample in input_dataset:

        audio_file_info.append({
            "speaker" : sample["speaker_id"],
            "array" : sample["audio"]["array"],
            "sr" : sample["audio"]["sampling_rate"],
            "text" : sample["text"],
            "filename" : sample["id"] 
        }) 

    return audio_file_info

def process_alignments_ds(input_dataset):
    
    alignments = []

    for sample in input_dataset:
        phone_list = []
        word_list = []

        for phone_alignment in sample['phonemes']:
            phone_list.append({
                "start" : phone_alignment['start'],
                "end" : phone_alignment["end"],
                "label" : phone_alignment["phoneme"]
            })

        for word_alignment in sample["words"]:
            word_list.append({
                "start" : word_alignment['start'],
                "end" : word_alignment["end"],
                "label" : word_alignment['word']
            })

        alignments.append({
            "audio_id" : sample['id'],
            "phone_alignment" : phone_list,
            "word_alignment" : word_list
        })
    
    return alignments


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
        timestamped_surprisal = list(zip(timestamps_start, timestamps_end, row_surprisal.tolist()))

        sample_meta.update({'timestamped_surprisal' : timestamped_surprisal})
        batch_info.append(sample_meta)

    return batch_info

def aggregate_batch(batch_losses, granularity, pooling):

    nan_count = 0
    batch_dict = {}

    for sample in batch_losses:
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

                if granularity == "phone":
                    phone_label = strip_stress(alignments[i]['label'])
    
                    if phone_label in batch_dict:
                        batch_dict[phone_label]['count'] += 1
                        batch_dict[phone_label]['losses'].append(loss_pooled)
                    else:
                        batch_dict[phone_label] = {
                            "count" : 1,
                            "losses" : [loss_pooled]
                        }
                elif granularity == "word":
                    # TODO: Implement wordfreq normalization here
                    word = alignments[i]['label']
                    freq = word_frequency(word, 'en')
                    neg_log_freq = -math.log(freq) if freq > 0 else np.nan  # guard against unknown words
    
                    if word in batch_dict:
                        batch_dict[word]['freq'] = neg_log_freq
                        batch_dict[word]['losses'].append(loss_pooled)
                    else:
                        batch_dict[word] = {
                            'freq' : neg_log_freq,
                            'losses' : [loss_pooled]
                        }

    return batch_dict

def get_losses(dataset, alignment_list, granularity, pooling, limit=None):
    '''
    dataset         : dataset object with speaker, filename, and path
    labels_dict     : dictionary of human annotated information, sorted by filename 
    alignments_path : json file of phone-level alignment boundaries
    granularity     : phone/word/utterance level
    pooling         : pooling method (max/mean/std)
    norm_dict      : dict for normalization
    '''

    result_dict = {}
    file_count = 0
    error_log = []
    nan_count = 0
    lim = limit if limit != None else len(dataset)

    pbar = tqdm(dataset)
    sample_buffer = []

    for sample in pbar:
        if file_count >= lim:
            break

        # info
        speaker = sample["speaker"]
        array = sample["array"]
        sr = sample['sr']
        filename = sample["filename"]
        utterance_ppl_info = []
        pbar.set_description(f"Getting per phone losses for file: {filename}")

        # external preparation
        alignment_obj = next((item for item in alignment_list if item.get('audio_id') == filename), None)
        phone_alignments = alignment_obj["phone_alignment"] # type: ignore list of phone objects {start, end, label}
        word_alignments = alignment_obj["word_alignment"] # type: ignore list of word objects {start, end, label}
        alignments = None

        if granularity == "phone":      
            alignments = phone_alignments
        elif granularity == "word":
            alignments = word_alignments

        # sample pre-processing
        canonical_text = sample['text']
        wav = load_wav_array(array, sr, 16000)
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
            'filename' : filename,
            'alignments' : alignments,     # depends on grnularity
            'text' : sample['text']
        })      

        if len(sample_buffer) == BATCH_SIZE:
            batch_losses = process_batch(sample_buffer, cosyvoice, device)                  # inference and get losses for batch
            batch_dict = aggregate_batch(batch_losses, granularity, pooling)  # pool and norm per granularity

            for id, batch_data in batch_dict.items():
                if id in result_dict:
                    result_dict[id]["losses"].extend(batch_data["losses"])
                    if granularity == "phone":
                        result_dict[id]["count"] += batch_data["count"]                        
                else:
                    result_dict[id] = batch_data

            sample_buffer = []

        file_count += 1

    if len(sample_buffer) > 0:
        batch_losses = process_batch(sample_buffer, cosyvoice, device)
        batch_dict = aggregate_batch(batch_losses, granularity, pooling)
        for id, batch_data in batch_dict.items():
            if id in result_dict:
                result_dict[id]["losses"].extend(batch_data["losses"])

                if granularity == "phone":
                    result_dict[id]["count"] += batch_data["count"]
            else:
                result_dict[id] = batch_data
    
    print(f"FILES RECORDED: {file_count}")

    return result_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--alignments_file", required=True)

    args = parser.parse_args()

    # alignments
    from datasets import load_dataset
    t1 = time.time()
    alignments_ds = load_dataset("gilkeyio/librispeech-alignments", streaming=True)
    t2 = time.time()
    print(f"Downloaded alignments ds in {t2 - t1} secs")
    processed_alignments_ds = process_alignments_ds(alignments_ds["dev_clean"])

    # process dataset
    t3 = time.time()
    print(f"Processed alignments in {t3 - t2} secs")

    dataset = load_dataset("openslr/librispeech_asr", "clean", split="validation", streaming=True)
    t4 = time.time()
    print(f"Downloaded speech ds in {t4 - t3} secs")
    processed_dataset = process_librispeech(dataset)
    print(f"Processed {len(processed_dataset)} samples.")

    # get losses
    NORM_DICT_DIR = None
    MODEL_NAME = "COSYVOICE"
    GRANULARITY = "phone"
    result_dicts_path = "/home/u5504709/new_work/speech_ppl/src/gslm/tools/result_dicts"

    for granularity in ['phone', 'word']:
        for pool in ["std"]:
            result_dict = get_losses(
                dataset=processed_dataset, 
                alignment_list=processed_alignments_ds, 
                granularity=granularity,
                pooling="mean",
                limit=None,
            )

            if granularity == "phone":
                result_dict.pop("spn")

                for key, phone_info in result_dict.items():
                    result_dict[key]['mean'] = np.mean(result_dict[key]['losses'])
                    result_dict[key]['std'] = np.std(result_dict[key]['losses'])

                with open(f"{result_dicts_path}/{MODEL_NAME}_phone_{pool}_norm.json", "w") as f:
                    json.dump(result_dict, f)

                with open("/home/u5504709/new_work/speech_ppl/src/gslm/tools/error_log", "a") as f:
                    f.write(f"In total, there are {len(result_dict)} unique phones in the {pool} dictionary.")
                    f.write("\n")
                    f.write(f"{sorted(list(result_dict.keys()))}\n")

            elif granularity == "word":

                # TODO: Implement bucketing here, then create a dictionary with only the buckets

                NUM_BUCKETS = 5

                print(f"Words to be sorted: {len(result_dict)}")
                all_neg_log_freqs = [item['freq'] for word, item in result_dict.items()]
                nan_count = np.isnan(all_neg_log_freqs).sum()
                x_series = pd.Series(all_neg_log_freqs)
                nan_count = x_series.isna().sum()
                all_neg_log_freqs = x_series.dropna().to_numpy() 
                
                bucket_boundaries = pd.qcut(all_neg_log_freqs, q=NUM_BUCKETS)

                print(f"all neg log freqs: {all_neg_log_freqs}, nan count {nan_count}")
                print(f"Boundaries: {bucket_boundaries}, type {type(bucket_boundaries)}")

                bucketed_word_dict = {}

                for idx, bucket in enumerate(np.unique(bucket_boundaries)):
                    print(f"Bucket {idx}: {bucket}")
                    print(f"Type: {type(bucket)}")

                    bucket_losses = []
                    bucket_word_count = 0

                    for word, info in result_dict.items():
                        if info['freq'] in bucket:
                            bucket_losses.append(np.nanmean(info['losses']))
                            bucket_word_count += 1

                    bucketed_word_dict[idx] = {
                        'freq_range' : str(bucket),
                        'losses' : bucket_losses,
                        'mean' : np.mean(bucket_losses),
                        'std' : np.std(bucket_losses),
                        'count' : bucket_word_count
                    }
                
                with open(f"{result_dicts_path}/{MODEL_NAME}_word_{pool}_norm.json", "w") as f:
                    json.dump(bucketed_word_dict, f)
        
        print(f"File count: {len(processed_dataset)}")