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
from pathlib import Path
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

def process_synth(input_dataset, audio_version, metadata, text_version):

        audio_file_info = []

        pbar = tqdm(os.listdir(input_dataset))

        for file_path in pbar:
            audio_path = f"{input_dataset}/{file_path}"
            file_metadata = os.path.basename(audio_path).split("_")            
            filename = file_metadata[0]
            ver = Path(file_metadata[1]).stem
            metadata_obj = metadata[metadata["stim_id"] == filename].iloc[0]

            if ver != audio_version:
                continue

            if text_version == "clean":
                text = metadata_obj['canonical_text']
            else:
                text = metadata_obj['substituted_text']

            audio_file_info.append({
                "filename" : filename,
                "audio_version" : ver,
                "path" : audio_path,
                "text" : text
            })
        return {
            "processed" : audio_file_info,
        }

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
        row_surprisal = row_surprisal[:-1]

        utterance_ppl = row_surprisal.mean().exp().item()
        timestamps_start = [(t * TOKEN_DURATION) for t in range(len(row_surprisal))]
        timestamps_end = [((t + 1) * TOKEN_DURATION) for t in range(len(row_surprisal))]
        timestamped_surprisal = list(zip(row_surprisal.tolist(), timestamps_start, timestamps_end)) # a list of (row_surprisal[idx], t_start, t_end)
        print(f"row surprisal: {len(row_surprisal)}")
        print(f"speech_tokens: {sample_meta['speech_token'].squeeze(0).size(0)}")
        sample_meta.update({'timestamped_surprisal' : timestamped_surprisal})
        batch_info.append(sample_meta)

    return batch_info

def get_losses(dataset, labels_dict, alignments_path, spk_emb_type, spk_emb_dict, limit=None):
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

    pbar = tqdm(dataset)
    sample_buffer = []

    if spk_emb_type == "native_retrieval":
        emb_arrays = []
        for spk, emb in spk_emb_dict.items(): 
            print(f"see what's inside the spk emb dict at each speaker: {torch.Tensor(emb).shape}") # emb should be (1, number of embeddings for the speaker)
            emb_arrays += emb

        emb_arrays = torch.Tensor(np.stack(emb_arrays)).to(device)
    
    for sample in pbar:
        if file_count >= lim:
            break

        # info
        audio_version = sample['audio_version']
        file_path = sample["path"]
        filename = sample["filename"]
        text = sample['text']
        pbar.set_description(f"Getting per phone losses for file: {filename}")

        # sample pre-processing
        canonical_text = text
        wav = load_wav(file_path, 16000)
        text_token, text_token_len = frontend._extract_text_token(canonical_text)
        speech_token, speech_token_len = frontend._extract_speech_token(wav)
        spk_embedding = frontend._extract_spk_embedding(wav)

        embedding = None

        if spk_emb_type == "default":
            embedding = spk_embedding
        elif spk_emb_type == "native_retrieval":
            sims = F.cosine_similarity(spk_embedding, emb_arrays, dim=1).cpu() # (1, ) * (40, 192)
            idx = int(torch.argmax(sims)) 
            embedding = emb_arrays[idx].unsqueeze(0)

        elif spk_emb_type == "domestic_retrieval":
            speaker_embs = spk_emb_dict.get(speaker)
            embs = []
            for file, emb in speaker_embs.items():
                if file == filename:
                    continue
                embs.append(emb)

            embedding = torch.Tensor(np.mean(embs, axis=0)).unsqueeze(0)
            
        #print(f"load_wav: {t1-t0:.3f}s | text_token: {t2-t1:.3f}s | speech_token: {t3-t2:.3f}s | embedding: {t4-t3:.3f}s")

        # get losses
        sample_buffer.append({
            'text_token': text_token,
            'speech_token': speech_token,
            'embedding': embedding,
            'audio_version' : audio_version,
            'file_path' : file_path,
            'filename' : filename,
            'text' : text
        })      

        if len(sample_buffer) == BATCH_SIZE:
            batch_losses = process_batch(sample_buffer, cosyvoice, device)                  # inference and get losses for batch
            for sample in batch_losses:
                for idx, loss in enumerate(sample['timestamped_surprisal']):
                    ppl_results.append({
                        'audio_version' : sample['audio_version'],
                        'file_path' : sample['file_path'],
                        'filename' : sample['filename'],
                        'token_id' : idx, 
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
                    'audio_version' : sample['audio_version'],
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

    # process dataset
    input_dataset = args.dataset_dir
    AUDIO_VERSION = "clean"
    TEXT_VERS = "sub"
    METADATA_PATH = "/home/ubuntu/speech_ppl/src/stim_final/stimuli_metadata_v3.csv"
    metadata = pd.read_csv(METADATA_PATH)

    processed = process_synth(input_dataset, AUDIO_VERSION, metadata, TEXT_VERS)
    processed_dataset = processed["processed"]
    print(f"Processed {len(processed_dataset)} samples.")

    NORM_DICT_DIR = f"{args.root_dir}/src/metrics/result_dicts"
    SPK_EMB_DIR = f"{args.root_dir}/src/metrics/"
    MODEL_TYPE = "COSYVOICE"
    MODEL_NAME = f"COSYVOICE"
    OUTPUT_DIR = args.output_dir
    SPK_EMB_TYPE = "default"

    csv_path = f"{OUTPUT_DIR}/{MODEL_TYPE}_{MODEL_NAME}_{AUDIO_VERSION}_{TEXT_VERS}_per_token_losses.csv"   

    if SPK_EMB_TYPE == "native_retrieval":
        dict_path = SPK_EMB_DIR + "libri_spk_dict.json"
        with open(dict_path, "r") as f:
            spk_emb_dict = json.load(f)
    elif SPK_EMB_TYPE == "domestic_retrieval":
        dict_path = SPK_EMB_DIR + "speechocean_spk_dict.json"
        with open(dict_path, "r") as f:
            spk_emb_dict = json.load(f)
    else:
        spk_emb_dict = None

    results = get_losses(
        dataset=processed_dataset, 
        labels_dict=metadata, 
        alignments_path=args.alignments_file, 
        spk_emb_type=SPK_EMB_TYPE,
        spk_emb_dict=spk_emb_dict,
        limit=None,
        )

    ppl_results = results["results"]

    with open(csv_path, "w") as f:
        fieldnames = ppl_results[0].keys()
        dict_writer = csv.DictWriter(f, fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(ppl_results)
    
    