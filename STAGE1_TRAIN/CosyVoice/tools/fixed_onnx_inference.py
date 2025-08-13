#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import torch
import torchaudio
from tqdm import tqdm
import onnxruntime
import torchaudio.compliance.kaldi as kaldi
import multiprocessing as mp
import os
import psutil

ort_session = None
process_cpu_id = None

def initialize_ort_session(onnx_path, cpu_assignments):
    global ort_session, process_cpu_id
    
    # 獲取當前進程在池中的索引
    current_process = mp.current_process()
    process_name = current_process.name
    
    # 從進程名稱中提取索引 (例如: ForkPoolWorker-1 -> 0)
    if 'ForkPoolWorker-' in process_name:
        worker_index = int(process_name.split('-')[1]) - 1
    else:
        worker_index = 0
    
    # 分配 CPU 核心
    if worker_index < len(cpu_assignments):
        process_cpu_id = cpu_assignments[worker_index]
    else:
        process_cpu_id = cpu_assignments[worker_index % len(cpu_assignments)]
    
    # 綁定進程到指定的 CPU 核心
    try:
        p = psutil.Process(os.getpid())
        p.cpu_affinity([process_cpu_id])
        print(f"Process {current_process.pid} ({process_name}) bound to CPU {process_cpu_id}")
    except Exception as e:
        print(f"Warning: Failed to bind process {current_process.pid} to CPU {process_cpu_id}: {e}")
    
    # 設置 ONNX Runtime session 選項
    option = onnxruntime.SessionOptions()
    option.inter_op_num_threads = 1
    option.intra_op_num_threads = 1
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 創建 CPU 執行提供者，並設置親和性
    cpu_provider_options = {
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cpu_allocator': 'arena',
    }
    
    providers = [("CPUExecutionProvider", cpu_provider_options)]
    
    # 設置環境變量來限制 ONNX Runtime 使用的 CPU
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    ort_session = onnxruntime.InferenceSession(onnx_path, sess_options=option, providers=providers)
    print(f"Process {current_process.pid}: ONNX session initialized and bound to CPU {process_cpu_id}")

def get_single_embedding(utt_wav_pair):
    global ort_session, process_cpu_id
    utt, wav = utt_wav_pair
    
    try:
        # 確保當前線程仍然綁定到正確的 CPU
        if process_cpu_id is not None:
            try:
                p = psutil.Process(os.getpid())
                current_affinity = p.cpu_affinity()
                if process_cpu_id not in current_affinity:
                    p.cpu_affinity([process_cpu_id])
            except:
                pass  # 靜默處理錯誤
        
        audio, sample_rate = torchaudio.load(wav)
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
        
        feat = kaldi.fbank(audio,
                            num_mel_bins=80,
                            dither=0,
                            sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        
        # 確保 ort_session 已初始化
        if ort_session is None:
            raise RuntimeError(f"ONNX session not initialized in process {mp.current_process().pid}")
        
        embedding = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        return (utt, embedding)
    
    except Exception as e:
        print(f"Error processing {utt} on CPU {process_cpu_id}: {str(e)}")
        return (utt, None)  # 返回 None 表示處理失敗

def multiprocess_inference(utt_wav_pairs, onnx_path, num_processes=8):
    print(f"Starting multiprocess inference with {num_processes} processes")
    
    # 獲取系統 CPU 核心數量
    total_cpus = psutil.cpu_count(logical=True)
    print(f"System has {total_cpus} CPU cores")
    
    # 為每個進程分配專屬的 CPU 核心
    if num_processes > total_cpus:
        print(f"Warning: Requested {num_processes} processes but only {total_cpus} CPU cores available")
        cpu_assignments = list(range(total_cpus)) * ((num_processes // total_cpus) + 1)
        cpu_assignments = cpu_assignments[:num_processes]
    else:
        cpu_assignments = list(range(num_processes))
    
    print(f"CPU assignments: {cpu_assignments}")
    
    # 創建進程池，並傳遞 CPU 分配信息
    with mp.Pool(processes=num_processes, 
                 initializer=initialize_ort_session, 
                 initargs=(onnx_path, cpu_assignments)) as pool:
        results = list(tqdm(pool.imap_unordered(get_single_embedding, utt_wav_pairs), 
                           total=len(utt_wav_pairs)))
    
    # 過濾掉失敗的結果
    successful_results = [(utt, emb) for utt, emb in results if emb is not None]
    failed_count = len(results) - len(successful_results)
    
    if failed_count > 0:
        print(f"Warning: {failed_count} utterances failed to process")
    
    return successful_results

def main(args):
    print(f"Loading data from directory: {args.dir}")
    
    # 在主進程中也設置環境變量，避免影響子進程
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['MKL_NUM_THREADS'] = str(args.threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(args.threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(args.threads)
    
    utt2wav, utt2spk = {}, {}
    
    # 讀取 wav.scp
    wav_scp_path = f'{args.dir}/wav.scp'
    print(f"Reading {wav_scp_path}")
    with open(wav_scp_path) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]
    print(f"Loaded {len(utt2wav)} utterances")
    
    # 讀取 utt2spk
    utt2spk_path = f'{args.dir}/utt2spk'
    print(f"Reading {utt2spk_path}")
    with open(utt2spk_path) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2spk[l[0]] = l[1]
    print(f"Loaded {len(utt2spk)} utterance-speaker mappings")

    # 執行多進程推理
    print(f"Starting inference with ONNX model: {args.onnx_path}")
    utt2embedding, spk2embedding = {}, {}
    utt_embedding_pairs = multiprocess_inference(utt2wav.items(), args.onnx_path, num_processes=args.threads)
    
    # 處理結果
    print("Processing results...")
    for utt, embedding in utt_embedding_pairs:
        utt2embedding[utt] = embedding
        spk = utt2spk[utt]
        if spk not in spk2embedding:
            spk2embedding[spk] = []
        spk2embedding[spk].append(embedding)
    
    # 計算說話者平均嵌入
    print("Computing speaker embeddings...")
    for k, v in spk2embedding.items():
        spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()

    # 保存結果
    utt_output_path = f'{args.dir}/utt2embedding.pt'
    spk_output_path = f'{args.dir}/spk2embedding.pt'
    
    print(f"Saving utterance embeddings to: {utt_output_path}")
    torch.save(utt2embedding, utt_output_path)
    
    print(f"Saving speaker embeddings to: {spk_output_path}")
    torch.save(spk2embedding, spk_output_path)
    
    print(f"Processing complete!")
    print(f"- Processed {len(utt2embedding)} utterances")
    print(f"- Generated embeddings for {len(spk2embedding)} speakers")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-process ONNX inference for speaker embeddings")
    parser.add_argument('--dir',
                        type=str,
                        required=True,
                        help='Directory containing wav.scp and utt2spk files')
    parser.add_argument('--onnx_path',
                        type=str,
                        required=True,
                        help='Path to ONNX model file')
    parser.add_argument('--threads', 
                        type=int, 
                        default=32,
                        help='Number of parallel processes (default: 32)')
    args = parser.parse_args()
    main(args)