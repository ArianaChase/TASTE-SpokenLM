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

from functools import partial
import onnxruntime
import torch
import torchaudio
import numpy as np
import whisper
from typing import Callable, Optional, List
import torchaudio.compliance.kaldi as kaldi
import torchaudio
import os
import re
import inflect
use_ttsfrd = False
from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph
from transformers import WhisperTokenizerFast

#onnxruntime.set_default_logger_severity(1)

class CosyVoiceFrontEnd:

    def __init__(
        self,
        get_tokenizer: Callable,
        feat_extractor: Callable,
        campplus_model: str,
        speech_tokenizer_model: str,
        spk2info: str = '',
        instruct: bool = False,
        allowed_special: str = 'all',
        audio_extractor: Optional = None,
        whisper_tokenizer_dir: Optional = None,
        pre_asr: bool = False,
        pre_asr_fpath: Optional = None,
        tokenize: Optional = None,
        tokenize_whisper: Optional = None,
    ):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("DEVICE: ", self.device)
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option, providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"])
        self.speech_tokenizer_session = onnxruntime.InferenceSession(speech_tokenizer_model, sess_options=option, providers=["CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"])

        print("CAMPLUS:", self.campplus_session.get_providers())
        print("SPEECH:", self.speech_tokenizer_session.get_providers())


        if os.path.exists(spk2info):
            self.spk2info = torch.load(spk2info, map_location=self.device)
        self.instruct = instruct
        self.allowed_special = allowed_special
        self.inflect_parser = inflect.engine()
        self.use_ttsfrd = use_ttsfrd
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            assert self.frd.initialize('{}/../../pretrained_models/CosyVoice-ttsfrd/resource'.format(ROOT_DIR)) is True, 'failed to initialize ttsfrd resource'
            self.frd.set_lang_type('pinyin')
            self.frd.enable_pinyin_mix(True)
            self.frd.set_breakmodel_index(1)
        else:
            pass
        self.audio_extractor = audio_extractor

        self.pre_asr = pre_asr
        self.pre_asr_fpath = pre_asr_fpath
        if self.pre_asr:
            from transformers import pipeline
            rtslm_storage_dir = os.getenv("RTSLM_STORAGE_DIR")
            self.asr_pipe = pipeline(
                'automatic-speech-recognition',
                # model=f'{rtslm_storage_dir}/pretrained_models/whisper-large-v3',
                model=f'{rtslm_storage_dir}/pretrained_models/distil-whisper-large-v3',
                torch_dtype=torch.float16,
                device=self.device,
                chunk_length_s=30,
            )
            if self.pre_asr_fpath is not None:
                import json
                with open(self.pre_asr_fpath, 'r') as jfr:
                    self.pre_asr_result = json.load(jfr)
                print(f"Will directly use pre-asr result from {self.pre_asr_fpath} to save time!")
        
        if whisper_tokenizer_dir: # != ('' or None); hard coded, doesnot allow toggling add_bos and add_eos
            self.whisper_tokenizer = WhisperTokenizerFast.from_pretrained(
                whisper_tokenizer_dir,
            )

            forced_decoder_ids = self.whisper_tokenizer.get_decoder_prompt_ids(
                task = 'transcribe',
                language = 'en',
                no_timestamps = True,
            ) # TODO: Make this controllable. # TODO: make `add_bos` and `add_eos` available. Now: use bos and eos
            print(f"Whisper tokenizer loadded. forced_decoder_ids={forced_decoder_ids}. Will add bos and eos")

            _prefix_tokens = self.whisper_tokenizer.prefix_tokens
            self.prefix_token_to_wrap  = _prefix_tokens
            self.postfix_token_to_wrap = [self.whisper_tokenizer.eos_token_id]
            self.eos_index = self.whisper_tokenizer.eos_token_id
        
        self.tokenize = tokenize
        self.tokenize_whisper = tokenize_whisper 
    
    def _extract_audio_feat(self, speech): # speech should be 16kHz
        if self.audio_extractor == None:
            print("Does not have audio extractor. Please make sure to pass the kwarg: audio_extractor during init")
            return None, None
        waveform = speech
        waveform_length = [waveform.shape[-1]]
        feat, feat_len = self.audio_extractor(waveform, waveform_length)
        return feat, feat_len            

    def _extract_text_token(self, text):
        text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
        return text_token, text_token_len
    
    def _extract_whisper_text_token_by_words(self, text: str, chunks: List, use_eos_as_bos=False):
        data = [
            {
                'text': text,
                'asr_chunks': chunks
            }
        ]
        new_data = [d for d in self.tokenize_whisper(self.tokenize(data))][0]
        text_token, whisper_text_token, words_begin_index, words_end_index = (
            new_data['text_token'],
            new_data['whisper_text_token'],
            new_data['words_begin_index'],
            new_data['words_end_index'],
        )
        if use_eos_as_bos:
            whisper_text_token[0] = self.eos_index
        text_token_len, whisper_text_token_len = len(text_token), len(whisper_text_token)
        
        text_token = torch.tensor([text_token]).to(self.device)
        text_token_len = torch.tensor([text_token_len], dtype=torch.int32).to(self.device)
        whisper_text_token = torch.tensor([whisper_text_token]).to(self.device)
        whisper_text_token_len = torch.tensor([whisper_text_token_len], dtype=torch.int32).to(self.device)
        
        words_index = []
        for t1, t2 in zip(words_begin_index, words_end_index):
            if t2 - t1 > 1:
                words_index.append((0, t1, t2))

        return text_token, text_token_len, whisper_text_token, whisper_text_token_len, words_index

    def _extract_speech_token(self, speech):
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        try:
            speech_token = self.speech_tokenizer_session.run(None, {self.speech_tokenizer_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                                                                self.speech_tokenizer_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        except:
            speech_token = [1, 1, 1]
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, speech, cutoff_threshold=None):
        if cutoff_threshold is not None:
            assert isinstance(cutoff_threshold, float) and cutoff_threshold <= 30.0, f"please set the cutoff_threshold to be a float number."
            print(f"spk_emb_cutoff_threshold is set. will cutoff the input speech to {cutoff_threshold} secs.")
            _cutoff_index = int(cutoff_threshold*16000)
            speech = speech[..., :_cutoff_index].clone()
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None, {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, speech):
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len

    def text_normalize(self, text, split=True):
        text = text.strip()
        if contains_chinese(text):
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "、")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            text = re.sub(r'[，,]+$', '。', text)
            texts = [i for i in split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=80,
                                                token_min_n=60, merge_len=20,
                                                comma_split=False)]
        else:
            text = spell_out_number(text, self.inflect_parser)
            texts = [i for i in split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "en", token_max_n=80,
                                                token_min_n=60, merge_len=20,
                                                comma_split=False)]
        if split is False:
            return text
        return texts

    def frontend_sft(self, tts_text, spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        embedding = self.spk2info[spk_id]['embedding']
        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len, 'llm_embedding': embedding, 'flow_embedding': embedding}
        return model_input

    def frontend_zero_shot(self, tts_text, prompt_text, prompt_speech_16k):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(prompt_speech_16k)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_22050)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len,
                       'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                       'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                       'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                       'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                       'llm_embedding': embedding, 'flow_embedding': embedding}
        return model_input

    def frontend_cross_lingual(self, tts_text, prompt_speech_16k):
        model_input = self.frontend_zero_shot(tts_text, '', prompt_speech_16k)
        # in cross lingual mode, we remove prompt in llm
        del model_input['prompt_text']
        del model_input['prompt_text_len']
        del model_input['llm_prompt_speech_token']
        del model_input['llm_prompt_speech_token_len']
        return model_input

    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        model_input = self.frontend_sft(tts_text, spk_id)
        # in instruct mode, we remove spk_embedding in llm due to information leakage
        del model_input['llm_embedding']
        instruct_text_token, instruct_text_token_len = self._extract_text_token(instruct_text + '<endofprompt>')
        model_input['prompt_text'] = instruct_text_token
        model_input['prompt_text_len'] = instruct_text_token_len
        return model_input

    def _get_alignment(self, audio_16k, audio_feat_len, audio_fpath=None, allowed_special='all', strip_first=False):
        audio_16k = audio_16k.numpy()[0]
        audio_length = audio_16k.shape[0] / 16000
        if self.pre_asr_fpath is not None:
            if audio_fpath is not None:
                fid = audio_fpath.split('/')[-1].split('.')[0]
                result = self.pre_asr_result[fid]
            else:
                print(f"Pre-asr fpath is set={self.pre_asr_fpath}. but audio_fpath is not passed. this will waste a lot of time re-asr!")
        else:
            result = self.asr_pipe(
                audio_16k,
                return_timestamps='word',
                generate_kwargs={
                    'language': 'english',
                    'forced_decoder_ids': None,
                    'task': 'transcribe'
                },
                batch_size=1,
            )
        chunks = [
            {'word': x['text'],
            'range': (
                x['timestamp'][0] / audio_length,
                x['timestamp'][1] / audio_length
            )}
            for x in result['chunks']
        ]
        asr_token = []
        alignment = []
        for i, d in enumerate(chunks):
            if strip_first and i == 0:
                d['word'] = d['word'].strip()
            segmented_asr_token = self.tokenizer.encode(d['word'], allowed_special=allowed_special)
            if audio_feat_len:
                left, right = [int(x * audio_feat_len) for x in d['range']]
                for _ in range(len(segmented_asr_token)):
                    alignment.append([left, right])
            asr_token += segmented_asr_token
        
        text_token = torch.tensor([asr_token], dtype=torch.int32).to(self.device)
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
        if audio_feat_len:
            alignment = torch.tensor([alignment], dtype=torch.int32).to(self.device)
        return text_token, text_token_len, alignment, chunks

    # add frontend for audio
    def frontend_audio(
        self, 
        text: str, # transcript of the audio fpath
        audio_16k,
        audio_fpath=None,
        extract_target_speech_token: bool = False,
        extract_whisper_text_token_by_words: bool = False,
        pre_asr: bool = False,
        spk_emb_cutoff_threshold: int = None,
        no_spk_emb: bool = False,
    ):
        '''
            return model_input:
            keys: [
                'audio_feat', 'audio_feat_len', 'text', 'text_len', 'llm_embedding', 'flow_embedding', 
                'word_index'
            ]
        '''
        # get audio_feat
        audio_feat, audio_feat_len = self._extract_audio_feat(audio_16k)

        if not self.pre_asr:
            # get text (text, text_len)
            text_token, text_token_len = self._extract_text_token(text) # currently not allow word-level tokenization
            asr_alignment = None
        else:
            text_token, text_token_len, asr_alignment, chunks = self._get_alignment(audio_16k, audio_feat_len, audio_fpath=audio_fpath)
            _words = [d['word'] for d in chunks]
            asr_text = ''.join(_words)
            if audio_feat_len == None:
                print("No audio_feat found, discard alignment.")
                text_token, text_token_len = self._extract_text_token(asr_text)
            print(f"Pre-asr text: |{asr_text}|")

        # TODO: retreive text_token word-by-word and get word_index if required
        # get embeddings
        if not no_spk_emb:
            embedding = self._extract_spk_embedding(audio_16k, cutoff_threshold=spk_emb_cutoff_threshold)
        else:
            print(f"`no_spk_emb` is set to True. will use zero vector as embedding.")
            embedding = torch.zeros((1, 192)).to(self.device)
        

        # audio_feat = torch.zeros_like(audio_feat) # NOTE: for testing only, please use cautiously.
        model_input = {
            'text': text_token,
            'text_len': text_token_len,
            'audio_feat': audio_feat,
            'audio_feat_len': audio_feat_len,
            'llm_embedding': embedding,
            'flow_embedding': embedding,
            'words_index': None, # TODO: add word_index if required
        }
        if asr_alignment is not None:
            model_input['asr_alignment'] = asr_alignment

        # extract speech tokens for calculating accuracy
        if extract_target_speech_token:
            speech_token, speech_token_len = self._extract_speech_token(audio_16k)
            # use llm_prompt_speech_token for passing though cosyvoice model inference mode seamlessly
            model_input['llm_prompt_speech_token'] = speech_token
            model_input['llm_prompt_speech_token_len'] = speech_token_len
        if extract_whisper_text_token_by_words:
            print("Use tokenize by words. Will completely use the original tokenize functions to avoid mismatch")
            print("Will use real bos")
            text_token, text_token_len, whisper_text_token, whisper_text_token_len, words_index = self._extract_whisper_text_token_by_words(
                text, chunks
            )
            model_input.update(
                {
                    'text': text_token,
                    'text_len': text_token_len,
                    'whisper_text_token': whisper_text_token,
                    'whisper_text_token_len': whisper_text_token_len,
                    'words_index': words_index,
                }
            )

        return model_input


