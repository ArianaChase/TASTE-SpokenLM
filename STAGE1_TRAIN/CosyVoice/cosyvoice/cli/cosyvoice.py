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
import os
import torch
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel

class CosyVoice:

    def __init__(self, model_dir, config_fpath=None, llm_fpath=None, flow_fpath=None, hift_fpath=None, pre_asr=False, pre_asr_fpath=None, whisper_tokenizer_dir=None):
        instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            print('Downloading model from modelscope')
            model_dir = snapshot_download(model_dir)
        self.config_fpath = config_fpath if config_fpath != None else '{}/cosyvoice.yaml'.format(model_dir)
        with open(self.config_fpath, 'r') as f:
            print('Loading configs')
            configs = load_hyperpyyaml(f)
        self.frontend = CosyVoiceFrontEnd(
            configs['get_tokenizer'],
            configs['feat_extractor'],
            '{}/campplus.onnx'.format(model_dir),
            '{}/speech_tokenizer_v1.onnx'.format(model_dir),
            '{}/spk2info.pt'.format(model_dir),
            instruct,
            configs['allowed_special'],
            configs.get('audio_extractor', None),
            whisper_tokenizer_dir,
            pre_asr=pre_asr,
            pre_asr_fpath=pre_asr_fpath,
            tokenize=configs.get('tokenize', None), 
            tokenize_whisper=configs.get('tokenize_whisper', None),
        )
        print("Frontend loaded")
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'])
        print("Model Initialized")
        self.llm_fpath = llm_fpath if llm_fpath != None else '{}/llm.pt'.format(model_dir)
        self.flow_fpath = flow_fpath if flow_fpath != None else '{}/flow.pt'.format(model_dir)
        self.hift_fpath = hift_fpath if hift_fpath != None else '{}/hift.pt'.format(model_dir)
        self.model.load(self.llm_fpath,
                        self.flow_fpath,
                        self.hift_fpath)
        print("Model loaded")
        del configs

    def list_avaliable_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def inference_sft(self, tts_text, spk_id):
        tts_speeches = []
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_sft(i, spk_id)
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output['tts_speech'])
        return {'tts_speech': torch.concat(tts_speeches, dim=1)}

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)
        tts_speeches = []
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k)
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output['tts_speech'])
        return {'tts_speech': torch.concat(tts_speeches, dim=1)}

    def inference_cross_lingual(self, tts_text, prompt_speech_16k):
        if self.frontend.instruct is True:
            raise ValueError('{} do not support cross_lingual inference'.format(self.model_dir))
        tts_speeches = []
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k)
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output['tts_speech'])
        return {'tts_speech': torch.concat(tts_speeches, dim=1)}

    def inference_instruct(self, tts_text, spk_id, instruct_text):
        if self.frontend.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize(instruct_text, split=False)
        tts_speeches = []
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output['tts_speech'])
        return {'tts_speech': torch.concat(tts_speeches, dim=1)}

    def inference_audio(self, text, audio_16k, 
        audio_fpath=None,
        normalize_text=True, 
        extract_target_speech_token=False, 
        adopt_teacher_forcing_for_test=False, 
        extract_whisper_text_token_by_words=False,
        sampling=25,
        drop_eos_before_llm=False,
        spk_emb_cutoff_threshold=None,
        no_spk_emb=False,
        use_target_speech_token=False,
    ):
        # TODO: add inference by audio
        # currently only support non-split text input
        if normalize_text:
            normalized_text = self.frontend.text_normalize(text, split=False)
            print(f"normalized_text: {normalized_text}")
        else:
            normalized_text = text
        model_input = self.frontend.frontend_audio(
            normalized_text, audio_16k, 
            audio_fpath=audio_fpath,
            extract_target_speech_token=extract_target_speech_token, 
            extract_whisper_text_token_by_words=extract_whisper_text_token_by_words,
            spk_emb_cutoff_threshold=spk_emb_cutoff_threshold,
            no_spk_emb=no_spk_emb,
        )
        if use_target_speech_token:
            print("[CAUTIOUS]: will use target speech unit for speech reconstruction. This will generate topline result. Please be aware of it.")
        model_input['llm_kwargs'] = {
            'adopt_teacher_forcing_for_test': adopt_teacher_forcing_for_test,
            'sampling': sampling,
            'drop_eos_before_llm': drop_eos_before_llm,
            'use_target_speech_token': use_target_speech_token,
        }
        print(f"llm_kwargs: {model_input['llm_kwargs']}")
        model_output = self.model.inference(**model_input)

        print(model_output)
        return model_output