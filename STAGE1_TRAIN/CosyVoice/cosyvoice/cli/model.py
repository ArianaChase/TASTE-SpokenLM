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
import torch
import time
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params}")
    return total_params

class CosyVoiceModel:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        llm_params = count_parameters(llm)
        print(f"llm_params: {llm_params}")
        self.flow = flow
        flow_params = count_parameters(flow)
        print(f"flow_params: {flow_params}")
        self.hift = hift
        hift_params = count_parameters(hift)
        print(f"hift_params: {hift_params}")

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), load_partial_list=[]) # to enable fully loaded state dict, set load_partial_list to empty list
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device))
        self.flow.to(self.device).eval()
        self.hift.load_state_dict(torch.load(hift_model, map_location=self.device))
        self.hift.to(self.device).eval()

    def inference(self, text, text_len, flow_embedding, llm_embedding=torch.zeros(0, 192),
                  audio_feat=None, audio_feat_len=None, words_index=None, asr_alignment=None, llm_kwargs={}, # for audio branch
                  whisper_text_token=None, whisper_text_token_len=None, # for audio branch with whisper token
                  prompt_text=torch.zeros(1, 0, dtype=torch.int32), prompt_text_len=torch.zeros(1, dtype=torch.int32),
                  llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32), llm_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
                  flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32), flow_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
                  prompt_speech_feat=torch.zeros(1, 0, 80), prompt_speech_feat_len=torch.zeros(1, dtype=torch.int32)):
        cur_time = time.time()
        if llm_kwargs.get("use_target_speech_token", False):
            print("Directly use target speech unit (s3 token) for speech generation. Skip llm speech decoder")
            tts_speech_token = llm_prompt_speech_token
        else:
            print(type(self.llm))
            result = self.llm.inference(
                text=text.to(self.device),
                text_len=text_len.to(self.device),
                prompt_text=prompt_text.to(self.device),
                prompt_text_len=prompt_text_len.to(self.device),
                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                prompt_speech_token_len=llm_prompt_speech_token_len.to(self.device),
                embedding=llm_embedding.to(self.device),
                audio_feat=None if audio_feat == None else audio_feat.to(self.device),
                audio_feat_len=None if audio_feat_len == None else audio_feat_len.to(self.device),
                words_index=None if words_index == None else words_index,
                whisper_text_token = None if whisper_text_token == None else whisper_text_token.to(self.device),
                whisper_text_token_len = None if whisper_text_token_len == None else whisper_text_token_len.to(self.device),
                asr_alignment=asr_alignment,
                **llm_kwargs,
            )
            print(result)
            tts_speech_token = result[0]
            ppl_results = result[1]
        
        print('llm time:', time.time()-cur_time)
        print(f"text shape: {text.shape}")
        print(f"text len shape: {text_len.shape}")
        print(f"prompt text shape: {prompt_text.shape}")
        print(f"prompt text len shape: {prompt_text_len.shape}")
        print(f"prompt speech token shape: {llm_prompt_speech_token.shape}")
        print(f"prompt speech token len shape: {llm_prompt_speech_token_len.shape}")
        print(f"tts speech token shape: {tts_speech_token.shape}")
        cur_time = time.time()
        tts_mel = self.flow.inference(token=tts_speech_token,
                                      token_len=torch.tensor([tts_speech_token.size(1)], dtype=torch.int32).to(self.device),
                                      prompt_token=flow_prompt_speech_token.to(self.device),
                                      prompt_token_len=flow_prompt_speech_token_len.to(self.device),
                                      prompt_feat=prompt_speech_feat.to(self.device),
                                      prompt_feat_len=prompt_speech_feat_len.to(self.device),
                                      embedding=flow_embedding.to(self.device))
        print('flow time:', time.time()-cur_time)
        cur_time = time.time()
        tts_speech = self.hift.inference(mel=tts_mel).cpu()
        print('hift time:', time.time()-cur_time)
        torch.cuda.empty_cache()
        return {'tts_speech': tts_speech, 'ppl_results': ppl_results}
