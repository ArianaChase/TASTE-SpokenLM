
from typing import Dict, Optional, Callable, List, Generator

import torch
from torch import nn
from cosyvoice.llm.llm import Qwen2LM, th_accuracy, IGNORE_ID


class TasteS3GenerationLM(Qwen2LM):
    def __init__(
        self,
        llm_input_size: int,
        llm_output_size: int,
        speech_token_size: int,
        llm: torch.nn.Module,
        sampling: Callable,
        # = new =
        taste_tokenizer: torch.nn.Module,
        taste_decoder_mixer: torch.nn.Module,
        # =======
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        mix_ratio: List[int] = [5, 15],
        
    ):
        super().__init__(
            llm_input_size=llm_input_size,
            llm_output_size=llm_output_size,
            speech_token_size=speech_token_size,
            llm=llm,
            sampling=sampling,
            length_normalized_loss=length_normalized_loss,
            lsm_weight=lsm_weight,
            mix_ratio=mix_ratio,
        )
        self.taste_tokenizer = taste_tokenizer
        self.taste_decoder_mixer = taste_decoder_mixer
        self.weight_commit_loss = 1.0
        self.is_text_only = (taste_tokenizer is None) or (taste_decoder_mixer is None)

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text_token: (B, L)
            text_token_len: (B,)
            audio_feature: 
            audio_feature_len: 
            speech_token: (B, T)
            speech_token_len: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)

        audio_feature = batch['audio_feature'].to(device)
        audio_feature_len = batch['audio_feature_len'].to(device)

        # 1-1. encode text_token
        text_token_emb = self.llm.model.model.embed_tokens(text_token)

        if not self.is_text_only:
            # 1-2. encode taste_token
            tokenized = self.taste_tokenizer(text_token, text_token_len, audio_feature, audio_feature_len)
            taste_token_emb = tokenized['taste_token_emb']

            # 1-3. mixing
            mixed_token_emb = self.taste_decoder_mixer(text_token_emb, taste_token_emb, text_token_len)
        else:
            mixed_token_emb = text_token_emb

        # 2. encode speech_token
        speech_token_emb = self.speech_embedding(speech_token)

        # 3. prepare llm_input/target
        lm_target, lm_input, lm_input_len = self.prepare_lm_input_target(text_token, mixed_token_emb, text_token_len, speech_token, speech_token_emb, speech_token_len)
        lm_target = lm_target.to(device)

        # 4. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target.to(device))
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 3), lm_target, ignore_label=IGNORE_ID)

        if not self.is_text_only and 'commit_loss' in tokenized:
            loss += self.weight_commit_loss * tokenized['commit_loss']

        return {'loss': loss, 'acc': acc}

    @torch.inference_mode()
    def inference(
        self,
        text_token: torch.Tensor,
        text_token_len: torch.Tensor,
        audio_feature: torch.Tensor,
        audio_feature_len: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        uuid: str = '',
        **kwargs,
    ) -> Generator[torch.Tensor, None, None]:

        # device = text_token.device

        text_token_emb = self.llm.model.model.embed_tokens(text_token)

        if not self.is_text_only:
            # 1-2. encode taste_token
            tokenized = self.taste_tokenizer(text_token, text_token_len, audio_feature, audio_feature_len)
            taste_token_emb = tokenized['taste_token_emb']

            print(text_token_emb.shape, taste_token_emb.shape)
            # 1-3. mixing
            mixed_token_emb = self.taste_decoder_mixer(text_token_emb, taste_token_emb, text_token_len)
        else:
            mixed_token_emb = text_token_emb

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        lm_input = torch.concat([sos_eos_emb, mixed_token_emb, task_id_emb], dim=1)

        # 4. cal min/max_length
        min_len = int(text_token_len * min_token_text_ratio)
        max_len = int(text_token_len * max_token_text_ratio)

        # 5. step by step decode
        for token in self.inference_wrapper(lm_input, sampling, min_len, max_len, uuid):
            yield token
            
    @torch.inference_mode()
    def inference_for_taste_emb(
        self,
        text_token: torch.Tensor,
        text_token_len: torch.Tensor,
        taste_token_emb: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        uuid: str = '',
        **kwargs,
    ):
        text_token_emb = self.llm.model.model.embed_tokens(text_token)
        
        if not self.is_text_only:
            # 1-3. mixing
            
            print(text_token_emb.shape, taste_token_emb.shape)
            mixed_token_emb = self.taste_decoder_mixer(text_token_emb, taste_token_emb, text_token_len)
        else:
            mixed_token_emb = text_token_emb
        
        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        lm_input = torch.concat([sos_eos_emb, mixed_token_emb, task_id_emb], dim=1)

        # 4. cal min/max_length
        min_len = int(text_token_len * min_token_text_ratio)
        max_len = int(text_token_len * max_token_text_ratio)

        # 5. step by step decode
        for token in self.inference_wrapper(lm_input, sampling, min_len, max_len, uuid):
            yield token

    @torch.inference_mode()
    def inference_bistream(
            self,
            text: Generator,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        raise NotImplementedError
