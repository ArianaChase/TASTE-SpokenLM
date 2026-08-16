from cosyvoice.llm.llm import TransformerLM # the text-only llm component in cosyvoice
from typing import Dict, Optional, Union, List, Tuple
import torch
import logging
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from cosyvoice.utils.class_utils import RTSLM_FUSION_CLASSES
from cosyvoice.utils.common import IGNORE_ID, th_accuracy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformerJointLM(TransformerLM):
    def __init__(
            self,
            text_encoder_input_size: int,
            llm_input_size: int,
            llm_output_size: int,
            text_token_size: int,
            speech_token_size: int,
            text_encoder: torch.nn.Module,
            llm: torch.nn.Module,
            # ------ NOTE: Below is the added args for the audio branch ------ #
            audio_embed_dim: int, # the size of the audio quantized vector (added)
            audio_token_encoder_input_size: int, # the size of audio token encoder input (added)
            audio_tokenizer: torch.nn.Module, # the audio subword / word level tokenizer
            audio_token_encoder: torch.nn.Module, # the audio token encoder (added)
            # ------ NOTE: End of added args --------------------------------- #
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            spk_embed_dim: int = 192,
            load_partial_list: Optional[List[str]] = [],
            freeze_partial_list: Optional[List[str]] = [],
            freeze_partial_during_init: bool = False,
            # ------ NOTE: Below is the added kwargs for the audio branch ---- #
            is_word_level: bool = False, # whether to use word-level audio token (English only)
            drop_eos_before_llm: bool = False, # whether to drop the last encoded repr. before llm
            fuse_encoded_audio_text_type: str = 'concat',
            fuse_encoded_audio_text_kwargs: Dict = {},
            quantized_loss_weight: float = 1.0,
            # ------ NOTE: End of the added kwargs --------------------------- #
    ):
        super().__init__(
            text_encoder_input_size,
            llm_input_size,
            llm_output_size,
            text_token_size,
            speech_token_size,
            text_encoder,
            llm,
            length_normalized_loss,
            lsm_weight,
            spk_embed_dim,
            load_partial_list,
            freeze_partial_list,
            freeze_partial_during_init=False, # avoid freezing partialy before all the modules have loaded
        )
        # What we have:
        # 1. text_encoder
        # 2. llm
        # 3. llm_decoder, a.k.a. the linear layer that maps llm_output to speech_token_size + 1 (including EOS)
        # 4. criterion_ce, a.k.a. the label smoothing loss
        # 5. speech_embedding (speech token embedding in lm_input_size)
        # 6. spk_embed_affine_layer, a linear layer that maps speaker embedding to llm_input_size
        #-----------------------------------------
        # What we need to add:
        # 1. audio tokenizer (including the audio encoder, audio segmneter, and audio quantizer)
        # 2. audio_embedding (from audio token size -> audio encoder input size) # can be triggered 
        self.audio_tokenizer = audio_tokenizer
        self.audio_embed_affine_layer = nn.Linear(audio_embed_dim, audio_token_encoder_input_size)
        self.audio_token_encoder = audio_token_encoder
        self.audio_token_encoder_affine_layer = nn.Linear(
            self.audio_token_encoder.output_size(), 
            llm_input_size,
        )
        self.is_word_level = is_word_level
        self.drop_eos_before_llm = drop_eos_before_llm
        if self.drop_eos_before_llm:
            logging.warning("Audio LLM will drop eos before tts llm (including audio/text encoder). Please ensure you have prepare eos during tokenization.")
        self.fuse_encoded_audio_text_type = fuse_encoded_audio_text_type
        self.fuse_encoded_audio_text_module = RTSLM_FUSION_CLASSES[fuse_encoded_audio_text_type](
            **fuse_encoded_audio_text_kwargs
        )
        if len(freeze_partial_list) > 0 and freeze_partial_during_init:
            self.partial_frozen_list = self.freeze_parameters_by_partial_list(self.freeze_partial_list)         
        self.quantized_loss_weight = quantized_loss_weight
    
    def encode_audio(
            self,
            audio: torch.Tensor,
            audio_lengths: torch.Tensor,
    ):
        # print(f"Audio Lengths: {audio_lengths}")
        audio = self.audio_embed_affine_layer(audio) # map audio_tokenizer feat size -> audio_token_encoder feat size
        encoder_out, encoder_mask = self.audio_token_encoder(audio, audio_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.audio_token_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens
    
    def forward(
            self,
            batch: dict,
            device: torch.device,
            return_logits: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        
        if self.drop_eos_before_llm:
            # assert eos is added. so should not produce error
            text_token = text_token[:, :-1] # NOTE: it's ok not to make the last tokens to be paddings. This would not affect the after encoding. 
            text_token_len = text_token_len - 1

        # add audio branch
        audio_feat = batch['audio_feat'].to(device)
        audio_feat_len = batch['audio_feat_len'].to(device)
        word_ids = None
        if self.is_word_level:
            words_index = batch['words_index'] # List of tuples: [(b, t1, t2), ...]
            if words_index == None:
                word_ids = batch['word_ids'].to(device)
        # end 
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 0. encode text_token
        text_token_embed = self.text_embedding(text_token)
        text_token_encoded, text_token_len = self.encode(text_token_embed, text_token_len)
        
        # 1. extract audio_token and encode (audio branch)
        text_token_for_audio = text_token.detach()
        text_token_embed_for_audio = text_token_embed.detach() # detach to prevent gradient from flowing back 
        asr_alignment = batch.get('asr_alignment', None)
        if asr_alignment != None:
            asr_alignment = asr_alignment.to(device).detach()
        whisper_text_token = batch.get("whisper_text_token", None)
        whisper_text_token_len = batch.get("whisper_text_token_len", None)
        # print(f"Whisper_text_token_shape: {whisper_text_token.shape}")
        # print(f"Whisper_text_token_len: {whisper_text_token_len}")
        if whisper_text_token != None:
            whisper_text_token = whisper_text_token.to(device)
            whisper_text_token_len = whisper_text_token_len.to(device)
        audio_tokenized_results = self.audio_tokenizer(
            audio_feat, # required
            audio_feat_len, # required 
            text_token_for_audio, #required
            text_token_embed_for_audio, # required
            text_token_len, # required
            asr_alignment,
            # kwargs
            words_index = None if not self.is_word_level else words_index, 
            word_ids = word_ids,
            whisper_text_token = whisper_text_token,
            whisper_text_token_len =  whisper_text_token_len,
        )
        quantized_results = audio_tokenized_results['quantized_results']
        audio_quantized_feats, audio_quantized_feats_len = quantized_results['quantized_feats'], quantized_results['quantized_feats_lengths']
        if self.drop_eos_before_llm:
            # drop eos which is the last token after audio tower
            audio_quantized_feats = audio_quantized_feats[:, :-1, :] # NOTE: it's ok not to make the last representations to be paddings. This would not affect the after encoding. 
            audio_quantized_feats_len = audio_quantized_feats_len - 1
        # print(f"Audio Quantized feats shape: {audio_quantized_feats.shape}")
        # print(f"Audio Quantized feats len: {audio_quantized_feats_len}")
        audio_token_encoded, audio_token_len = self.encode_audio(audio_quantized_feats, audio_quantized_feats_len)
        
        if self.is_word_level:
            # duplicate back to attach on each text_token (but maybe we can directly concat them)
            pass
        # fuse audio and text token
        audio_text_token_encoded, audio_text_token_len = self.fuse_encoded_audio_text_module(
            audio_token_encoded,
            audio_token_len,
            text_token_encoded,
            text_token_len,
        )
        # print(audio_text_token_len, audio_text_token_len.shape)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. prepare llm_target
        lm_target = [torch.tensor([IGNORE_ID] * (2 + audio_text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() + [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)
        
        # 5. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 6. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_eos_emb, embedding, audio_text_token_encoded, audio_text_token_len, task_id_emb, speech_token, speech_token_len)
        

        # 7. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        # add quantized_commit loss
        _quantized_loss = self.quantized_loss_weight * quantized_results.get('quantized_loss', 0.0)
        loss += _quantized_loss
        # end of add quantized_commit loss
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 1), lm_target, ignore_label=IGNORE_ID)
        valid_length = torch.sum(lm_target != IGNORE_ID).detach()
        results = {'loss': loss, 'acc': acc, 'len': valid_length}
        if return_logits:
            results['logits'] = logits.detach()
        return results
    
    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            # prompt_audio: torch.Tensor,
            # prompt_audio_len: torch.Tensor, # currently doesn't support
            prompt_speech_token: torch.Tensor, # for evaluate the performance
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            beam_size: int = 1,
            sampling: int = 25,
            max_token_text_ratio: float = 30,
            min_token_text_ratio: float = 3,
            audio_feat: Optional[torch.Tensor] = None,
            audio_feat_len: Optional[torch.Tensor] = None,
            asr_alignment: Optional = None,
            words_index: Optional[List[Tuple[int, int, int]]] = None,
            whisper_text_token: Optional[torch.Tensor] = None,
            whisper_text_token_len: Optional[torch.Tensor] = None,
            taste_token: Optional[torch.Tensor] = None,
            taste_token_len: Optional[torch.Tensor] = None,
            adopt_teacher_forcing_for_test: bool = False,
            drop_eos_before_llm: bool = False,
            **kwargs,
    ):
        print(f"HELLO -> {adopt_teacher_forcing_for_test}", flush=True)

        device = text.device
        assert prompt_text.shape[1] == 0, f"Doesn't support prompting mode when using audio llm (weired behavior)!"
        text_token = torch.concat([prompt_text, text], dim=1)
        text_token_len = text_len
        text_token_len += prompt_text_len
        if self.drop_eos_before_llm or drop_eos_before_llm:
            # assert eos is added. so should not produce error
            text_token = text_token[:, :-1]
            text_token_len = text_token_len - 1
        text_token_embed = self.text_embedding(text_token)

        # 1. encode text
        text_token_encoded, text_token_len = self.encode(text_token_embed, text_token_len)

        if taste_token == None:
            assert audio_feat != None, f"Please set audio feat during audio llm inferencing!"
            # 2. encode audio
            text_token_for_audio, text_token_embed_for_audio = text_token, text_token_embed
            audio_tokenized_results = self.audio_tokenizer(
                audio_feat, # required
                audio_feat_len, # required 
                text_token_for_audio, #required
                text_token_embed_for_audio, # required
                text_token_len, # required
                asr_alignment,
                # kwargs
                words_index = None if not self.is_word_level else words_index,
                whisper_text_token = whisper_text_token,
                whisper_text_token_len = whisper_text_token_len,
            )
            quantized_results = audio_tokenized_results['quantized_results']
        else:
            # directly use taste token
            quantized_results = self.audio_tokenizer.audio_quantizer.encode(
                taste_token,
                taste_token_len,
            )
        audio_quantized_feats, audio_quantized_feats_len = quantized_results['quantized_feats'], quantized_results['quantized_feats_lengths']
        if self.drop_eos_before_llm or drop_eos_before_llm:
            # drop eos which is the last token before llm
            audio_quantized_feats = audio_quantized_feats[:, :-1, :]
            audio_quantized_feats_len = audio_quantized_feats_len - 1
            
        # print(f"taste_token_embeds: {audio_quantized_feats[0][:,:5]}, taste_token_lengths: {audio_quantized_feats_len}")
        audio_token_encoded, audio_token_len = self.encode_audio(audio_quantized_feats, audio_quantized_feats_len)
        
        if self.is_word_level:
            # duplicate back to attach on each text_token (but maybe we can directly concat them)
            pass
        # fuse audio and text token
        audio_text_token_encoded, audio_text_token_len = self.fuse_encoded_audio_text_module(
            audio_token_encoded,
            audio_token_len,
            text_token_encoded,
            text_token_len,
        )

        # 2. encode embedding
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size).to(device)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            print("Does not allow prompting! This is a task more like \'RECONSTRUCTION\'. We will treat the given speech token as target for evaluation.")
            speech_token, speech_token_len = prompt_speech_token, prompt_speech_token_len
            speech_token_emb = self.speech_embedding(speech_token)
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size).to(device)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, audio_text_token_encoded, task_id_emb, prompt_speech_token_emb], dim=1)

        # lm_input, lm_input_len = self.pad_unpad_sequence(sos_eos_emb, embedding, audio_text_token_encoded, audio_text_token_len, task_id_emb, speech_token, speech_token_len)

        # 4. cal min/max_length
        min_len = int((text_token_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_token_len - prompt_text_len) * max_token_text_ratio)


        # 4.5 teacher forcing decode for testing
        if adopt_teacher_forcing_for_test:
            print(f"teachering", flush=True)


            lm_input_with_teacher = torch.concat([sos_eos_emb, embedding, audio_text_token_encoded, task_id_emb, speech_token_emb], dim=1)
            prepend_length = sos_eos_emb.size(1) + embedding.size(1) + audio_text_token_encoded.size(1) 
            lm_input_len = torch.tensor([prepend_length + 1 + speech_token_emb.size(1)], dtype=torch.int32).to(device)
            lm_output, lm_output_mask = self.llm(lm_input_with_teacher, lm_input_len.to(device))
            logits = self.llm_decoder(lm_output)

            print("Length of logits: ", len(logits))

            surprisal = F.cross_entropy(  
                logits.view(-1, logits.size(-1)),   
                speech_token.view(-1),   
                reduction='none'  
            ).view(speech_token.shape)

            print(f"Surprisal length: {len(surprisal)}")

            output_token = logits.argmax(-1)[:, prepend_length:(prepend_length+speech_token.size(-1))].to(torch.int32)
            eos_mask = (output_token == self.speech_token_size)
            print(f"There are {eos_mask.sum()} invalid pred tokens, set it to {IGNORE_ID}")
            output_token[eos_mask] = IGNORE_ID
            # calculate accuracy since it is possible
            mask = speech_token != IGNORE_ID
            numerator = torch.sum(
                output_token[:, :speech_token.size(-1)].masked_select(mask) == speech_token.masked_select(mask)
            )
            print(output_token[0][:10], speech_token[0][:10])
            print(output_token[0][-10:], speech_token[0][-10:])
            denominator = torch.sum(mask)
            print(f"Accuracy: {(numerator / denominator).detach().item():.4f}")

            print("HELLO")
            return (output_token, surprisal)
        
        # 5. step by step decode
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros((0, 0, 0, 0), device=lm_input.device)
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=0, required_cache_size=-1, att_cache=att_cache, cnn_cache=cnn_cache,
                                                                  att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool))
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), sampling, beam_size, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            out_tokens.append(top_ids)
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
        # assert False, f"{prompt_speech_token.shape} {prompt_speech_token_len}"
        output_token = torch.tensor([out_tokens], dtype=torch.int32, device=device)

        return output_token
        # return speech_token # for test only



def test(config_fpath = "config_for_test.yaml"):
    import os
    WORK_DIR = os.getenv('WORK_DIR')
    from hyperpyyaml import load_hyperpyyaml
    from cosyvoice.dataset.dataset import Dataset
    # test TransformerJointLM
    with open(config_fpath, 'r') as f:
        print('Loading configs')
        model_config = load_hyperpyyaml(f)
    print(model_config)

    dataset_config_fpath = "/proj/mtklmadm/dev/mtk53678/rtslm/CosyVoice/cosyvoice/dataset/config_for_test.yaml"
    with open(dataset_config_fpath, 'r') as fr:
        dataset_config = load_hyperpyyaml(fr)
    
    # audio_extractor = config['audio_extractor']
    # print(audio_extractor)
    # print(audio_extractor.output_size())
    
    dataset_for_test = Dataset(dataset_config["test_dataset_parquet_data_list_fpath"], data_pipeline=dataset_config['data_pipeline'], mode='train', shuffle=True, partition=True)
    test_batch = None
    for d in dataset_for_test:
        test_batch = d
        break
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    audio_llm = model_config['llm']
    audio_llm.eval()
    audio_llm.to(device)

    result = audio_llm(
        test_batch,
        device=device,
    )
    

if __name__ == "__main__":
    test()