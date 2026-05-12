import torch
from typing import Dict, Optional, Union, List, Tuple
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from funasr import AutoModel
from funasr.utils.misc import deep_update
import librosa
import logging
import os
from einops import rearrange
from transformers import WhisperModel, WhisperProcessor
from cosyvoice.utils.model_utils import load_whisper_whole_model, get_s3_encoder_dict
RTSLM_WORK_DIR = os.getenv(
    'RTSLM_WORK_DIR'
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAudioEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    # process that should be done during data collation
    # TODO: should be moved to other place

    def to(self, device):
        self.device = device
        return super().to(device)
    
    def get_device(self):
        if next(self.parameters(), None) is not None:
            return next(self.parameters()).device
        elif next(self.buffers(), None) is not None:
            return next(self.buffers()).device
        else:
            return 'cpu'
        
    def extract_feature(
        self,
        audio_fpaths: List[str],
    ):
        raise NotImplementedError

    # Parameter-related functions
    def forward(
        self,
        audio_features: torch.Tensor,
        audio_features_lengths: torch.Tensor,
        **kwargs,
    ):
        raise NotImplementedError

class SenseVoiceAudioEncoder(BaseAudioEncoder):
    def __init__(
        self,
        model_card: str = "iic/SenseVoiceSmall",
        model_code_dir: str = "customized_sensevoice/model.py",
        dither: float = 1.0, # If you don't want to use dither in kaldi.fbank, set it to 0.0 for reproducability
        hub: str = "ms",
        prepend_inputs_before_encoding: bool = False, 
        extract_hidden: bool = True,
    ):
        super().__init__()
        self.funasr_model = AutoModel(
            model=model_card,
            trust_remote_code=True,
            remote_code=model_code_dir,
            hub=hub,
            device="cpu",
            disable_update=True
        )

        self.kwargs = self.funasr_model.kwargs
        logger.info(self.kwargs)
        # separate components for flexible usage. 
        self.frontend = self.kwargs["frontend"] # fbank
        self.frontend.dither = dither
        # I examined and it is a SentencepiecesTokenizer (built by themselves)
        self.text_tokenizer = self.kwargs["tokenizer"]
        self.model = self.funasr_model.model
        self.feature_dim = self.kwargs['encoder_conf']['output_size']
        self.prepend_inputs_before_encoding  = prepend_inputs_before_encoding
        if self.prepend_inputs_before_encoding:
            logger.info("Will automatically prepend SenseVoice input tokens before encoding!")
        self.extract_hidden = extract_hidden
        if self.extract_hidden:
            logger.info("Will extract hidden repr from SenseVoice-Small (before tp_encoders, after encoders)")
    

    def extract_feature(
        self,
        audio_fpaths: List[str],
        **cfg,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg['device'] = cfg.get("device", self.get_device())
        meta_data, audio_features, audio_features_lengths = self.model.prepare_inputs(
            audio_fpaths,
            data_lengths=None,
            tokenizer=self.text_tokenizer,
            frontend=self.frontend,
            **cfg,
        )
        # print(meta_data)
        return audio_features, audio_features_lengths

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_features_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        '''
        Args:
            audio_features: torch.Tensor, shape (batch_size, max_audio_len, audio_feat_dim)
            audio_features_lengths: torch.Tensor, shape (batch_size,)
        '''
        if self.prepend_inputs_before_encoding:
            audio_features, audio_features_lengths = self.model.prepend_inputs(audio_features, audio_features_lengths, **kwargs)
        encoder_out, encoder_out_lengths, hidden_out = self.model.encoder(audio_features, audio_features_lengths, extract_hidden=self.extract_hidden)

        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        
        if isinstance(hidden_out, torch.Tensor):
            # print(f"Use hidden_out!, enc_out_shape={encoder_out.shape}, hidden_out_shape={hidden_out.shape}")
            assert encoder_out.shape[1] == hidden_out.shape[1], f"length between endoer out and hidden out does not match ({encoder_out.shape}, {hidden_out.shape})."
        # generate ctc output

        ctc_logits = self.model.ctc.log_softmax(encoder_out)
        if self.kwargs.get("ban_emo_unk", False):
            ctc_logits[:, :, self.emo_dict["unk"]] = -float("inf")

        # prepare output
        # Currently we extract only the last hidden layer.
        results = {
            "encoded_feats": hidden_out if self.extract_hidden else encoder_out,
            "encoded_feats_lengths": encoder_out_lengths,
            "ctc_logits": ctc_logits,
        }
        # print(meta_data)
        return results

class WhisperAudioEncoder(BaseAudioEncoder):
    def __init__(
        self, 
        model_name_or_path: str,
        s3_encoder_ckpt: str = None, # checkpoint path of s3_encoder model 
        target_hidden_layer: int = 6, # specify which layer to extract. NOTE: zero means to extract the embed feature. Set to -1 to extract all hidden
        encoder_model: nn.Module = None, # allow passing a WhisperEncoder down for usage.
        attn_implementation: str = "eager", # possible choices: [eager, sdpa, flash_attention_2]
        dtype: str = "bfloat16",
    ):
        print("dtype is ", dtype)
        super().__init__()
        if encoder_model == None:
            whole_model, _torch_dtype = load_whisper_whole_model(
                model_name_or_path,
                attn_implementation = attn_implementation,
                dtype = dtype,
            )
            self.encoder = whole_model.get_encoder()
        else:
            self.encoder = encoder_model

        self.processor = WhisperProcessor.from_pretrained(model_name_or_path)
        # print(self.encoder)
        if target_hidden_layer != -1 and not return_last_hidden: # -1 means extract all hidden layers. 
            for i, layer in enumerate(self.encoder.layers):
                if i > target_hidden_layer:
                    print(f"Delete layer {i}")
                    self.encoder.layers[i] = None
                
        # check load s3_encoder_ckpt or no
        if s3_encoder_ckpt != None:
            s3_encoder_dict = get_s3_encoder_dict(self.state_dict(), s3_encoder_ckpt)
            # print(s3_encoder_dict)
            self.load_state_dict(s3_encoder_dict)
        
        self.extractor_hop_length = self.processor.feature_extractor.hop_length  # This is basically 160 for whisper extractor
        self.extractor_max_frames = self.processor.feature_extractor.nb_max_frames  # This is basically 30 * 16000 // 160  
        self.expected_seq_length = self.encoder.max_source_positions * self.encoder.conv1.stride[0] * self.encoder.conv2.stride[0]
        print(f"WhisperAudioEncoder | expected sequence lengths: {self.expected_seq_length}")
        self.target_hidden_layer = target_hidden_layer
        print(f"WhisperAudioEncoder | target layer: {self.target_hidden_layer}")
    
    def extract_feature(
        self,
        audio_fpaths: List[str],
        permute: bool = True,
        # pad_to_whisper_input_size: Optional[bool] = None, NOTE: deprecated. This behavior does not align with the original whisper designation
        **cfg,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_feat_list, audio_feat_len_list = [], []
        waveforms = []
        for audio_fpath in audio_fpaths:
            waveform, sr = librosa.load(audio_fpath, sr=16_000, mono=True) # target sr is 16_000 for whisper
            assert sr==16_000, "Something went wrong"
            waveforms.append(waveform)
            audio_feat_len_list.append(len(waveform) // self.extractor_hop_length)

        inputs = self.processor(waveforms, sampling_rate=16000, return_tensors='pt', max_length=None) # will automatically pad to 
        audio_feat = inputs["input_features"]
        if permute:
            audio_feat = audio_feat.transpose(-1, -2) # (B, C, T) -> (B, T, C)
        audio_feat_len = torch.tensor(audio_feat_len_list, dtype=torch.int32) # This is for reference only. Whisper encoder accepts same input sizes (=3000) only. 
        return audio_feat, audio_feat_len

    def pad_to_whisper_input_size(self, audio_feat: List[torch.Tensor], padding_value=0.0):
        b, t = len(audio_feat), self.expected_seq_length
        c = audio_feat[0].shape[-1] # each tensor is with shape (T, C)
        padded_tensors = torch.full(
            (b, t, c),
            padding_value,
        )
        for i, tensor in enumerate(audio_feat):
            length = tensor.size(0)
            padded_tensors[i, :length] = tensor
        
        return padded_tensors

    #@torch.cuda.amp.autocast()
    #@torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    #@torch.amp.autocast('cuda')
    @torch.amp.autocast('cuda', dtype=torch.bfloat16)  
    def forward(
        self,
        audio_features: torch.Tensor,
        audio_features_lengths: torch.Tensor,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        # print(audio_features.shape)

        input_features = audio_features.transpose(1, 2).to(torch.bfloat16)   # (B, T, C) -> (B, C, T)
        print("dtype at audio_encoder 260 is ", input_features.dtype)

        if input_features.shape[-1] != self.expected_seq_length:
            if input_features.shape[-1] < self.expected_seq_length:
                # already padded but should be extended to fit whisper's input seq length
                p1d = (0, self.expected_seq_length - input_features.shape[-1])
                input_features = F.pad(input_features, p1d, 'constant', 0.0)
            else:
                raise ValueError(
                    f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
                )
            
        inputs_embeds = nn.functional.gelu(self.encoder.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.encoder.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1) # (B, T, C)
        embed_pos = self.encoder.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.encoder.dropout, training=self.encoder.training)

        encoder_states = {} if output_hidden_states else None

        for idx, encoder_layer in enumerate(self.encoder.layers):
            if idx == self.target_hidden_layer:
                results = {
                    'encoded_feats': hidden_states,
                    'encoded_feats_lengths': audio_features_lengths // 2, # whisper encoder will down-sample by 2 
                }
                return results
            elif idx == output_hidden_states:
                encoder_states[f'{idx}'] = hidden_states
            # if self.target_hidden_layer < 0:
            #     encoder_states = encoder_states + (hidden_states,) # forbidden returning all hidden states
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            # else:
            layer_outputs = encoder_layer(
                hidden_states,
                None,
                layer_head_mask=None,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                raise NotImplementedError
                # attentions = layer_outputs[1]

        if self.target_hidden_layer < 0:
            hidden_states = self.encoder.layer_norm(hidden_states)
            encoder_states['last_hidden'] = hidden_states
            results = {
                'encoded_feats': encoder_states,
                'encoded_feats_lengths': audio_features_lengths // 2, 
            } # return all encoder states
        else: 
            # return last hidden
            results = {
                'encoded_feats': hidden_states,
                'encoded_feats_lengths': audio_features_lengths // 2, 
            }
        return results


# TODO: Implement EncodecAudioEncoder
class EncodecAudioEncoder(BaseAudioEncoder):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                dropout=dropout,
            ),
            num_layers=num_layers,
        )
        self.linear = nn.Linear(input_size, output_size)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
    ):
        audio = self.encoder(audio)
        audio = self.linear(audio)
        return audio


def test_sensevoice_enc(): # testing audio extraction
    # import os
    # RTSLM_WORK_DIR = os.getenv('RTSLM_WORK_DIR')
    print(RTSLM_WORK_DIR)
    sensevoice_encoder = SenseVoiceAudioEncoder(
        model_card="/proj/mtklmadm/dev/mtk53678/rtslm_storage/pretrained_models/SenseVoiceSmall",
        model_code_dir="customized_sensevoice/model.py",
        extract_hidden=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sensevoice_encoder.to(device)

    audio_fpaths = [
        # "/root/rtslm/CosyVoice/cross_lingual_prompt.wav",
        # "/root/rtslm/CosyVoice/cross_lingual.wav",
        # f"{RTSLM_WORK_DIR}/CosyVoice/instruct4.wav",
        f"{RTSLM_WORK_DIR}/CosyVoice/en.mp3",
    ]
    print(sensevoice_encoder)

    audio_features, audio_features_lengths = sensevoice_encoder.extract_feature(
        audio_fpaths,
        cache={},
        language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size_s=60,
    )
    # extract encoded audio features
    results = sensevoice_encoder.forward(audio_features, audio_features_lengths)
    for feat in results['encoded_feats']:
        print(feat.shape)
    for feat_length in results['encoded_feats_lengths']:
        print(feat_length)
    for ctc in results['ctc_logits']:
        print(ctc.shape)

def test_whisper_enc():
    model_fpath = "/proj/mtklmadm/dev/mtk53678/rtslm_storage/pretrained_models/whisper-large-v3"
    s3_encoder_ckpt = '/proj/mtklmadm/dev/mtk53684/new_model.pth'
    whisper_encoder = WhisperAudioEncoder(model_fpath, s3_encoder_ckpt = None)
    
    audio_fpaths = [
        # "/root/rtslm/CosyVoice/cross_lingual_prompt.wav",
        # "/root/rtslm/CosyVoice/cross_lingual.wav",
        # "/root/rtslm/CosyVoice/instruct.wav",
        f"{RTSLM_WORK_DIR}/CosyVoice/en.mp3",
        # f"{RTSLM_WORK_DIR}/CosyVoice/instruct4.wav"
    ]
    audio_feat, audio_feat_len = whisper_encoder.extract_feature(
        audio_fpaths,
        pad_to_whisper_input_size=True
    )
    # print(rearrange(audio_feat, 'b t c -> b c t').is_contiguous())
    print(audio_feat_len)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    whisper_encoder.to(device)
    whisper_encoder.eval()
    with torch.no_grad():
        hidden_state = whisper_encoder(
            audio_feat.to(device),
            audio_feat_len.to(device)
        )
        print(hidden_state)


if __name__ == "__main__":
    test_whisper_enc()