from typing import Dict, Optional, Union, List, Tuple
from torch import nn
import torch
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


class WhisperAudioEncoder(BaseAudioEncoder):
    def __init__(
        self, 
        model_name_or_path: str,
        target_hidden_layer: int = 6, # specify which layer to extract. NOTE: zero means to extract the embed feature. Set to -1 to extract all hidden
        encoder_model: nn.Module = None, # allow passing a WhisperEncoder down for usage.
        attn_implementation: str = "eager", # possible choices: [eager, sdpa, flash_attention_2]
        dtype: str = "float16",
    ):
        super().__init__()
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        else: 
            torch_dtype = torch.float32
        self.torch_dtype = torch_dtype
        if encoder_model == None:
            whole_model = WhisperModel.from_pretrained(
                model_name_or_path,
                torch_dtype = torch_dtype,
                attn_implementation = attn_implementation,
            )
            self.encoder = whole_model.get_encoder()
        else:
            self.encoder = encoder_model
        self.processor = WhisperProcessor.from_pretrained(model_name_or_path)
        print(self.encoder)
        for i, layer in enumerate(self.encoder.layers):
            if target_hidden_layer == -1: break # -1 means extract all hidden layers. 
            if i > target_hidden_layer:
                print(f"Delete layer {i}")
                self.encoder.layers[i] = None
        self.expected_seq_length = self.encoder.max_source_positions * self.encoder.conv1.stride[0] * self.encoder.conv2.stride[0]
        print(f"WhisperAudioEncoder | expected sequence lengths: {self.expected_seq_length}")
        self.target_hidden_layer = target_hidden_layer
        print(f"WhisperAudioEncoder | target layer: {self.target_hidden_layer}")
    
    def extract_feature(
        self,
        audio_fpaths: List[str],
        pad_to_whisper_input_size: Optional[bool] = None,
        **cfg,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_feat_list, audio_feat_len_list = [], []
        for audio_fpath in audio_fpaths:
            waveform, sr = librosa.load(audio_fpath, sr=16_000, mono=True) # target sr is 16_000 for whisper
            inputs = self.processor(waveform, sampling_rate=16000, return_tensors='pt', max_length=waveform.shape[-1])
            audio_feat = inputs["input_features"].transpose(-1, -2).squeeze(0) # (B, C, T) -> (T, C)
            audio_feat_list.append(audio_feat)
            audio_feat_len_list.append(audio_feat.shape[0])
        if pad_to_whisper_input_size:
            audio_feat = self.pad_to_whisper_input_size(audio_feat_list)
        else:
            audio_feat = pad_sequence(audio_feat_list, batch_first=True)
        audio_feat_len = torch.tensor(audio_feat_len_list, dtype=torch.int32)

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
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            input_features = audio_features.transpose(1, 2) # (B, T, C) -> (B, C, T)

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

            encoder_states = () if output_hidden_states else None

            for idx, encoder_layer in enumerate(self.encoder.layers):
                # hidden_states = hidden_states.to(self.torch_dtype)
                if idx == self.target_hidden_layer:
                    results = {
                        'encoded_feats': hidden_states,
                        'encoded_feats_lengths': audio_features_lengths // 2, # whisper encoder will down-sample by 2 
                    }
                    return results
                if self.target_hidden_layer < 0:
                    encoder_states = encoder_states + (hidden_states,)
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                else:
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

            hidden_states = self.layer_norm(hidden_states)
            if self.target_hidden_layer < 0:
                encoder_states = encoder_states + (hidden_states,)
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


def test_whisper_enc():
    model_fpath = "/proj/mtklmadm/dev/mtk53684/rtslm_storage/pretrained_models/whisper-large-v3"
    whisper_encoder = WhisperAudioEncoder(model_fpath)

    # Load onnx2torch model
    model_pth = '/proj/mtklmadm/dev/mtk53684/new_model.pth'
    pretrained_weights = torch.load(model_pth)
    model_dict = whisper_encoder.state_dict()
    for key in model_dict:
        print(key)

    for name, param in pretrained_weights.items():
        if name in model_dict:
            model_dict[name].copy_(param)
        else:
            print(f"Skipping {name} as it doesn't exist in the whispher encode's state_dict")
    whisper_encoder.load_state_dict(model_dict)

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
        # print(hidden_state.shape)

if __name__ == "__main__":
    test_whisper_enc()