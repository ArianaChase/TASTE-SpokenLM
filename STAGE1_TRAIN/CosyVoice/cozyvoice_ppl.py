import argparse
from cosyvoice.cli.cosyvoice import CosyVoice  
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
import torchaudio
from modelscope import snapshot_download
import torch
import torch.nn.functional as F
from cosyvoice.utils.common import IGNORE_ID

TEST_PATH = "/home/u5504709/new_work/speech_ppl/speechocean762/WAVE/SPEAKER0001/000010011.WAV"
REF_PATH = "/home/u5504709/new_work/speech_ppl/work/data/test_speech.mp3"

OUTPUT_PATH = "/home/u5504709/new_work/speech_ppl/work/outputs"

#snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')

cosyvoice = CosyVoice('/home/u5504709/new_work/speech_ppl/pretrained_models/CosyVoice-300M-SFT')

frontend = cosyvoice.frontend
canonical_text = "WE CALL IT BEAR"
wav = load_wav(TEST_PATH, 16000)

text_token, text_token_len = frontend._extract_text_token(canonical_text)
speech_token, speech_token_len = frontend._extract_speech_token(wav)
embedding = frontend._extract_spk_embedding(wav)   # name may vary

batch = {
    'text_token': text_token,
    'text_token_len': text_token_len,
    'speech_token': speech_token,
    'speech_token_len': speech_token_len,
    'embedding': embedding,
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get per token loss

logits, lm_target = cosyvoice.model.llm.forward(batch, device)

# cross_entropy wants class dim in position 1, not last
logits_reshaped = logits.transpose(1, 2)          # (B, V, T)

print(logits.shape)      # should be (B, T, vocab_size) e.g. (1, 138, ~4097)
print(lm_target.shape)   # should be (B, T)             e.g. (1, 138)

surprisal_per_position = F.cross_entropy(
    logits_reshaped,
    lm_target.long(),                              # still needs int64
    ignore_index=IGNORE_ID,
    reduction='none'
)   # (B, T) — value is 0 at ignored (text/padding) positions

mask = (lm_target != IGNORE_ID)
surprisal = surprisal_per_position[mask]           # only real speech-token positions, nats

print(len(surprisal))
print(surprisal)

