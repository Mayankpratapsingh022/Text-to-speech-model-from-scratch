import pandas as pd
import torch
import torch.nn.functional as F 
import torchaudio 
from torch.utils.data import Dataset
import librosa 
from tokenizer import Tokenizer 
import numpy as np 
import warnings
import os
import matplotlib.pyplot as  plt
warnings.filterwarnings('ignore')



def load_wav(path_to_audio,sr=22050):
    # Use librosa to load audio (avoids torchcodec dependency issues)
    audio, orig_sr = librosa.load(path_to_audio, sr=None)
    audio = torch.from_numpy(audio).float()

    if sr != orig_sr:
        audio = torchaudio.functional.resample(audio.unsqueeze(0), orig_freq=orig_sr, new_freq=sr)
        audio = audio.squeeze(0)

    return audio


def amp_to_db(x,min_db=-100):
    # 20 * torch.log10(1e-5) = 20 * -5 = -100
    clip_val = 10 ** (min_db / 20) 
    return 20 * torch.log10(torch.clamp(x,min=min_db))


def db_to_amp(x):
    return 10 ** (x/20)


def normalize(x,min_db=-100,max_abs_val=4):
    x = (x - min_db) / - min_db 
    x = 2 * max_abs_val * x - max_abs_val 
    x = torch.clip(x,min=-max_abs_val,max=max_abs_val)
    return x 


    
def denormalize(x,min_db=-100, max_abs_val=4):
    x = torch.clip(x,min=-max_abs_val,max=max_abs_val)
    x = (x + max_abs_val) / (2 * max_abs_val)
    x = x * min_db + min_db 
    return x 



class AudioMelConversions:
    def __init__(self,num_mels=80,sampling_rate=22050,n_fft=1024,window_size=1024,hop_size=256,fmin=0,fmax=8000,center=False,min_db=-100,max_scaled_abs=4):

        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.window_size = window_size
        self.hop_size = hop_size 
        self.fmin = fmin
        self.fmax =  fmax
        self.center = center
        self.min_db = min_db
        self.max_scaled_abs = max_scaled_abs

        self.spec2mel = self._get_spec2mel_proj()
        self.mel2spec = torch.linalg.pinv(self.spec2mel)

    def _get_spec2mel_proj(self):

        mel = librosa.filters.mel(sr=self.sampling_rate,
    n_fft=self.n_fft,
    n_mels=self.num_mels,
    fmin=self.fmin,
    fmax=self.fmax
    )
        return torch.from_numpy(mel)

    def audio2mel(self,audio,do_norm=False):
        if not isinstance(audio,torch.Tensor):
            audio = torch.tensor(audio,dtype=torch.float32)
        spectrogram = torch.stft(
            input=audio,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.window_size,
            window=torch.hann_window(self.window_size).to(audio.device),
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True

        )

        spectrogram = torch.abs(spectrogram)
        print(spectrogram.shape)
        mel = torch.matmul(self.spec2mel.to(spectrogram.device),spectrogram)
        mel = amp_to_db(mel,self.min_db) 
        if do_norm:
            mel = normalize(mel,min_db=self.min_db,max_abs_val=self.max_scaled_abs)
        return mel 

    def mel2audio(self,mel,do_denorm=False,griffin_lim_iter=50):
        if do_denorm:
            mel = denormalize(mel,min_db=self.min_db,max_abs_val=self.max_scaled_abs)

        mel = db_to_amp(mel)
        spectrogram = torch.matmul(self.mel2spec.to(mel.device),mel).cpu().numpy()
        audio = librosa.griffinlim(S=spectrogram,
                                   n_iter=griffin_lim_iter,
                                   hop_length=self.hop_size,
                                   win_length=self.window_size,
                                   n_fft=self.n_fft,
                                   window="hann" 
        )
        audio *= 32767 / max ( 0.01,np.max(np.abs(audio)))
        audio = audio.astype(np.int16)
        return audio



class TTSDataset(Dataset):
    def __init__(
        self,
        path_to_metadata,
        sample_rate=22050,
        n_fft=1024,
        window_size=1024,
        hop_size=256,
        fmin=0,
        fmax=8000,
        num_mels=80,
        center=False,
        normalized=False,
        min_db=-100,
        max_scaled_abs=4

    ):
        self.metadata = pd.read_csv(path_to_metadata)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_size = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.num_mels = num_mels
        self.center = center
        self.normalized = normalized
        self.min_db = min_db
        self.max_scaled_abs = max_scaled_abs

        self.transcript_lengths = [len(Tokenizer().encode(t)) for t in self.metadata["normalized_transcript"]]
        self.audio_proc = AudioMelConversions(
            num_mels=self.num_mels,
            sampling_rate=self.sample_rate,
            n_fft=self.n_fft,
            window_size=self.win_size,
            hop_size=self.hop_size,
            fmin=self.fmin,
            fmax=self.fmax,
            center=self.center,
            min_db=self.min_db,
            max_scaled_abs=self.max_scaled_abs


    )
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self,idx):
        sample = self.metadata.iloc[idx]
        path_to_audio = sample["file_path"]
        transcript = sample["normalized_transcript"]
        audio = load_wav(path_to_audio,sr=self.sample_rate)
        mel= self.audio_proc.audio2mel(audio,do_norm=True)
        return transcript, mel.squeeze(0)

    


      

    
if __name__ == "__main__":
    # Get the project root directory (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_to_audio = os.path.join(project_root, 'LJSpeech-1.1/wavs/LJ002-0169.wav')
    audio = load_wav(path_to_audio)
    
   
    amc = AudioMelConversions()
    amc.audio2mel(audio)


