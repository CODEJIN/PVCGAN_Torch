import torch
import numpy as np
import yaml, librosa, pickle, os
from random import choice

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)


def Stack(audios, mels, pitches, audio_length):
    upsample_Pad = hp_Dict['WaveNet']['Upsample']['Pad']
    audio_Length = audio_length
    mel_Length = audio_Length // hp_Dict['Sound']['Frame_Shift']
    pitch_Length = mel_Length

    audio_List, mel_List, pitch_List = [], [], []
    for audio, mel, pitch in zip(audios, mels, pitches):
        mel_Pad = max(0, mel_Length + 2 * upsample_Pad - mel.shape[0])
        audio_Pad = max(0, audio_Length + 2 * upsample_Pad * hp_Dict['Sound']['Frame_Shift'] - audio.shape[0])            
        pitch_Pad = max(0, pitch_Length + 2 * upsample_Pad - pitch.shape[0])
        
        mel = np.pad(
            mel,
            [[int(np.floor(mel_Pad / 2)), int(np.ceil(mel_Pad / 2))], [0, 0]],
            mode= 'reflect'
            )
        audio = np.pad(
            audio,
            [int(np.floor(audio_Pad / 2)), int(np.ceil(audio_Pad / 2))],
            mode= 'reflect'
            )
        pitch = np.pad(
            pitch,
            [int(np.floor(pitch_Pad / 2)), int(np.ceil(pitch_Pad / 2))],
            mode= 'reflect'
            )

        mel_Offset = np.random.randint(upsample_Pad, max(mel.shape[0] - (mel_Length + upsample_Pad), upsample_Pad + 1))
        audio_Offset = mel_Offset * hp_Dict['Sound']['Frame_Shift']
        pitch_Offset = mel_Offset
        mel = mel[mel_Offset - upsample_Pad:mel_Offset + mel_Length + upsample_Pad]
        audio = audio[audio_Offset:audio_Offset + audio_Length]
        pitch = pitch[pitch_Offset:pitch_Offset + pitch_Length]

        audio_List.append(audio)
        mel_List.append(mel)
        pitch_List.append(pitch)

    max_Audio_Length = max([audio.shape[0] for audio in audio_List])
    max_Mel_Length = max([mel.shape[0] for mel in mel_List])
    max_Pitch_Length = max([pitch.shape[0] for pitch in pitch_List])
    
    audios = np.stack([
        np.pad(audio, [0, max_Audio_Length - audio.shape[0]], constant_values= 0)
        for audio in audio_List
        ], axis= 0)
    mels =  np.stack([
        np.pad(mel, [0, max_Mel_Length - mel.shape[0]], constant_values= -hp_Dict['Sound']['Max_Abs_Mel'])
        for mel in mel_List
        ], axis= 0)
    pitches =  np.stack([
        np.pad(pitch, [0, max_Pitch_Length - pitch.shape[0]], constant_values= 0)
        for pitch in pitch_List
        ], axis= 0)

    return np.stack(audio_List, axis= 0), np.stack(mel_List, axis= 0), np.stack(pitch_List, axis= 0)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super(Dataset, self).__init__()
        
        metadata_Dict = pickle.load(open(path, 'rb'))

        self.pattern_List = []
        for file in metadata_Dict['File_List']:
            with open(os.path.join(hp_Dict['Train']['Train_Pattern']['Path'], file).replace('\\', '/'), 'rb') as f:
                pattern_Dict = pickle.load(f)
                self.pattern_List.append((
                    pattern_Dict['Signal'],
                    pattern_Dict['Mel'],
                    pattern_Dict['Pitch'],
                    pattern_Dict['Singer_ID'],
                    pattern_Dict['Singer_ID'],
                    ))
        
    def __getitem__(self, idx):
        return self.pattern_List[idx]

    def __len__(self):
        return len(self.pattern_List)



class Train_Dataset(Dataset):
    def __init__(self):
        super(Train_Dataset, self).__init__(
            path= os.path.join(
                hp_Dict['Train']['Train_Pattern']['Path'],
                hp_Dict['Train']['Train_Pattern']['Metadata_File']
                ).replace('\\', '/')
            )

        if hp_Dict['Train']['Shared_Train_and_Eval']:
            self.pattern_List = [
                (
                    audio[hp_Dict['Train']['Wav_Length'] * 3:],
                    mel[hp_Dict['Train']['Wav_Length'] // hp_Dict['Sound']['Frame_Shift'] * 3:],
                    pitch[hp_Dict['Train']['Wav_Length'] // hp_Dict['Sound']['Frame_Shift'] * 3:],
                    audio_Singer,
                    mel_Singer
                    )
                for audio, mel, pitch, audio_Singer, mel_Singer in self.pattern_List
                ]

        self.original_Pattern_List = self.pattern_List * hp_Dict['Train']['Train_Pattern']['Accumulated_Dataset_Epoch']
        self.mixup_Pattern_List = []
        self.back_Translate_Pattern_List = []
        self.pattern_List = self.original_Pattern_List
        
    def Accumulation_Renew(
        self,
        mixup_Pattern_List,
        back_Translate_Pattern_List
        ):
        self.mixup_Pattern_List = mixup_Pattern_List * hp_Dict['Train']['Train_Pattern']['Accumulated_Dataset_Epoch']
        self.back_Translate_Pattern_List = back_Translate_Pattern_List * hp_Dict['Train']['Train_Pattern']['Accumulated_Dataset_Epoch']
        self.pattern_List = self.original_Pattern_List + self.mixup_Pattern_List + self.back_Translate_Pattern_List

class Dev_Dataset(Dataset):
    def __init__(self):
        super(Dev_Dataset, self).__init__(
            path= os.path.join(
                hp_Dict['Train']['Eval_Pattern']['Path'],
                hp_Dict['Train']['Eval_Pattern']['Metadata_File']
                ).replace('\\', '/')
            )

        if hp_Dict['Train']['Shared_Train_and_Eval']:
            hp_Dict['Train']['Wav_Length']
            hp_Dict['Train']['Wav_Length'] // hp_Dict['Sound']['Frame_Shift']
            self.pattern_List = [
                (
                    audio[:hp_Dict['Train']['Wav_Length'] * 3],
                    mel[:hp_Dict['Train']['Wav_Length'] // hp_Dict['Sound']['Frame_Shift'] * 3],
                    pitch[:hp_Dict['Train']['Wav_Length'] // hp_Dict['Sound']['Frame_Shift'] * 3],
                    audio_Singer,
                    mel_Singer
                    )
                for audio, mel, pitch, audio_Singer, mel_Singer in self.pattern_List
                ]

class Accumulation_Dataset(Dataset):
    def __init__(self):
        super(Accumulation_Dataset, self).__init__(
            path= os.path.join(
                hp_Dict['Train']['Train_Pattern']['Path'],
                hp_Dict['Train']['Train_Pattern']['Metadata_File']
                ).replace('\\', '/')
            )

        if hp_Dict['Train']['Shared_Train_and_Eval']:
            self.pattern_List = [
                (
                    audio[hp_Dict['Train']['Wav_Length'] * 3:],
                    mel[hp_Dict['Train']['Wav_Length'] // hp_Dict['Sound']['Frame_Shift'] * 3:],
                    pitch[hp_Dict['Train']['Wav_Length'] // hp_Dict['Sound']['Frame_Shift'] * 3:],
                    singer
                    )
                for audio, mel, pitch, singer, _ in self.pattern_List
                ]
        else:
            self.pattern_List = [x[:-1] for x in self.pattern_List]

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pattern_path= 'Inference_for_Training.txt'
        ):
        self.pattern_List = []
        with open(pattern_path, 'r') as f:
            for line in f.readlines()[1:]:                
                source_Label, path, singer_Label, singer, start_Index, end_Index = line.strip().split('\t')
                self.pattern_List.append((
                    source_Label,
                    path,
                    singer_Label,
                    int(singer),
                    int(start_Index),
                    int(end_Index)
                    ))

        self.pattern_Cache_Dict = {}

    def __getitem__(self, idx):
        if idx in self.pattern_Cache_Dict.keys():
            return self.pattern_Cache_Dict[idx]

        source_Label, path, singer_Label, singer, start_Index, end_Index = self.pattern_List[idx]
        with open(path, 'rb') as f:
            pattern_Dict = pickle.load(f)
            pattern = (
                pattern_Dict['Signal'][start_Index * hp_Dict['Sound']['Frame_Shift']:end_Index * hp_Dict['Sound']['Frame_Shift']],
                pattern_Dict['Mel'][start_Index:end_Index],
                pattern_Dict['Pitch'][start_Index:end_Index],
                singer,
                source_Label,
                singer_Label
                )

        self.pattern_Cache_Dict[idx] = pattern

        return pattern

    def __len__(self):
        return len(self.pattern_List)



class Collater:
    def __call__(self, batch):
        audios, mels, pitches, audio_Singers, mel_Singers = zip(*batch)
        audios, mels, pitches = Stack(audios, mels, pitches, audio_length= hp_Dict['Train']['Wav_Length'])
        audio_Singers = np.stack(audio_Singers, axis= 0)
        mel_Singers = np.stack(mel_Singers, axis= 0)

        audios = torch.FloatTensor(audios)   # [Batch, Time]
        mels = torch.FloatTensor(mels).transpose(2, 1)   # [Batch, Time, Mel_dim] -> [Batch, Mel_dim, Time]
        pitches = torch.FloatTensor(pitches)   # [Batch, Time]
        audio_Singers = torch.LongTensor(audio_Singers)   # [Batch]
        mel_Singers = torch.LongTensor(mel_Singers)   # [Batch]
        noises = torch.randn(size= audios.size()) # [Batch, Time]

        return audios, mels, pitches, audio_Singers, mel_Singers, noises

class Accumulation_Collater:
    def __call__(self, batch):
        audios, mels, pitches, singers = zip(*batch)
        audios, mels, pitches = Stack(audios, mels, pitches, audio_length= hp_Dict['Train']['Wav_Length'] * 2)
        singers = np.stack(singers, axis= 0)
        
        total_Audios = [audio for audio, _, _, _ in batch]
        total_Pitches = [pitch for _, _, pitch, _ in batch]
        audios = torch.FloatTensor(audios)   # [Batch, Time]
        mels = torch.FloatTensor(mels).transpose(2, 1)   # [Batch, Time, Mel_dim] -> [Batch, Mel_dim, Time]
        pitches = torch.FloatTensor(pitches)   # [Batch, Time]
        singers = torch.LongTensor(singers)   # [Batch]
        noises = torch.randn(size= audios.size()) # [Batch, Time]

        return total_Audios, total_Pitches, audios, mels, pitches, singers, noises

class Inference_Collater:
    def __call__(self, batch):
        audios, mels, pitches, singers, source_Labels, singer_Labels = zip(*batch)
        audios, mels, pitches = Stack(audios, mels, pitches, audio_length= max([audio.shape[0] for audio in audios]))
        singers = np.stack(singers, axis= 0)

        audios = torch.FloatTensor(audios)   # [Batch, Time]
        mels = torch.FloatTensor(mels).transpose(2, 1)   # [Batch, Time, Mel_dim] -> [Batch, Mel_dim, Time]
        pitches = torch.FloatTensor(pitches)   # [Batch, Time]
        singers = torch.LongTensor(singers)   # [Batch]
        noises = torch.randn(size= audios.size()) # [Batch, Time]

        return audios, mels, pitches, singers, noises, source_Labels, singer_Labels