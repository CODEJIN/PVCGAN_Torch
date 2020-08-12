import torch
import numpy as np
import yaml, librosa, pickle, os
from random import choice

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

class AccumulationDataset(torch.utils.data.Dataset):
    def __init__(self):
        with open(os.path.join(hp_Dict['Train']['Pattern_Path'], hp_Dict['Train']['Metadata_File']).replace('\\', '/'), 'rb') as f:
            metadata_Dict = pickle.load(f)

        self.file_List = [
            file for file, length in metadata_Dict['Sig_Length_Dict'].items()
            if length > hp_Dict['Train']['Wav_Length']
            ]
        
        self.pattern_Cache_Dict = {}

    
    def __getitem__(self, idx):
        if idx in self.pattern_Cache_Dict.keys():
            return self.pattern_Cache_Dict[idx]

        with open(os.path.join(hp_Dict['Train']['Pattern_Path'], self.file_List[idx]).replace('\\', '/'), 'rb') as f:
            pattern_Dict = pickle.load(f)
            pattern = (
                pattern_Dict['Signal'],
                pattern_Dict['Mel'],
                pattern_Dict['Pitch'],
                pattern_Dict['Singer_ID'],
                )

        self.pattern_Cache_Dict[idx] = pattern
                
        return pattern

    def __len__(self):
        return len(self.file_List)

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.original_Pattern_List = []
        self.mixup_Pattern_List = []
        self.back_Translate_Pattern_List = []

        with open(os.path.join(hp_Dict['Train']['Pattern_Path'], hp_Dict['Train']['Metadata_File']).replace('\\', '/'), 'rb') as f:
            metadata_Dict = pickle.load(f)
        
        for file in metadata_Dict['File_List']:
            with open(os.path.join(hp_Dict['Train']['Pattern_Path'], file).replace('\\', '/'), 'rb') as f:
                pattern_Dict = pickle.load(f)
                self.original_Pattern_List.append((
                    pattern_Dict['Signal'],
                    pattern_Dict['Mel'],
                    pattern_Dict['Pitch'],
                    pattern_Dict['Singer_ID'],
                    pattern_Dict['Singer_ID'],
                    ))
        
    def __getitem__(self, idx):
        pattern_List = \
            self.original_Pattern_List + \
            self.mixup_Pattern_List + \
            self.back_Translate_Pattern_List
        pattern_List *= hp_Dict['Train']['Accumulation_Inverval']

        return pattern_List[idx]

    def __len__(self):
        return hp_Dict['Train']['Accumulation_Inverval'] * len(
            self.original_Pattern_List +
            self.mixup_Pattern_List +
            self.back_Translate_Pattern_List
            )

    def Accumulation_Renew(
        self,
        mixup_Pattern_List,
        back_Translate_Pattern_List
        ):
        self.mixup_Pattern_List = mixup_Pattern_List
        self.back_Translate_Pattern_List = back_Translate_Pattern_List

class DevDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pattern_path= 'Inference_for_Training.txt'
        ):
        self.pattern_List = []
        with open(pattern_path, 'r') as f:
            for line in f.readlines()[1:]:                
                source_Label, path, conversion_Label, conversion_Singer, start_Index, end_Index = line.strip().split('\t')
                self.pattern_List.append((
                    source_Label,
                    path,
                    conversion_Label,
                    int(conversion_Singer),
                    int(start_Index),
                    int(end_Index)
                    ))

        self.pattern_Cache_Dict = {}

    def __getitem__(self, idx):
        if idx in self.pattern_Cache_Dict.keys():
            return self.pattern_Cache_Dict[idx]

        source_Label, path, conversion_Label, conversion_Singer, start_Index, end_Index = self.pattern_List[idx]
        with open(path, 'rb') as f:
            pattern_Dict = pickle.load(f)
            pattern = (
                pattern_Dict['Signal'],
                pattern_Dict['Mel'],
                pattern_Dict['Pitch'],
                pattern_Dict['Singer_ID'],
                conversion_Singer,
                start_Index,
                end_Index,
                source_Label,
                conversion_Label
                )

        self.pattern_Cache_Dict[idx] = pattern

        return pattern

    def __len__(self):
        return len(self.pattern_List)

class Accumulation_Collater:
    def __init__(self):
        self.audio_Length = hp_Dict['Train']['Wav_Length'] * 2 
        self.mel_Length = hp_Dict['Train']['Wav_Length'] * 2 // hp_Dict['Sound']['Frame_Shift'] + 2 * hp_Dict['WaveNet']['Upsample']['Pad']
        self.pitch_Length = hp_Dict['Train']['Wav_Length'] * 2 // hp_Dict['Sound']['Frame_Shift']

    def __call__(self, batch):
        audios, mels, pitches, singers = [], [], [], []
        for index, (audio, mel, pitch, singer) in enumerate(batch):
            max_Offset = mel.shape[0] - 2 - (self.mel_Length + 2 * hp_Dict['WaveNet']['Upsample']['Pad'])
            if max_Offset <= 0:
                continue
                
            mel_Offset = np.random.randint(0, max_Offset)
            audio_Offset = (mel_Offset + hp_Dict['WaveNet']['Upsample']['Pad']) * hp_Dict['Sound']['Frame_Shift']
            pitch_Offset = mel_Offset + hp_Dict['WaveNet']['Upsample']['Pad']

            audio = audio[audio_Offset:audio_Offset + self.audio_Length]
            mel = mel[mel_Offset:mel_Offset + self.mel_Length]
            pitch = pitch[pitch_Offset:pitch_Offset + self.pitch_Length]

            audios.append(audio)
            mels.append(mel)
            pitches.append(pitch)
            singers.append(singer)

        total_Audios = [audio for audio, _, _, _ in batch]
        total_Pitches = [pitch for _, _, pitch, _ in batch]
        audios = torch.FloatTensor(np.stack(audios, axis= 0))   # [Batch, Time]
        mels = torch.FloatTensor(np.stack(mels, axis= 0)).transpose(2, 1)   # [Batch, Time, Mel_dim] -> [Batch, Mel_dim, Time]
        pitches = torch.FloatTensor(np.stack(pitches, axis= 0))   # [Batch, Time]
        singers = torch.LongTensor(np.stack(singers, axis= 0))   # [Batch]
        noises = noises = torch.randn(size= audios.size()) # [Batch, Time]

        return total_Audios, total_Pitches, audios, mels, pitches, singers, noises

class Train_Collater:
    def __init__(self):
        self.mel_Length = hp_Dict['Train']['Wav_Length'] // hp_Dict['Sound']['Frame_Shift'] + 2 * hp_Dict['WaveNet']['Upsample']['Pad']
        self.pitch_Length = hp_Dict['Train']['Wav_Length'] // hp_Dict['Sound']['Frame_Shift']

    def __call__(self, batch):
        audios, mels, pitches, audio_Singers, mel_Singers = [], [], [], [], []
        for index, (audio, mel, pitch, audio_Singer, mel_Singer) in enumerate(batch):
            max_Offset = mel.shape[0] - 2 - (self.mel_Length + 2 * hp_Dict['WaveNet']['Upsample']['Pad'])
            if max_Offset <= 0:
                continue
                
            mel_Offset = np.random.randint(0, max_Offset)
            audio_Offset = (mel_Offset + hp_Dict['WaveNet']['Upsample']['Pad']) * hp_Dict['Sound']['Frame_Shift']
            pitch_Offset = mel_Offset + hp_Dict['WaveNet']['Upsample']['Pad']

            audio = audio[audio_Offset:audio_Offset + hp_Dict['Train']['Wav_Length']]
            mel = mel[mel_Offset:mel_Offset + self.mel_Length]
            pitch = pitch[pitch_Offset:pitch_Offset + self.pitch_Length]

            audios.append(audio)
            mels.append(mel)
            pitches.append(pitch)
            audio_Singers.append(audio_Singer)
            mel_Singers.append(mel_Singer)
                
        
        audios = torch.FloatTensor(np.stack(audios, axis= 0))   # [Batch, Time]
        mels = torch.FloatTensor(np.stack(mels, axis= 0)).transpose(2, 1)   # [Batch, Time, Mel_dim] -> [Batch, Mel_dim, Time]
        pitches = torch.FloatTensor(np.stack(pitches, axis= 0))   # [Batch, Time]
        audio_Singers = torch.LongTensor(np.stack(audio_Singers, axis= 0))   # [Batch]
        mel_Singers = torch.LongTensor(np.stack(mel_Singers, axis= 0))   # [Batch]
        noises = torch.randn(size= audios.size()) # [Batch, Time]

        return audios, mels, pitches, audio_Singers, mel_Singers, noises

class Dev_Collater:
    def __call__(self, batch):
        audios, mels, pitches, original_Singers, conversion_Singers, start_Indices, end_Indices, source_Labels, conversion_Labels = zip(*[
            (audio, mel, pitch, original_Singer, conversion_Singer, start_Index, end_Index, source_Label, conversion_Label)
            for audio, mel, pitch, original_Singer, conversion_Singer, start_Index, end_Index, source_Label, conversion_Label in batch
            ])
        audios, mels, pitches = self.Stack(audios, mels, pitches, start_Indices, end_Indices)

        audios = torch.FloatTensor(np.stack(audios, axis= 0))   # [Batch, Time]
        mels = torch.FloatTensor(np.stack(mels, axis= 0)).transpose(2, 1)   # [Batch, Time, Mel_dim] -> [Batch, Mel_dim, Time]
        pitches = torch.FloatTensor(np.stack(pitches, axis= 0))   # [Batch, Time]
        original_Singers = torch.LongTensor(np.stack(original_Singers, axis= 0))   # [Batch]
        conversion_Singers = torch.LongTensor(np.stack(conversion_Singers, axis= 0))   # [Batch]
        noises = torch.randn(size= audios.size()) # [Batch, Time]

        return audios, mels, pitches, original_Singers, conversion_Singers, noises, source_Labels, conversion_Labels
    
    def Stack(self, audios, mels, pitches, start_Indices, end_Indices):
        frame_Shift = hp_Dict['Sound']['Frame_Shift']
        upsample_Pad = hp_Dict['WaveNet']['Upsample']['Pad']

        audios, mels, pitches = zip(*[
            (
                audio[start_Index * frame_Shift:end_Index * frame_Shift],
                np.pad(mel, pad_width=[[upsample_Pad, upsample_Pad], [0, 0]], mode= 'reflect')[start_Index:end_Index + upsample_Pad * 2],
                pitch[start_Index:end_Index]
                )
            for audio, mel, pitch, start_Index, end_Index in zip(audios, mels, pitches, start_Indices, end_Indices)
            ])
        
        max_Audio_Length = max([audio.shape[0] for audio in audios])
        max_Mel_Length = max([mel.shape[0] for mel in mels])
        max_Pitch_Length = max([pitch.shape[0] for pitch in pitches])
        audios = np.stack([
            np.pad(audio, [0, max_Audio_Length - audio.shape[0]], constant_values= 0)
            for audio in audios
            ], axis= 0)
        mels =  np.stack([
            np.pad(mel, [0, max_Mel_Length - mel.shape[0]], constant_values= -hp_Dict['Sound']['Max_Abs_Mel'])
            for mel in mels
            ], axis= 0)
        pitches =  np.stack([
            np.pad(pitch, [0, max_Pitch_Length - pitch.shape[0]], constant_values= 0)
            for pitch in pitches
            ], axis= 0)

        return audios, mels, pitches


