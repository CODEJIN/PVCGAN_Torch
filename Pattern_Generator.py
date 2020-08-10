import numpy as np
import yaml, pickle, os, librosa, argparse
from concurrent.futures import ThreadPoolExecutor as PE
from collections import deque
from threading import Thread
from tqdm import tqdm

from Audio import Audio_Prep, Mel_Generate
from yin import pitch_calc

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

using_Extension = [x.upper() for x in ['.wav', '.m4a', '.flac']]

def Pitch_Generate(audio):
    pitch = pitch_calc(
        sig= audio,
        sr= hp_Dict['Sound']['Sample_Rate'],
        w_len= hp_Dict['Sound']['Frame_Length'],
        w_step= hp_Dict['Sound']['Frame_Shift'],
        confidence_threshold= hp_Dict['Sound']['Confidence_Threshold'],
        gaussian_smoothing_sigma = hp_Dict['Sound']['Gaussian_Smoothing_Sigma']
        )
    return (pitch - np.min(pitch)) / (np.max(pitch) - np.min(pitch) + 1e-7)

def Pattern_Generate(audio= None, path= None, keyword_Index_Dict= None, top_db= 60, reverse= False, invert= False):
    audio = audio if not audio is None else Audio_Prep(path, hp_Dict['Sound']['Sample_Rate'], top_db)
    if reverse:
        audio = audio[::-1]
    if invert:
        audio = -audio

    mel = Mel_Generate(
        audio= audio,
        sample_rate= hp_Dict['Sound']['Sample_Rate'],
        num_frequency= hp_Dict['Sound']['Spectrogram_Dim'],
        num_mel= hp_Dict['Sound']['Mel_Dim'],
        window_length= hp_Dict['Sound']['Frame_Length'],
        hop_length= hp_Dict['Sound']['Frame_Shift'],
        mel_fmin= hp_Dict['Sound']['Mel_F_Min'],
        mel_fmax= hp_Dict['Sound']['Mel_F_Max'],
        max_abs_value= hp_Dict['Sound']['Max_Abs_Mel']
        )
    pitch = Pitch_Generate(audio)

    singer_ID = None
    if not keyword_Index_Dict is None:
        for keyword, index in keyword_Index_Dict.items():
            if keyword in path:
                singer_ID = index
                break
        if singer_ID is None:
            raise ValueError('No keyword in keyword_Index_Dict.')

    return audio, mel, pitch, singer_ID

def Pattern_File_Generate(path, keyword_Index_Dict, dataset, file_Prefix='', top_db= 60):
    for reverse in [False, True]:
        for invert in [False, True]:
            sig, mel, pitch, singer_ID = Pattern_Generate(
                path= path,
                keyword_Index_Dict= keyword_Index_Dict,
                top_db= top_db,
                reverse= reverse,
                invert= invert
                )
            
            new_Pattern_Dict = {
                'Signal': sig.astype(np.float32),
                'Mel': mel.astype(np.float32),
                'Pitch': pitch.astype(np.float32),
                'Singer_ID': singer_ID,
                'Dataset': dataset,
                }

            pickle_File_Name = '{}.{}{}{}{}.PICKLE'.format(
                dataset,
                file_Prefix,
                os.path.splitext(os.path.basename(path))[0],
                '.REV' if reverse else '',
                '.INV' if invert else '',
                ).upper()

            with open(os.path.join(hp_Dict['Train']['Pattern_Path'], pickle_File_Name).replace("\\", "/"), 'wb') as f:
                pickle.dump(new_Pattern_Dict, f, protocol=4)


def NUS48E_Info_Load(nus48e_Path, sex_Type):
    wav_Path_List = []
    singer_Dict = {}

    sex_Dict = {
        'ADIZ': 'F',
        'JLEE': 'M',
        'JTAN': 'M',
        'KENN': 'M',
        'MCUR': 'F',
        'MPOL': 'F',
        'MPUR': 'F',
        'NJAT': 'F',
        'PMAR': 'F',
        'SAMF': 'M',
        'VKOW': 'M',
        'ZHIY': 'M',
        }
    sex_Type = sex_Type.upper()

    for root, _, files in os.walk(nus48e_Path):
        root = root.replace('\\', '/')
        for file in files:
            if root.strip().split('/')[-1].upper() != 'sing'.upper():
                continue
            elif not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            path = os.path.join(root, file).replace('\\', '/')
            singer = root.strip().split('/')[-2]

            if sex_Type != 'B' and sex_Dict[singer] != sex_Type:
                continue

            wav_Path_List.append(path)
            singer_Dict[path] = singer

    print('NUS-48E info generated: {}'.format(len(wav_Path_List)))
    return wav_Path_List, singer_Dict, list(sorted(list(set(singer_Dict.values()))))


def Metadata_Generate(keyword_Index_Dict):
    new_Metadata_Dict = {
        'Sample_Rate': hp_Dict['Sound']['Sample_Rate'],
        'Confidence_Threshold': hp_Dict['Sound']['Confidence_Threshold'],
        'Gaussian_Smoothing_Sigma': hp_Dict['Sound']['Gaussian_Smoothing_Sigma'],
        'Keyword_Index_Dict': keyword_Index_Dict,
        'File_List': [],
        'Sig_Length_Dict': {},
        'Pitch_Length_Dict': {},
        'Singer_Index_Dict': {},
        'Dataset_Dict': {},
        }

    files_TQDM = tqdm(
        total= sum([len(files) for root, _, files in os.walk(hp_Dict['Train']['Pattern_Path'])]),
        desc= 'Metadata'
        )

    for root, _, files in os.walk(hp_Dict['Train']['Pattern_Path']):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_Dict = pickle.load(f)                
                try:
                    new_Metadata_Dict['Sig_Length_Dict'][file] = pattern_Dict['Signal'].shape[0]
                    new_Metadata_Dict['Pitch_Length_Dict'][file] = pattern_Dict['Pitch'].shape[0]
                    new_Metadata_Dict['Singer_Index_Dict'][file] = pattern_Dict['Singer_ID']
                    new_Metadata_Dict['Dataset_Dict'][file] = pattern_Dict['Dataset']
                    new_Metadata_Dict['File_List'].append(file)
                except:
                    print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))
                files_TQDM.update(1)

    with open(os.path.join(hp_Dict['Train']['Pattern_Path'], hp_Dict['Train']['Metadata_File'].upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol=2)

    print('Metadata generate done.')


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-nus48e', '--nus48e_path', required=False)
    argParser.add_argument('-sex', '--sex_type', required= False, default= 'B')
    args = argParser.parse_args()

    if not args.sex_type in ['M', 'F', 'B']:
        raise ValueError('Unsupported sex type. Only M, F, or B is supported')

    total_Pattern_Count = 0
    keyword_Index_Dict = {}

    if not args.nus48e_path is None:
        nus48e_File_Path_List, nus48e_Singer_Dict, nus48e_Keyword_List = NUS48E_Info_Load(
            nus48e_Path= args.nus48e_path,
            sex_Type= args.sex_type
            )
        total_Pattern_Count += len(nus48e_File_Path_List)
        
        for index, keyword in enumerate(nus48e_Keyword_List, len(keyword_Index_Dict)):
            if keyword in keyword_Index_Dict.keys():
                raise ValueError('There is an overlapped keyword: \'{}\'.'.format(keyword))
            keyword_Index_Dict[keyword] = index

    if total_Pattern_Count == 0:
        raise ValueError('Total pattern count is zero.')
    
    os.makedirs(hp_Dict['Train']['Pattern_Path'], exist_ok= True)
    
    if not args.nus48e_path is None:
        for index, file_Path in tqdm(
            enumerate(nus48e_File_Path_List),
            desc= 'Pattern',
            total= len(nus48e_File_Path_List)
            ):
            Pattern_File_Generate(
                file_Path,
                keyword_Index_Dict,
                'NUS48E',
                nus48e_Singer_Dict[file_Path],
                20
                )

    Metadata_Generate(keyword_Index_Dict)