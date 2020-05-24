import numpy as np
import yaml, pickle, os, librosa, argparse
from concurrent.futures import ThreadPoolExecutor as PE
from collections import deque
from threading import Thread
from Audio import melspectrogram
from yin import pitch_calc

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

using_Extension = [x.upper() for x in ['.wav', '.m4a', '.flac']]

def Pattern_Generate(path, keyword_Index_Dict, top_db= 60, reverse= False, invert= False):
    sig = librosa.core.load(
        path,
        sr= hp_Dict['Sound']['Sample_Rate']
        )[0]

    sig = librosa.effects.trim(sig, top_db= top_db, frame_length= 32, hop_length= 16)[0] * 0.99
    sig = librosa.util.normalize(sig)
    
    mel = np.transpose(melspectrogram(
        y= sig,
        num_freq= hp_Dict['Sound']['Spectrogram_Dim'],        
        hop_length= hp_Dict['Sound']['Frame_Shift'],
        win_length= hp_Dict['Sound']['Frame_Length'],
        num_mels= hp_Dict['Sound']['Mel_Dim'],
        sample_rate= hp_Dict['Sound']['Sample_Rate'],
        max_abs_value= hp_Dict['Sound']['Max_Abs_Mel']
        ))

    pitch = pitch_calc(
        sig= sig,
        sr= hp_Dict['Sound']['Sample_Rate'],
        w_len= hp_Dict['Sound']['Frame_Length'],
        w_step= hp_Dict['Sound']['Frame_Shift'],
        confidence_threshold= hp_Dict['Sound']['Confidence_Threshold'],
        gaussian_smoothing_sigma = hp_Dict['Sound']['Gaussian_Smoothing_Sigma']
        )

    singer_ID = None
    for keyword, index in keyword_Index_Dict.items():
        if keyword in path:
            singer_ID = index
            break
    if singer_ID is None:
        raise ValueError('No keyword in keyword_Index_Dict.')

    return sig, mel, pitch, singer_ID

def Pattern_File_Generate(path, keyword_Index_Dict, dataset, file_Prefix='', display_Prefix = '', top_db= 60):
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
            
            print('[{}]'.format(display_Prefix), '{}'.format(path), '->', '{}'.format(pickle_File_Name))


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

    with open(os.path.join(hp_Dict['Train']['Pattern_Path'], hp_Dict['Train']['Metadata_File'].upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol=2)

    print('Metadata generate done.')


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-nus48e', '--nus48e_path', required=False)
    argParser.add_argument('-sex', '--sex_type', required= False, default= 'M')
    # argParser.add_argument("-mw", "--max_worker", required=False)
    # argParser.set_defaults(max_worker = 10)
    argument_Dict = vars(argParser.parse_args())

    if not argument_Dict['sex_type'] in ['M', 'F', 'B']:
        raise ValueError('Unsupported sex type. Only M, F, or B is supported')

    total_Pattern_Count = 0
    keyword_Index_Dict = {}

    if not argument_Dict['nus48e_path'] is None:
        nus48e_File_Path_List, nus48e_Singer_Dict, nus48e_Keyword_List = NUS48E_Info_Load(
            nus48e_Path= argument_Dict['nus48e_path'],
            sex_Type= argument_Dict['sex_type']
            )
        total_Pattern_Count += len(nus48e_File_Path_List)
        
        for index, keyword in enumerate(nus48e_Keyword_List, len(keyword_Index_Dict)):
            if keyword in keyword_Index_Dict.keys():
                raise ValueError('There is an overlapped keyword: \'{}\'.'.format(keyword))
            keyword_Index_Dict[keyword] = index

    if total_Pattern_Count == 0:
        raise ValueError('Total pattern count is zero.')
    
    os.makedirs(hp_Dict['Train']['Pattern_Path'], exist_ok= True)
    total_Generated_Pattern_Count = 0
    # with PE(max_workers = int(argument_Dict['max_worker'])) as pe:
    if not argument_Dict['nus48e_path'] is None:
        for index, file_Path in enumerate(nus48e_File_Path_List):
            Pattern_File_Generate(
                file_Path,
                keyword_Index_Dict,                    
                'NUS48E',
                nus48e_Singer_Dict[file_Path],
                'NUS48E {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                    index,
                    len(nus48e_File_Path_List),
                    total_Generated_Pattern_Count,
                    total_Pattern_Count
                    ),
                20
                )
            total_Generated_Pattern_Count += 1


    Metadata_Generate(keyword_Index_Dict)