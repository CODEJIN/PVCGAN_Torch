import torch
import numpy as np
import logging, yaml, os, sys, argparse, time
from tqdm import tqdm
from collections import defaultdict
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from scipy.io import wavfile
from random import choice

from Modules import Encoder, Singer_Classification_Network, Pitch_Regression_Network, Generator, Discriminator, MultiResolutionSTFTLoss
from Datasets import AccumulationDataset, TrainDataset, DevDataset, Accumulation_Collater, Train_Collater, Dev_Collater
from Radam import RAdam

from Audio import melspectrogram
from yin import pitch_calc

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

if not hp_Dict['Device'] is None:
    os.environ['CUDA_VISIBLE_DEVICES']= hp_Dict['Device']

if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)

logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

class Trainer:
    def __init__(self, steps= 0):
        self.steps = steps
        self.epochs = 0

        self.Datset_Generate()        
        self.Model_Generate()

        self.loss_Dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }
        self.writer = SummaryWriter(hp_Dict['Log_Path'])

        if self.steps > 0:
            self.Load_Checkpoint()


    def Datset_Generate(self):
        accumulation_Dataset = AccumulationDataset()
        train_Dataset = TrainDataset()
        dev_Dataset = DevDataset()
        logging.info('The number of base train files = {}.'.format(len(accumulation_Dataset)))
        logging.info('The number of train patterns = {}.'.format(len(train_Dataset)))
        logging.info('The number of development patterns = {}.'.format(len(dev_Dataset)))

        accumulation_Collater = Accumulation_Collater()
        train_Collater = Train_Collater()
        dev_Collater = Dev_Collater()

        self.dataLoader_Dict = {}
        self.dataLoader_Dict['Accumulation'] = torch.utils.data.DataLoader(
            dataset= accumulation_Dataset,
            shuffle= True,
            collate_fn= accumulation_Collater,
            batch_size= hp_Dict['Train']['Batch_Size'],
            num_workers= hp_Dict['Train']['Num_Workers'],
            pin_memory= True
            )
        self.dataLoader_Dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_Dataset,
            shuffle= True,
            collate_fn= train_Collater,
            batch_size= hp_Dict['Train']['Batch_Size'],
            num_workers= hp_Dict['Train']['Num_Workers'],
            pin_memory= True
            )
        self.dataLoader_Dict['Dev'] = torch.utils.data.DataLoader(
            dataset= dev_Dataset,
            shuffle= False,
            collate_fn= dev_Collater,
            batch_size= hp_Dict['Train']['Batch_Size'],
            num_workers= hp_Dict['Train']['Num_Workers'],
            pin_memory= True
            )

        self.train_Dataset = train_Dataset

    def Model_Generate(self):
        self.model_Dict = {
            'Generator': Generator().to(device),
            'Discriminator': Discriminator().to(device),
            'Encoder': Encoder().to(device),
            'Singer_Classification_Network': Singer_Classification_Network().to(device),
            'Pitch_Regression_Network': Pitch_Regression_Network().to(device),            
            }
        self.criterion_Dict = {
            'STFT': MultiResolutionSTFTLoss(
                fft_sizes= hp_Dict['STFT_Loss_Resolution']['FFT_Sizes'],
                shift_lengths= hp_Dict['STFT_Loss_Resolution']['Shfit_Lengths'],
                win_lengths= hp_Dict['STFT_Loss_Resolution']['Win_Lengths'],
                ).to(device),
            'MSE': torch.nn.MSELoss().to(device),
            'CE': torch.nn.CrossEntropyLoss().to(device),
            'MAE': torch.nn.L1Loss().to(device)
            }
        
        self.optimizer_Dict = {
            'Generator': RAdam(
                params= list(self.model_Dict['Encoder'].parameters()) + list(self.model_Dict['Generator'].parameters()),
                lr= hp_Dict['Train']['Learning_Rate']['Generator']['Initial'],
                eps= hp_Dict['Train']['Learning_Rate']['Generator']['Epsilon'],
                ),
            'Discriminator': RAdam(
                params= self.model_Dict['Discriminator'].parameters(),
                lr= hp_Dict['Train']['Learning_Rate']['Discriminator']['Initial'],
                eps= hp_Dict['Train']['Learning_Rate']['Discriminator']['Epsilon'],
                ),
            'Confuser': RAdam(
                params= list(self.model_Dict['Singer_Classification_Network'].parameters()) + list(self.model_Dict['Pitch_Regression_Network'].parameters()),
                lr= hp_Dict['Train']['Learning_Rate']['Confuser']['Initial'],
                eps= hp_Dict['Train']['Learning_Rate']['Confuser']['Epsilon'],
                )
            }
        self.scheduler_Dict = {
            'Generator': torch.optim.lr_scheduler.StepLR(
                optimizer= self.optimizer_Dict['Generator'],
                step_size= hp_Dict['Train']['Learning_Rate']['Generator']['Decay_Step'],
                gamma= hp_Dict['Train']['Learning_Rate']['Generator']['Decay_Rate'],
                ),
            'Discriminator': torch.optim.lr_scheduler.StepLR(
                optimizer= self.optimizer_Dict['Discriminator'],
                step_size= hp_Dict['Train']['Learning_Rate']['Discriminator']['Decay_Step'],
                gamma= hp_Dict['Train']['Learning_Rate']['Discriminator']['Decay_Rate'],
                ),
            'Confuser': torch.optim.lr_scheduler.StepLR(
                optimizer= self.optimizer_Dict['Confuser'],
                step_size= hp_Dict['Train']['Learning_Rate']['Confuser']['Decay_Step'],
                gamma= hp_Dict['Train']['Learning_Rate']['Confuser']['Decay_Rate'],
                )
            }
        logging.info(self.model_Dict['Generator'])
        logging.info(self.model_Dict['Discriminator'])
        logging.info(self.model_Dict['Encoder'])
        logging.info(self.model_Dict['Singer_Classification_Network'])
        logging.info(self.model_Dict['Pitch_Regression_Network'])


    def Train_Step(self, audios, mels, pitches, audio_Singers, mel_Singers, noises):
        loss_Dict = {}

        audios = audios.to(device)
        mels = mels.to(device)
        pitches = pitches.to(device)
        audio_Singers = audio_Singers.to(device)
        mel_Singers = mel_Singers.to(device)
        noises = noises.to(device)

        # Generator
        encodings = self.model_Dict['Encoder'](mels)
        fake_Audios = self.model_Dict['Generator'](noises, encodings, audio_Singers, pitches)
        singer_Logits = self.model_Dict['Singer_Classification_Network'](encodings)
        pitch_Logits = self.model_Dict['Pitch_Regression_Network'](encodings)
        
        loss_Dict['Singer'] = self.criterion_Dict['CE'](singer_Logits, mel_Singers)        
        loss_Dict['Pitch'] = self.criterion_Dict['MAE'](pitch_Logits, pitches)
        loss_Dict['Confuser'] = \
            hp_Dict['Train']['Adversarial_Weight']['Singer_Classification'] * loss_Dict['Singer'] + \
            hp_Dict['Train']['Adversarial_Weight']['Pitch_Regression'] * loss_Dict['Pitch']

        loss_Dict['Spectral_Convergence'], loss_Dict['Magnitude'] = self.criterion_Dict['STFT'](fake_Audios, audios)

        loss_Dict['Generator'] = loss_Dict['Spectral_Convergence'] + loss_Dict['Magnitude'] - loss_Dict['Confuser']

        if self.steps > hp_Dict['Train']['Discriminator_Delay']:
            fake_Discriminations = self.model_Dict['Discriminator'](fake_Audios)
            loss_Dict['Adversarial'] = self.criterion_Dict['MSE'](
                fake_Discriminations,
                fake_Discriminations.new_ones(fake_Discriminations.size())
                )
            loss_Dict['Generator'] += hp_Dict['Train']['Adversarial_Weight']['Discriminator'] * loss_Dict['Adversarial']
        
        
        self.optimizer_Dict['Generator'].zero_grad()
        loss_Dict['Generator'].backward()
        torch.nn.utils.clip_grad_norm_(
            parameters= list(self.model_Dict['Encoder'].parameters()) + list(self.model_Dict['Generator'].parameters()),
            max_norm= hp_Dict['Train']['Generator_Gradient_Norm']
            )
        self.optimizer_Dict['Generator'].step()
        self.scheduler_Dict['Generator'].step()
        
        # Confuser
        singer_Logits = self.model_Dict['Singer_Classification_Network'](encodings.detach())
        pitch_Logits = self.model_Dict['Pitch_Regression_Network'](encodings.detach())
        loss_Dict['Singer'] = self.criterion_Dict['CE'](singer_Logits, mel_Singers)        
        loss_Dict['Pitch'] = self.criterion_Dict['MAE'](pitch_Logits, pitches)
        loss_Dict['Confuser'] = \
            hp_Dict['Train']['Adversarial_Weight']['Singer_Classification'] * loss_Dict['Singer'] + \
            hp_Dict['Train']['Adversarial_Weight']['Pitch_Regression'] * loss_Dict['Pitch']

        self.optimizer_Dict['Confuser'].zero_grad()
        loss_Dict['Confuser'].backward()
        torch.nn.utils.clip_grad_norm_(
            parameters= list(self.model_Dict['Singer_Classification_Network'].parameters()) + list(self.model_Dict['Pitch_Regression_Network'].parameters()),
            max_norm= hp_Dict['Train']['Confuser_Gradient_Norm']
            )
        self.optimizer_Dict['Confuser'].step()
        self.scheduler_Dict['Confuser'].step()

        # Discriminator
        if self.steps > hp_Dict['Train']['Discriminator_Delay']:
            real_Discriminations = self.model_Dict['Discriminator'](audios)
            fake_Discriminations = self.model_Dict['Discriminator'](fake_Audios.detach())

            loss_Dict['Real'] = self.criterion_Dict['MSE'](
                real_Discriminations,
                real_Discriminations.new_ones(real_Discriminations.size())
                )
            loss_Dict['Fake'] = self.criterion_Dict['MSE'](
                fake_Discriminations,
                fake_Discriminations.new_zeros(fake_Discriminations.size())
                )
            loss_Dict['Discriminator'] = loss_Dict['Real'] + loss_Dict['Fake']

            self.optimizer_Dict['Discriminator'].zero_grad()
            loss_Dict['Discriminator'].backward()
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model_Dict['Discriminator'].parameters(),
                max_norm= hp_Dict['Train']['Discriminator_Gradient_Norm']
                )
            self.optimizer_Dict['Discriminator'].step()
            self.scheduler_Dict['Discriminator'].step()
        
        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_Dict.items():
            self.loss_Dict['Train'][tag] += loss

    def Train_Epoch(self):
        if self.epochs % hp_Dict['Train']['Accumulation_Inverval'] == 0:
            self.Data_Accumulation()

        for step, (audios, mels, pitches, audio_Singers, mel_Singers, noises) in enumerate(self.dataLoader_Dict['Train'], 1):
            self.Train_Step(audios, mels, pitches, audio_Singers, mel_Singers, noises)
            
            if self.steps % hp_Dict['Train']['Checkpoint_Save_Interval'] == 0:
                self.Save_Checkpoint()

            if self.steps % hp_Dict['Train']['Logging_Interval'] == 0:
                self.loss_Dict['Train'] = {
                    tag: loss / hp_Dict['Train']['Logging_Interval']
                    for tag, loss in self.loss_Dict['Train'].items()
                    }
                self.Write_to_Tensorboard('Train', self.loss_Dict['Train'])
                self.loss_Dict['Train'] = defaultdict(float)

            if self.steps % hp_Dict['Train']['Evaluation_Interval'] == 0:
                self.Evaluation_Epoch()
            
            if self.steps >= hp_Dict['Train']['Max_Step']:
                return

        self.epochs += 1

    
    @torch.no_grad()
    def Evaluation_Step(self, audios, mels, pitches, singers, noises):
        loss_Dict = {}

        audios = audios.to(device)
        mels = mels.to(device)
        pitches = pitches.to(device)
        singers = singers.to(device)
        noises = noises.to(device)

        encodings = self.model_Dict['Encoder'](mels)
        fake_Audios = self.model_Dict['Generator'](noises, encodings, singers, pitches)
        singer_Logits = self.model_Dict['Singer_Classification_Network'](encodings)
        pitch_Logits = self.model_Dict['Pitch_Regression_Network'](encodings)
        
        loss_Dict['Singer'] = self.criterion_Dict['CE'](singer_Logits, singers)        
        loss_Dict['Pitch'] = self.criterion_Dict['MAE'](pitch_Logits, pitches)
        loss_Dict['Confuser'] = \
            hp_Dict['Train']['Adversarial_Weight']['Singer_Classification'] * loss_Dict['Singer'] + \
            hp_Dict['Train']['Adversarial_Weight']['Pitch_Regression'] * loss_Dict['Pitch']

        loss_Dict['Spectral_Convergence'], loss_Dict['Magnitude'] = self.criterion_Dict['STFT'](fake_Audios, audios)

        loss_Dict['Generator'] = loss_Dict['Spectral_Convergence'] + loss_Dict['Magnitude'] - loss_Dict['Confuser']

        if self.steps > hp_Dict['Train']['Discriminator_Delay']:
            fake_Discriminations = self.model_Dict['Discriminator'](fake_Audios)
            loss_Dict['Adversarial'] = self.criterion_Dict['MSE'](
                fake_Discriminations,
                fake_Discriminations.new_ones(fake_Discriminations.size())
                )
            loss_Dict['Generator'] += hp_Dict['Train']['Adversarial_Weight']['Discriminator'] * loss_Dict['Adversarial']
                       
        if self.steps > hp_Dict['Train']['Discriminator_Delay']:
            real_Discriminations = self.model_Dict['Discriminator'](audios)
            fake_Discriminations = self.model_Dict['Discriminator'](fake_Audios.detach())

            loss_Dict['Real'] = self.criterion_Dict['MSE'](
                real_Discriminations,
                real_Discriminations.new_ones(real_Discriminations.size())
                )
            loss_Dict['Fake'] = self.criterion_Dict['MSE'](
                fake_Discriminations,
                fake_Discriminations.new_zeros(fake_Discriminations.size())
                )
            loss_Dict['Discriminator'] = loss_Dict['Real'] + loss_Dict['Fake']

        for tag, loss in loss_Dict.items():
            self.loss_Dict['Evaluation'][tag] += loss

    @torch.no_grad()
    def Inference_Step(self, audios, mels, pitches, singers, noises):
        mels = mels.to(device)
        pitches = pitches.to(device)
        singers = singers.to(device)
        noises = noises.to(device)

        encodings = self.model_Dict['Encoder'](mels)
        fakes = self.model_Dict['Generator'](noises, encodings, singers, pitches).cpu().numpy()

        os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps)).replace("\\", "/"), exist_ok= True)

        for index, (real, fake) in enumerate(zip(audios, fakes)):            
            new_Figure = plt.figure(figsize=(80, 10 * 2), dpi=100)
            plt.subplot(211)
            plt.plot(real)
            plt.title('Original wav    Index: {}'.format(index))
            plt.subplot(212)            
            plt.plot(fake)
            plt.title('Fake wav    Index: {}'.format(index))
            plt.tight_layout()
            plt.savefig(
                os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), 'Step-{}.IDX_{}.PNG'.format(self.steps, index)).replace("\\", "/")
                )
            plt.close(new_Figure)

            wavfile.write(
                filename= os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), 'Step-{}.IDX_{}.WAV'.format(self.steps, index)).replace("\\", "/"),
                data= (fake * 32767.5).astype(np.int16),
                rate= hp_Dict['Sound']['Sample_Rate']
                )

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation.'.format(self.steps))

        for model in self.model_Dict.values():
            model.eval()

        for step, (audios, mels, pitches, original_Singers, inference_Singers, noises) in tqdm(enumerate(self.dataLoader_Dict['Dev'], 1), desc='[Evaluation]'):
            self.Evaluation_Step(audios, mels, pitches, original_Singers, noises)
            self.Inference_Step(audios, mels, pitches, inference_Singers, noises)

        self.loss_Dict['Evaluation'] = {
            tag: loss / step
            for tag, loss in self.loss_Dict['Evaluation'].items()
            }
        self.Write_to_Tensorboard('Evaluation', self.loss_Dict['Evaluation'])
        self.loss_Dict['Evaluation'] = defaultdict(float)
        
        for model in self.model_Dict.values():
            model.train()


    @torch.no_grad()
    def Back_Translate_Step(self, mels, pitches, singers, noises):        
        mels = mels.to(device)
        pitches = pitches.to(device)
        singers = singers.to(device)
        noises = noises.to(device)

        encodings = self.model_Dict['Encoder'](mels)
        return self.model_Dict['Generator'](noises, encodings, singers, pitches).cpu().numpy()

    def Data_Accumulation(self):
        def Mixup(audio):
            max_Offset = audio.shape[0] - hp_Dict['Train']['Wav_Length'] * 2
            offset1 = np.random.randint(0, max_Offset)
            offset2 = np.random.randint(0, max_Offset)
            beta = np.random.uniform(
                low= hp_Dict['Train']['Mixup']['Min_Beta'],
                high= hp_Dict['Train']['Mixup']['Max_Beta'],
                )

            new_Audio = \
                beta * audio[offset1:offset1 + hp_Dict['Train']['Wav_Length'] * 2] + \
                (1 - beta) * audio[offset2:offset2 + hp_Dict['Train']['Wav_Length'] * 2]

            new_Mel = np.transpose(melspectrogram(
                y= new_Audio,
                num_freq= hp_Dict['Sound']['Spectrogram_Dim'],
                hop_length= hp_Dict['Sound']['Frame_Shift'],
                win_length= hp_Dict['Sound']['Frame_Length'],        
                num_mels= hp_Dict['Sound']['Mel_Dim'],
                sample_rate= hp_Dict['Sound']['Sample_Rate'],
                max_abs_value= hp_Dict['Sound']['Max_Abs_Mel']
                ))

            new_Pitch = pitch_calc(
                sig= new_Audio,
                sr= hp_Dict['Sound']['Sample_Rate'],
                w_len= hp_Dict['Sound']['Frame_Length'],
                w_step= hp_Dict['Sound']['Frame_Shift'],
                confidence_threshold= hp_Dict['Sound']['Confidence_Threshold'],
                gaussian_smoothing_sigma = hp_Dict['Sound']['Gaussian_Smoothing_Sigma']
                )

            return new_Audio, new_Mel, new_Pitch

        def Back_Translate(mels, pitches, singers, noises):
            new_Audios = self.Back_Translate_Step(
                mels= mels,
                pitches= pitches,
                singers= singers,
                noises= noises
                )

            new_Mels = [
                np.transpose(melspectrogram(
                    y= new_Audio,
                    num_freq= hp_Dict['Sound']['Spectrogram_Dim'],
                    hop_length= hp_Dict['Sound']['Frame_Shift'],
                    win_length= hp_Dict['Sound']['Frame_Length'],        
                    num_mels= hp_Dict['Sound']['Mel_Dim'],
                    sample_rate= hp_Dict['Sound']['Sample_Rate'],
                    max_abs_value= hp_Dict['Sound']['Max_Abs_Mel']
                    ))
                for new_Audio in new_Audios
                ]

            return new_Mels

        mixup_List = []
        back_Translate_List = []
        for total_Audios, audios, mels, pitches, singers, noises in tqdm(self.dataLoader_Dict['Accumulation'], desc='[Accumulation]'):
            #Mixup
            if hp_Dict['Train']['Mixup']['Use'] and self.steps > hp_Dict['Train']['Mixup']['Apply_Delay']:
                for audio, singer in zip(total_Audios, singers.numpy()):
                    mixup_Audio, mixup_Mel, mixup_Pitch = Mixup(audio)                
                    mixup_List.append((
                        mixup_Audio,
                        mixup_Mel,
                        mixup_Pitch,
                        singer,
                        singer
                        ))
            
            #Backtranslate
            if hp_Dict['Train']['Back_Translate']['Use'] and self.steps > hp_Dict['Train']['Back_Translate']['Apply_Delay']:
                mel_Singers = torch.LongTensor(np.stack([
                    choice([x for x in range(hp_Dict['Num_Singers']) if x != singer])
                    for singer in singers
                    ], axis= 0))
                back_Translate_Mels = Back_Translate(mels, pitches, mel_Singers, noises)
                for audio, back_Translate_Mel, pitch, audio_Singer, mel_Singer in zip(
                    audios.numpy(), back_Translate_Mels, pitches.numpy(), singers.numpy(), mel_Singers.numpy()):
                    back_Translate_List.append((
                        audio,
                        back_Translate_Mel,
                        pitch,
                        audio_Singer,
                        mel_Singer
                        ))

        self.train_Dataset.Accumulation_Renew(
            mixup_Pattern_List= mixup_List,
            back_Translate_Pattern_List= back_Translate_List
            )


    def Load_Checkpoint(self):
        state_Dict = torch.load(
            os.path.join(hp_Dict['Checkpoint_Path'], 'S_{}.pkl'.format(self.steps).replace('\\', '/')),
            map_location= 'cpu'
            )

        self.model_Dict['Generator'].load_state_dict(state_Dict['Model']['Generator'])
        self.model_Dict['Discriminator'].load_state_dict(state_Dict['Model']['Discriminator'])
        
        self.optimizer_Dict['Generator'].load_state_dict(state_Dict['Optimizer']['Generator'])
        self.optimizer_Dict['Discriminator'].load_state_dict(state_Dict['Optimizer']['Discriminator'])

        self.scheduler_Dict['Generator'].load_state_dict(state_Dict['Scheduler']['Generator'])
        self.scheduler_Dict['Discriminator'].load_state_dict(state_Dict['Scheduler']['Discriminator'])
        
        self.steps = state_Dict['Steps']
        self.epochs = state_Dict['Epochs']

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    def Save_Checkpoint(self):
        os.makedirs(hp_Dict['Checkpoint_Path'], exist_ok= True)

        state_Dict = {
            'Model': {
                'Generator': self.model_Dict['Generator'].state_dict(),
                'Discriminator': self.model_Dict['Discriminator'].state_dict(),
                },
            'Optimizer': {
                'Generator': self.optimizer_Dict['Generator'].state_dict(),
                'Discriminator': self.optimizer_Dict['Discriminator'].state_dict(),
                },
            'Scheduler': {
                'Generator': self.scheduler_Dict['Generator'].state_dict(),
                'Discriminator': self.scheduler_Dict['Discriminator'].state_dict(),
                },
            'Steps': self.steps,
            'Epochs': self.epochs,
            }

        torch.save(
            state_Dict,
            os.path.join(hp_Dict['Checkpoint_Path'], 'S_{}.pkl'.format(self.steps).replace('\\', '/'))
            )

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))
       

    def Train(self):
        self.tqdm = tqdm(
            initial= self.steps,
            total= hp_Dict['Train']['Max_Step'],
            desc='[Training]'
            )

        while self.steps < hp_Dict['Train']['Max_Step']:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)
            
        self.tqdm.close()
        logging.info('Finished training.')

    def Write_to_Tensorboard(self, category, loss_Dict):
        for tag, loss in loss_Dict.items():
            self.writer.add_scalar('{}/{}'.format(category, tag), loss, self.steps)

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', '--steps', default= 0, type= int)
    args = argParser.parse_args()
    
    new_Trainer = Trainer(steps= args.steps)    
    new_Trainer.Train()