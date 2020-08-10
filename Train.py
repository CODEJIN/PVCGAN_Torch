import torch
import numpy as np
import logging, yaml, os, sys, argparse, time, math
from tqdm import tqdm
from collections import defaultdict
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from scipy.io import wavfile
from random import choice

from Modules import PVCGAN, MultiResolutionSTFTLoss
from Datasets import AccumulationDataset, TrainDataset, DevDataset, Accumulation_Collater, Train_Collater, Dev_Collater
from Radam import RAdam
from Noam_Scheduler import Modified_Noam_Scheduler

from Pattern_Generator import Pattern_Generate
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
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    )

if hp_Dict['Use_Mixed_Precision']:
    try:
        from apex import amp
    except:
        logging.info('There is no apex modules in the environment. Mixed precision does not work.')
        hp_Dict['Use_Mixed_Precision'] = False

class Trainer:
    def __init__(self, steps= 0):
        self.steps = steps
        self.epochs = 0

        self.Datset_Generate()
        self.Model_Generate()

        self.scalar_Dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }

        self.writer_Dict = {
            'Train': SummaryWriter(os.path.join(hp_Dict['Log_Path'], 'Train')),
            'Evaluation': SummaryWriter(os.path.join(hp_Dict['Log_Path'], 'Evaluation')),
            }

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

    def Model_Generate(self):
        self.model = PVCGAN().to(device, )
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
        
        self.optimizer = RAdam(
            params= self.model.parameters(),
            lr= hp_Dict['Train']['Learning_Rate']['Initial'],
            betas=(hp_Dict['Train']['ADAM']['Beta1'], hp_Dict['Train']['ADAM']['Beta2']),
            eps= hp_Dict['Train']['ADAM']['Epsilon'],
            )
        self.scheduler = Modified_Noam_Scheduler(
            optimizer= self.optimizer,
            base= hp_Dict['Train']['Learning_Rate']['Base']
            )

        if hp_Dict['Use_Mixed_Precision']:
            self.model, self.optimizer = amp.initialize(
                models=self.model,
                optimizers=self.optimizer
                )

        logging.info(self.model)

    def Train_Step(self, audios, mels, pitches, audio_Singers, mel_Singers, noises):
        loss_Dict = {}

        audios = audios.to(device)
        mels = mels.to(device)
        pitches = pitches.to(device)
        audio_Singers = audio_Singers.to(device)
        mel_Singers = mel_Singers.to(device)
        noises = noises.to(device)

        # Generator
        fakes, classified_Singers, classified_Pitches, fakes_Discriminations, reals_Discriminations = self.model(
            mels= mels,
            pitches= pitches,
            singers= audio_Singers,
            noises= noises,
            discrimination= self.steps >= hp_Dict['Train']['Discriminator_Delay'],
            reals= audios
            )

        loss_Dict['Generator/Spectral_Convergence'], loss_Dict['Generator/Magnitude'] = self.criterion_Dict['STFT'](fakes, audios)
        loss_Dict['Generator'] = loss_Dict['Generator/Spectral_Convergence'] + loss_Dict['Generator/Magnitude']
        
        loss_Dict['Confuser/Singer'] = self.criterion_Dict['CE'](classified_Singers, mel_Singers)
        loss_Dict['Confuser/Pitch'] = self.criterion_Dict['MAE'](classified_Pitches, pitches)
        loss_Dict['Confuser'] = loss_Dict['Confuser/Singer'] + loss_Dict['Confuser/Pitch']

        if self.steps >= hp_Dict['Train']['Discriminator_Delay']:
            loss_Dict['Discriminator/Fake'] = torch.mean(fakes_Discriminations)   # As the discriminator, negative is correct.
            loss_Dict['Discriminator/Real'] = -torch.mean(reals_Discriminations)  # As the discriminator, positive is correct.
            loss_Dict['Discriminator'] = loss_Dict['Discriminator/Fake'] + loss_Dict['Discriminator/Real']
                
        self.optimizer.zero_grad()
        loss = loss_Dict['Generator'] + loss_Dict['Confuser'] + loss_Dict['Discriminator']

        if hp_Dict['Use_Mixed_Precision']:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters= amp.master_params(self.optimizer),
                max_norm= hp_Dict['Train']['Gradient_Norm']
                )
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model.parameters(),
                max_norm= hp_Dict['Train']['Gradient_Norm']
                )

        self.optimizer.step()
        self.scheduler.step()

        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_Dict.items():
            self.scalar_Dict['Train']['Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        if any([
            hp_Dict['Train']['Mixup']['Use'] and self.steps >= hp_Dict['Train']['Mixup']['Apply_Delay'],
            hp_Dict['Train']['Back_Translate']['Use'] and self.steps >= hp_Dict['Train']['Back_Translate']['Apply_Delay']
            ]):
            self.Data_Accumulation()

        for step, (audios, mels, pitches, audio_Singers, mel_Singers, noises) in enumerate(self.dataLoader_Dict['Train'], 1):
            self.Train_Step(audios, mels, pitches, audio_Singers, mel_Singers, noises)
            
            if self.steps % hp_Dict['Train']['Checkpoint_Save_Interval'] == 0:
                self.Save_Checkpoint()

            if self.steps % hp_Dict['Train']['Logging_Interval'] == 0:
                self.scalar_Dict['Train'] = {
                    tag: loss / hp_Dict['Train']['Logging_Interval']
                    for tag, loss in self.scalar_Dict['Train'].items()
                        }
                self.scalar_Dict['Train']['Learning_Rate'] = self.scheduler.get_last_lr()
                self.Write_to_Tensorboard('Train', self.scalar_Dict['Train'])
                self.scalar_Dict['Train'] = defaultdict(float)

            if self.steps % hp_Dict['Train']['Evaluation_Interval'] == 0:
                self.Evaluation_Epoch()
            
            if self.steps >= hp_Dict['Train']['Max_Step']:
                return

        self.epochs += hp_Dict['Train']['Accumulation_Inverval']

    
    @torch.no_grad()
    def Evaluation_Step(self, audios, mels, pitches, audio_Singers, mel_Singers, noises):
        loss_Dict = {}

        audios = audios.to(device, non_blocking=True)
        mels = mels.to(device, non_blocking=True)
        pitches = pitches.to(device, non_blocking=True)
        audio_Singers = audio_Singers.to(device, non_blocking=True)
        mel_Singers = mel_Singers.to(device, non_blocking=True)
        noises = noises.to(device, non_blocking=True)

        # Generator
        fakes, classified_Singers, classified_Pitches, fakes_Discriminations, reals_Discriminations = self.model(
            mels= mels,
            pitches= pitches,
            singers= audio_Singers,
            noises= noises,
            discrimination= self.steps >= hp_Dict['Train']['Discriminator_Delay'],
            reals= audios
            )

        loss_Dict['Generator/Spectral_Convergence'], loss_Dict['Generator/Magnitude'] = self.criterion_Dict['STFT'](fakes, audios)
        loss_Dict['Generator'] = loss_Dict['Generator/Spectral_Convergence'] + loss_Dict['Generator/Magnitude']
        
        loss_Dict['Confuser/Singer'] = self.criterion_Dict['CE'](classified_Singers, mel_Singers)
        loss_Dict['Confuser/Pitch'] = self.criterion_Dict['MAE'](classified_Pitches, pitches)
        loss_Dict['Confuser'] = loss_Dict['Confuser/Singer'] + loss_Dict['Confuser/Pitch']

        if self.steps >= hp_Dict['Train']['Discriminator_Delay']:
            loss_Dict['Discriminator/Fake'] = torch.mean(fakes_Discriminations)   # As the discriminator, negative is correct.
            loss_Dict['Discriminator/Real'] = -torch.mean(reals_Discriminations)  # As the discriminator, positive is correct.
            loss_Dict['Discriminator'] = loss_Dict['Discriminator/Fake'] + loss_Dict['Discriminator/Real']

        for tag, loss in loss_Dict.items():
            self.scalar_Dict['Evaluation']['Loss/{}'.format(tag)] += loss

    @torch.no_grad()
    def Inference_Step(self, audios, mels, pitches, singers, noises, start_Index= 0):
        audios = audios.to(device, non_blocking=True)
        mels = mels.to(device, non_blocking=True)
        pitches = pitches.to(device, non_blocking=True)
        singers = singers.to(device, non_blocking=True)
        noises = noises.to(device, non_blocking=True)

        fakes, *_ = self.model(
            mels= mels,
            pitches= pitches,
            singers= singers,
            noises= noises
            )

        os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps)).replace("\\", "/"), exist_ok= True)
        for index, (real, fake) in enumerate(zip(
            audios.cpu().numpy(),
            fakes.cpu().numpy()
            )):
            file = os.path.join(
                hp_Dict['Inference_Path'],
                'Step-{}'.format(self.steps),
                'Step-{}.IDX_{}'.format(self.steps, index + start_Index)
                ).replace("\\", "/")


            new_Figure = plt.figure(figsize=(80, 10 * 2), dpi=100)
            plt.subplot(211)
            plt.plot(real)
            plt.title('Original wav    Index: {}'.format(index))
            plt.subplot(212)            
            plt.plot(fake)
            plt.title('Fake wav    Index: {}'.format(index))
            plt.tight_layout()
            plt.savefig('{}.png'.format(file))
            plt.close(new_Figure)

            wavfile.write(
                filename= '{}.wav'.format(file),
                data= (np.clip(fake, -1.0 + 1e-7, 1.0 - 1e-7) * 32767.5).astype(np.int16),
                rate= hp_Dict['Sound']['Sample_Rate']
                )

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation.'.format(self.steps))

        self.model.eval()

        for step, (audios, mels, pitches, original_Singers, inference_Singers, noises) in tqdm(
            enumerate(self.dataLoader_Dict['Dev'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataLoader_Dict['Dev'].dataset) / hp_Dict['Train']['Batch_Size'])
            ):
            self.Evaluation_Step(audios, mels, pitches, original_Singers, original_Singers, noises)
            self.Inference_Step(audios, mels, pitches, inference_Singers, noises, start_Index= (step - 1) * hp_Dict['Train']['Batch_Size'])

        self.scalar_Dict['Evaluation'] = {
            tag: loss / step
            for tag, loss in self.scalar_Dict['Evaluation'].items()
            }
        self.Write_to_Tensorboard('Evaluation', self.scalar_Dict['Evaluation'])
        self.scalar_Dict['Evaluation'] = defaultdict(float)

        
        self.model.train()


    @torch.no_grad()
    def Back_Translate_Step(self, mels, pitches, singers, noises):
        mels = mels.to(device, non_blocking=True)
        pitches = pitches.to(device, non_blocking=True)
        singers = singers.to(device, non_blocking=True)
        noises = noises.to(device, non_blocking=True)

        fakes, *_ = self.model(
            mels= mels,
            pitches= pitches,
            singers= singers,
            noises= noises
            )
        
        return fakes.cpu().numpy()

    def Data_Accumulation(self):
        def Mixup(audio, pitch):
            max_Offset = pitch.shape[0] - hp_Dict['Train']['Wav_Length'] // hp_Dict['Sound']['Frame_Shift'] * 2
            offset1 = np.random.randint(0, max_Offset)
            offset2 = np.random.randint(0, max_Offset)
            beta = np.random.uniform(
                low= hp_Dict['Train']['Mixup']['Min_Beta'],
                high= hp_Dict['Train']['Mixup']['Max_Beta'],
                )

            new_Audio = \
                beta * audio[offset1 * hp_Dict['Sound']['Frame_Shift']:offset1 * hp_Dict['Sound']['Frame_Shift'] + hp_Dict['Train']['Wav_Length'] * 2] + \
                (1 - beta) * audio[offset2* hp_Dict['Sound']['Frame_Shift']:offset2 * hp_Dict['Sound']['Frame_Shift'] + hp_Dict['Train']['Wav_Length'] * 2]

            new_Pitch = \
                beta * pitch[offset1:offset1 + hp_Dict['Train']['Wav_Length'] // hp_Dict['Sound']['Frame_Shift'] * 2] + \
                (1 - beta) * pitch[offset2:offset2 + hp_Dict['Train']['Wav_Length'] // hp_Dict['Sound']['Frame_Shift'] * 2]
            
            _, new_Mel, _, _ = Pattern_Generate(audio= new_Audio)

            return new_Audio, new_Mel, new_Pitch

        def Back_Translate(mels, pitches, singers, noises):
            fakes = self.Back_Translate_Step(
                mels= mels,
                pitches= pitches,
                singers= singers,
                noises= noises
                )

            new_Mels = [
                Pattern_Generate(audio= fake)[1]
                for fake in fakes
                ]

            return new_Mels

        print()
        mixup_List = []
        back_Translate_List = []
        for total_Audios, total_Pitches, audios, mels, pitches, singers, noises in tqdm(self.dataLoader_Dict['Accumulation'], desc='[Accumulation]'):
            #Mixup
            if hp_Dict['Train']['Mixup']['Use'] and self.steps >= hp_Dict['Train']['Mixup']['Apply_Delay']:
                for audio, pitch, singer in zip(total_Audios, total_Pitches, singers.numpy()):
                    mixup_Audio, mixup_Mel, mixup_Pitch = Mixup(audio, pitch)
                    mixup_List.append((
                        mixup_Audio,
                        mixup_Mel,
                        mixup_Pitch,
                        singer,
                        singer
                        ))
            
            #Backtranslate
            if hp_Dict['Train']['Back_Translate']['Use'] and self.steps >= hp_Dict['Train']['Back_Translate']['Apply_Delay']:
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

        self.dataLoader_Dict['Train'].dataset.Accumulation_Renew(
            mixup_Pattern_List= mixup_List,
            back_Translate_Pattern_List= back_Translate_List
            )


    def Load_Checkpoint(self):
        if self.steps == 0:
            paths = [
                os.path.join(root, file).replace('\\', '/')                
                for root, _, files in os.walk(hp_Dict['Checkpoint_Path'])
                for file in files
                if os.path.splitext(file)[1] == '.pt'
                ]
            if len(paths) > 0:
                path = max(paths, key = os.path.getctime)
            else:
                return  # Initial training
        else:
            path = os.path.join(hp_Dict['Checkpoint_Path'], 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        state_Dict = torch.load(path, map_location= 'cpu')
        self.model.load_state_dict(state_Dict['Model'])
        self.optimizer.load_state_dict(state_Dict['Optimizer'])
        self.scheduler.load_state_dict(state_Dict['Scheduler'])
        self.steps = state_Dict['Steps']
        self.epochs = state_Dict['Epochs']

        if hp_Dict['Use_Mixed_Precision']:
            if not 'AMP' in state_Dict.keys():
                logging.info('No AMP state dict is in the checkpoint. Model regards this checkpoint is trained without mixed precision.')
            else:                
                amp.load_state_dict(state_Dict['AMP'])

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    def Save_Checkpoint(self):
        os.makedirs(hp_Dict['Checkpoint_Path'], exist_ok= True)

        state_Dict = {
            'Model' : self.model.state_dict(),
            'Optimizer': self.optimizer.state_dict(),
            'Scheduler': self.scheduler.state_dict(),
            'Steps': self.steps,
            'Epochs': self.epochs,
            }
        if hp_Dict['Use_Mixed_Precision']:
            state_Dict['AMP'] = amp.state_dict()

        torch.save(
            state_Dict,
            os.path.join(hp_Dict['Checkpoint_Path'], 'S_{}.pt'.format(self.steps).replace('\\', '/'))
            )

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))

    def Train(self):
        if not os.path.exists(os.path.join(hp_Dict['Checkpoint_Path'], 'Hyper_Parameter.yaml')):
            os.makedirs(hp_Dict['Checkpoint_Path'], exist_ok= True)
            with open(os.path.join(hp_Dict['Checkpoint_Path'], 'Hyper_Parameters.yaml').replace("\\", "/"), "w") as f:
                yaml.dump(hp_Dict, f)

        if hp_Dict['Train']['Initial_Inference']:
            self.Evaluation_Epoch()

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

    def Write_to_Tensorboard(self, category, scalar_Dict):
        for tag, scalar in scalar_Dict.items():
            self.writer_Dict[category].add_scalar(tag, scalar, self.steps)

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', '--steps', default= 0, type= int)
    args = argParser.parse_args()
    
    new_Trainer = Trainer(steps= args.steps)    
    new_Trainer.Train()