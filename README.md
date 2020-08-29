__※ This project has been discontinued. If you would like to suggest any improvements, please register as an issue.__

# Parallel Voice Conversion GAN

This code is an implementation of Parallel Voice Conversion GAN(PVCGAN). The algorithm is based on the following papers:

```
Oord, A. V. D., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., ... & Kavukcuoglu, K. (2016). Wavenet: A generative model for raw audio. arXiv preprint arXiv:1609.03499.
Yamamoto, R., Song, E., & Kim, J. M. (2020, May). Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 6199-6203). IEEE.
Deng, C., Yu, C., Lu, H., Weng, C., & Yu, D. (2020, May). Pitchnet: Unsupervised Singing Voice Conversion with Pitch Adversarial Network. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 7749-7753). IEEE.
Nachmani, E., & Wolf, L. (2019). Unsupervised singing voice conversion. arXiv preprint arXiv:1904.06590.
```

# Requirements
Please see the 'requirements.txt'.

# Structrue
![Structure](./Figures/PVCGAN.svg)

# Used dataset
Currently uploaded code is compatible with NUS-48E datasets: [NUS-48E](https://smcnus.comp.nus.edu.sg/nus-48e-sung-and-spoken-lyrics-corpus/)

# Hyper parameters
Before proceeding, please set the pattern, inference, and checkpoint paths in 'Hyper_Parameters.yaml' according to your environment.

* Sound
    * Setting basic sound parameters.

* Num_Singers
    * The number of singers.

* Encoder
    * Setting the encoder which is to generate a encoding pattern which the speaker and pitch information are removed.

* Singer_Classification
    * Setting singer classification network parameters.

* Pitch_Regression
    * Setting singer classification network parameters.

* WaveNet
    * Setting the parameters of generator based on WaveNet.
    * In upsample, the product of all of upsample scales must be same to frame shift size of sound.

* Discriminator
    * Setting the parameters of discriminator

* STFT_Loss_Resolution
    * Setting the parameters of multi resolution STFT loss.

* Train
    * Setting the parameters of training.    
    * Wav length must be a multiple of frame shift size of sound.
    * Shared_Train_and_Eval
        * This function is used because the pattern amount of music vocal dataset like NUS-48 is usually small.
        * When this value is true, the initial 'Wav_Length * 5' length of each train pattern is excluded in training.
        * At the same time, the initial 'Wav_Length * 5' length of each eval pattern is only used in evaluation.
    * Adversarial_Weight
        * These paratmers set the weight of reversed gradients by `GRL`.
        * The values are bigger than 1.0, `Generator` and `Encoder` have advantage.
        * If you want to change the dataset or parameters, I recommend to check the images and histogram in Tensorboard.
            * If the plots are not changed while several steps, model's adversarial balancing may be broken.
            * If so, please decrease the weight parameter to intensify the discriminator, classifier, or regressor.
    
* Use_Mixed_Precision
    * __Currently, this parameters is ignored.__ 
    * Several parameters including adversarial weights are affected by the bits of float.
       
* Inference_Path
    * Setting the inference path

* Checkpoint_Path
    * Setting the checkpoint path

* Log_Path
    * Setting the tensorboard log path

* Device
    * Setting which GPU device is used in multi-GPU enviornment.
    * Or, if using only CPU, please set '-1'. (But, I don't recommend while training.)

# Generate pattern

## Command
```
python Pattern_Generate.py [parameters]
```

## Parameters

At least, one or more of datasets must be used.

* -nus48e `<path>`
    * Set the path of NUS-48E.
* -sex `M|F|B`
    * `M`: Use only male singers
    * `F`: Use only female singers
    * `B`: Use all singers
    
# Inference file path while training for verification.

* Inference_for_Training.txt
    * Pickle file paths and inference singer id which are used to evaluate while training.

# Run

## Command
```
python Train.py -s <int>
```

* `-s <int>`
    * The resume step parameter.
    * Default is 0.

# Result
    * https://codejin.github.io/PVCGAN_Demo/index.html

# Trained checkpoint

* [Checkpoint](./Checkpoint/S_400000.pkl)
* [Hyper parameter](./Checkpoint/Hyper_Parameters.yaml)