## Speech Emotion Recognition Problem

For Speech Emotion Recognition one can use either of these: raw wave form (1d), logmelspectrum (2d), logspecgram (2d), mfcc (2d), filterbank (2d) or mixture of these.

### Useful Definitions:
<b>Specgram</b>-Time-dependent frequency analysis (spectrogram).Specgram computes the windowed discrete-time Fourier transform of a signal using a sliding window. The spectrogram is the magnitude of this function.

<b>Mel-Spectrogram</b>- Mel-scaled spectrogram.

<b>Log Mel-Spectrogram</b>- Log of Mel-scaled spectrogram.

<b>MFCC</b>-Mel Frequency Cepstral Coefficents (MFCCs) are a feature widely used in automatic speech and speaker recognition. They were introduced by Davis and Mermelstein in the 1980's, and have been state-of-the-art ever since.To get MFCC, compute the DCT on the mel-spectrogram. The mel-spectrogram is often log-scaled before.MFCC is a very compressible representation, often using just 20 or 13 coefficients instead of 32-64 bands in Mel spectrogram. The MFCC is a bit more decorrelarated, which can be beneficial with linear models like Gaussian Mixture Models.

<b>Filter Banks</b> -Filter banks are arrangements of low pass, bandpass, and highpass filters used for the spectral decomposition and composition of signals. They play an important role in many modern signal processing applications such as audio and image coding. The reason for their popularity is the fact that they easily allow the extraction of spectral components of a signal while providing very efficient implementations. Since most filter banks involve various sampling rates, they are also referred to as multirate systems.

### My Approaches:
1. Training 1D and 2D CNN using Log-Specgram features of a wav file (Using Keras)
2. Training LSTM Network using Filter Bank features of a wav file (Using Keras)
3. Training 2D CNN after augentation of a wav file and taking logmelspectrum features. Augmentations include adding random noise, random padding and random time shift.(Using PyTorch)

<b>make_conda_env.sh</b> - Bash Script for making conda environment and installing all dependancies.
