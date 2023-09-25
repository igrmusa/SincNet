# SincNet

This project is part of my Master Studies Scientific Computation course. The code is writen using Python. <br>
The project is done in collaboration with Milos Nikolic. <br><br><br> The main idea of this project is to present SincNet which is a neural network for efficiently processing raw audio samples. SincNet is based on parametrized sinc functions, which implement band-pass filters. It is used on TIMIT database for speaker identification.

# Dataset <br><br>
SincNet is trained and evaluated on **DARPA Timit dataset**. This dataset is designed to provide speach data for acoustic-phonetic studies and for the development and evaluation of automatic speach recognition systems. TIMIT contains broadband recordings of 630 speakers of eight major dialects of American English, each reading ten phonetically rich sentences. This dataset also includes time-aligned orthographic, phonetic and word transcriptios as well as a 16-bit, 16kHz speach waveform file for each utterance. 
Code for preparing this dataset can be found in **TIMIT_preparation.ipynb** notebook where each waveform  we will frist remove silances according to the information in the wrd files and normalizes the amplitude of each sentences.
<br> This dataset can be found on the following link: **https://www.kaggle.com/datasets/mfekadu/darpa-timit-acousticphonetic-continuous-speech**. <br><br>

# SincNet model

The major difference between this model and previous attempts to use CNN in this problem, is that CNN is fed by raw speach samples directly. Previous attempts were based on employing hand-crafted features such as FBANK and MFCC coeficients. These enginered features are originally designed for perceptual evidence and there are no guarantees that such representations are optimal for all speach-related tasks. So with this difference, naturally first layer of CNN is most critical part. To help first layer to discover more meaningful filters, author of this model proposes to add some constraints on their shape. Compared to standard CNNs, where filter-bank characteristics depend on several parameters, the SincNet convolves the waveform with a set of parametrized sinc functions that implement band-pass filters. The low and high frequencies are the only parameters of the filter elarned from data. This solution forces the network to focus on high-lever tunable parameters. Results achived on varaiety of the datsets, show that proposed SincNet converges faster and achives better end task performance. <br>

## The SincNet architecture
The first layer of a standard CNN perform a set of time-domain convolutions between input waveform and some Finite Impulse Response filters. Each convolutions is defined as follows:
```math
y[n] = x[n]*h[n] = \sum_{l=0}^{L-1}x[l]\cdot h[n-l]
```
where **x[n]** is a chunk of speach signal, **h[n]** is the filter of length L and **y[n]** is the filtered output. In standard CNNs all L elements are learned from data. <br>
The proposed SincNet performs the convolution with a predefined function **g** that depends on few learnable parameters only
```math
y[n] = x[n]*g[n, \Omega]
```
This function can be written as difference between two low-pass filters:
```math
G[f, f_1, f_2] = rect(\frac{f}{2f_2}) - rect(\frac{f}{2f_1})
```
where $f_1$ and  $f_2$ are learned low and high cutoff frequenciesm and **rect** is rectangular function in the magnitude frequency domain. After returning to the domain with **IFFT** this function becomes 
```math
g[f, f_1, f_2] = 2f_2 sinc(2 \pi f_2 n) - 2f_1 sinc(2 \pi f_1 n))
```
The cut-frequencies can be initialized randomly in the range $[0, f_s /2]$ or alternatively filters can be initialized with cutoff frequencies of the **Mel-Scale** filter bank. <br>
An ideak bandpass filter requires an infinite number of elements L. Any trunctation of g thus leads to an aproximation of the ideal filter. Popular option to mitigate this is issue is windowing. This is performed by multiplying the trunced function g with a window function $\omega$, which aims to smooth out the abrupt discontinuties at the ends of g. This model use **Hamming windows**. Using other window function were tested by author and they didn't show any improvment of model. <br>

## Model properties

- **Fast Convergence**
- **Few Parameters**: SincNet drasticly reduces the number of parameters in the first convolutionall layer. For instance, if we consider a layer composed of F filters with length of L, a standard CNN will employ F*L parameters, agains 2*F considered by SuncNet. Moreover, if we double the filter length L, a standard CNN doubles its parameters, while SincNet has unchanged parameter count. This offers the possibility to derive very selective filters with many tapsm without adding parameters to the optimization problem.
- **Interpretability**: SincNet feature maps obtained in the first convolution layer are more interpretable and human-readable than other approaches.

### Training and evaluation
Training is done with **RMSprop optimizer**, with  $learning rade = 0.001$ and   $\alpha = 0.95$ and $epsilon=1e-7$ with minibatches of size 128. With training on 10 epoch we reached accuracy 0.7351 on training set and  
0.4326 on validation set. <br>
**Note:** Originaly this model have been trained for over 300epochs and it reachs better accuracy on both training and validations set. Due to hardware limitations, we were unable to train the model for more than 10 epochs.
