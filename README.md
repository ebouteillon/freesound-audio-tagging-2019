# Kaggle Freesound Audio Tagging 2019 Competition Solution

This is the solution I proposed for [Kaggle Freesound Audio Tagging 2019 Competition](https://www.kaggle.com/c/freesound-audio-tagging-2019/overview).

## Presentation

This repository presents a semi-supervised **warm-up pipeline** used to create an efficient audio tagging system as well as a novel data augmentation technique for multi-labels audio tagging named by the author **SpecMix**.

 These new techniques were applied to our submitted audio tagging system to the _Freesound Audio Tagging 2019_ challenge carried out within the _DCASE 2019 Task 2 challenge_ [3]. Purpose of this challenge consist of predicting the audio labels for every test clips using machine learning techniques trained on a small amount of reliable, manually-labeled data, and a larger quantity of noisy web audio data in a multi-label audio tagging task with a large vocabulary setting.

## TL;DR - give me code!

Provided Jupyter notebooks result in a [lwlrap](https://www.kaggle.com/c/freesound-audio-tagging-2019/overview/evaluation) of .738 in public leaderboard, that is to say 12th position in this competition.

- [Training CNN model 1](code/training-cnn-model1.ipynb)
- [Training VGG16 model](code/training-vgg16.ipynb)
- [Inference kernel](code/inference-kernel.ipynb)

You can also find resulting weights of CNN-model-1 and VGG-16 training in [weights directory](weights). Note [git-lfs](https://git-lfs.github.com/) might be required to download them using git.

## Audio Data Preprocessing

Audio clips were first trimmed of leading and trailing silence (threshold of 60 dB), then converted into 128-bands mel-spectrogram using a 44.1 kHz sampling rate, hop length of 347 samples between successive frames, 2560 FFT components and frequencies kept in range 20 Hz – 22,050 Hz. Last preprocessing consisted in normalizing (mean=0, variance=1) the resulting images and duplicating to 3 channels.

## Models Summary

In this section, we describe the neural network architectures used:

**Version 1** consists in  an ensemble of a custom CNN &quot;CNN-model-1&quot; defined in Table 1 and a VGG-16 with batch-normalization. Both are trained in the same manner.

**Version 2** consist of only our custom CNN &quot;CNN-model-1&quot;, defined in Table 1.

**Version 3** is evaluated for Judge award and it is same model as version 2.

| Input 128 × 128 × 3 |
| --- |
| 3 × 3 Conv(stride=1, pad=1)−64−BN−ReLU |
| 3 × 3 Conv(stride=1, pad=1)−64−BN−ReLU |
| 3 × 3 Conv(stride=1, pad=1)−128−BN−ReLU |
| 3 × 3 Conv(stride=1, pad=1)−128−BN−ReLU |
| 3 × 3 Conv(stride=1, pad=1)−256−BN−ReLU |
| 3 × 3 Conv(stride=1, pad=1)−256−BN−ReLU |
| 3 × 3 Conv(stride=1, pad=1)−512−BN−ReLU |
| 3 × 3 Conv(stride=1, pad=1)−512−BN−ReLU |
| concat(AdaptiveAvgPool2d + AdaptiveMaxPool2d) |
| Flatten−1024-BN-Dropout 25% |
| Dense-512-Relu-BN-Dropout 50% |
| Dense-80 |

Table 1: CNN-model-1. BN: Batch Normalisation, ReLU: Rectified Linear Unit,

## Data Augmentation

One important technique to leverage a small training set is to augment this set using data augmentation. For this purpose we created a new augmentation named **SpecMix**. This new augmentation is an extension of _SpecAugment_ [1] inspired by _mixup_ [2].

**SpecAugment** applies 3 transformations to augment a training sample: time warping, frequency masking and  time masking on mel-spectrograms.

**mixup** creates a virtual training example by computing a weighted average of two samples inputs and targets.

### SpecMix

**SpecMix** is inspired from the two most effective transformations from  _SpecAugment_ and extends them to create virtual multi-labels training examples:

1. _Frequency replacement_ is applied so that _f_ consecutive mel-frequency channels _[f0, f0+f)_ are replaced from another training sample, where _f_ is first chosen  from  a  uniform  distribution  from  minimal  to maximum the frequency  mask  parameter _F_,  and _f0_ is  chosen  from _[0, ν−f)_. _ν_ is the number of mel frequency channels.
2. _Time replacement_ is applied so that _t_ consecutive time steps _[t0, t0+t)_ are replaced from another training sample,  where _t_ is first chosen from a uniform distribution from 0 to the time mask parameter _T_, and _t0_ is chosen from _[0, τ−t)_.  _τ_ is the number of time samples.
3. _Target_ of the new training sample is computed as the weighted average of each original samples. The weight for each original sample is proportional to the number of pixel from that sample. Our implementation uses same replacement sample for _Frequency replacement_ and _Time replacement_, so it gives us a new target computed based on:

                (1)

        where

### Others data augmentation

We added other data augmentation techniques:

- **mixup** before SpecMix. A small improvement is observed (lwlrap increased by +0.001). mixup is first applied on current batch, generating new samples for the current batch and then SpecMix is applied on these newly created samples. In the end, combining mixup and SpecMix, up to four samples are involved in the generation of one single sample.
- **zoom and crop**  small improvement too (lwlrap increased by +0.001)

## Training

At training time, we give to the network batches of 128  augmented excerpts of randomly selected sample mel-spectrograms. We use a 10-fold cross validation setup.

  Training is done in 4 stages, each stage generating a model which is used for 3 things:

- **warm-up the model** training in the next stage
- help in a **semi-supervised selection** of noisy elements
- participate in the **test prediction** (except model 1)

An important point of this competition, is that we are not allowed to use external data nor pretrained models. So our pipeline presented below only used curated and noisy sets from the competition:

- **Stage 1** : Train a model (model1) from scratch only using the noisy set. Then compute cross-validated lwlrap on noisy set (lwlrap1).
- **Stage 2** : Train a model (model2) only on curated set but use model1 as pretrained model. Then compute cross-validated lwlrap on noisy set (lwlrap2).
- **Stage 3** : Let&#39;s start semi-supervised learning: our algorithm select samples from noisy set that are (almost) correctly classified by both model1 and model2. This algorithm simply keep sample from noisy set getting a geometric mean of (lwlrap1, lwlrap2) higher or equal to 0,5. A maximum of 5 samples per fold and per label is selected. Then train a model (model3) on curated plus selected noisy samples and use model2 as pretrained model. Then compute cross-validated lwlrap on noisy set (lwlrap3).
- **Stage 4** : Let&#39;s continue semi-supervised learning: our algorithm select again samples from noisy set that are strictly correctly classified by model3. This algorithm simply keep sample from noisy set getting a lwlrap3 equal to 1. Then train a model (model4) on curated plus selected noisy samples and use model3 as pretrained model.
- **Last stage:** ensemble predictions on test set from model2, model3 and model4.

Figure 1: warm-up pipeline
![model-explained](images/model-explained.png)

| **Model** | **lwlrap noisy** | **lwlrap curated** | **leaderboard** |
| --- | --- | --- | --- |
| model1 | 0.65057 | 0.41096 | N/A |
| model2 | 0.38142 | 0.86222 | 0.723 |
| model3 | 0.56716 | 0.87930 | 0.724 |
| model4 | 0.57590 | 0.87718 | 0.724 |
| ensemble | N/A | N/A | 0.733 |

Table 2: Empirical results of CNN-model-1 using proposed warm-up pipeline

## Hardware / Software

- Intel Core i7 4790k
- Nvidia RTX 2080 ti
- 24 GB RAM
- Ubuntu 18.04.2
- Detailed list of installed [python packages](conda_list.txt) (more than necessary)

## REFERENCES

[1] Daniel S. Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D. Cubuk, Quoc V. Le, &quot;SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition&quot;, [arXiv:1904.08779](https://arxiv.org/abs/1904.08779), 2019.

[2] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz.  &quot;_mixup: Beyondempirical risk minimization_&quot;. arXiv preprint arXiv:1710.09412, 2017.

[3] Eduardo Fonseca, Manoj Plakal, Frederic Font, Daniel P. W. Ellis, and Xavier Serra. &quot;Audio tagging with noisy labels and minimal supervision&quot;. Submitted to DCASE2019 Workshop, 2019. URL: [https://arxiv.org/abs/1906.02975](https://arxiv.org/abs/1906.02975)
