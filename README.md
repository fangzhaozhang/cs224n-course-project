# dialogDiffAE

## Introduction
 
Goal: In this project, we seek to answer the question: whether diffusion model, acted as a distribution learner, can help capture the complex distribution of latent space in word embeddings? If so, how does it perform compared to VAE and GAN? The motivation of our goal is due to limitation of VAE and GAN in capturing latent variable distribution of word embeedings. Recent development in diffusion based generative models serve as a remedy. With no specific assumption on latent variable distribution, nor unstable adversarial training procedure being involved, we believe using diffusion in word embedding latent variable modeling is both more accurate and more stable.

## Installation

* Python 
* PyTorch
* Numpy
* sklearn
* NLTK

```
pip install -r requirements.txt
```

## Model
<img src="network.png"
     alt="network structure"
     style="float: left; margin-right: 10px;" />
     
## Algorithm 

Training: train AE phase I + train diffuser + train AE phase II

## Implementation

XXX class: XXX


## Getting started

```
python main.py
```

## Dataset

## Experiment Results

## References

* AE-related code adapted from [DialogWAE: Multimodal Response Generation with Conditional Wasserstein Auto-Encoder](https://arxiv.org/abs/1805.12352)

* Diffusion-related code adapted from [tiny-diffusion](https://github.com/tanelp/tiny-diffusion)





