# PyTorch implementation of SPATE-GAN

![Generated data from the turbulent flows dataset](https://raw.githubusercontent.com/konstantinklemmer/spate-gan/main/images/tf.png)

This is the official repository for the paper [SPATE-GAN: Improved Generative Modeling of Dynamic Spatio-Temporal Patterns with an Autoregressive Embedding Loss](https://arxiv.org/abs/2109.15044/) (Konstantin Klemmer\*, Tianlin Xu\*, Beatrice Acciaio, Daniel B. Neill).

\* These authors contributed equally.

## Structure

The source code for *SPATE-GAN* can be found in the `src` folder. It builds on the code base for *[COT-GAN](https://papers.nips.cc/paper/2020/file/641d77dd5271fca28764612a028d9c8e-Paper.pdf)* (NeurIPS 2020), accessible here: \[[Tensorflow](https://github.com/tianlinxu312/cot-gan),[PyTorch](https://github.com/tianlinxu312/cot-gan-pytorch)\]

We also provide an interactive example notebook to test *SPATE-GAN* via Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konstantinklemmer/spate-gan/blob/master/spate_gan_example.ipynb)

## Citation 

If you want to cite our work, you can use the following reference:

```
@misc{klemmer2021spategan,
	    title={SPATE-GAN: Improved Generative Modeling of Dynamic Spatio-Temporal Patterns with an Autoregressive Embedding Loss},
	    author={Konstantin Klemmer and Tianlin Xu and Beatrice Acciaio and Daniel B. Neill},
	    year={2021},
	    eprint={2109.15044},
	    archivePrefix={arXiv},
	    primaryClass={cs.LG}
}
```
