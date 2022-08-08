# PyTorch implementation of SPATE-GAN

![Real and generated data from the turbulent flows dataset](https://raw.githubusercontent.com/konstantinklemmer/spate-gan/main/images/tf.png)

*(Real and generated data from the turbulent flows dataset)*

This is the official repository for the *AAAI 2022* paper [SPATE-GAN: Improved Generative Modeling of Dynamic Spatio-Temporal Patterns with an Autoregressive Embedding Loss](https://arxiv.org/abs/2109.15044/) (Konstantin Klemmer\*, Tianlin Xu\*, Beatrice Acciaio, Daniel B. Neill).

\* These authors contributed equally.

## Structure

The source code for *SPATE-GAN* (using `PyTorch`) can be found in the `src` folder. It builds on the code base for *[COT-GAN](https://papers.nips.cc/paper/2020/file/641d77dd5271fca28764612a028d9c8e-Paper.pdf)* (NeurIPS 2020), accessible here: \[[Tensorflow](https://github.com/tianlinxu312/cot-gan),[PyTorch](https://github.com/tianlinxu312/cot-gan-pytorch)\]

We also provide an interactive example notebook to test *SPATE-GAN* via Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konstantinklemmer/spate-gan/blob/master/spate_gan_example.ipynb)

## SPATE - SPAtio-TEmporal Association

![The different approaches for obtaining the spatio-temporal expectations needed to compute SPATE](https://raw.githubusercontent.com/konstantinklemmer/spate-gan/main/images/stx.png)

*(The different approaches for obtaining the spatio-temporal expectations needed to compute SPATE)*

Contained within the `src` folder, the `spatial_utils.py` file contains all needed functions to compute the *SPATE* embedding in its different configurations.

Beyond our new *SPATE* metric, `spatial_utils.py` also includes the (to our knowledge) first `PyTorch` implementation of the original local Moran's I metric, along with the capacity to compute it for batches of spatial patterns / images.  

![Differences between Moran's I and SPATE in its different configurations](https://raw.githubusercontent.com/konstantinklemmer/spate-gan/main/images/emb.png)

*(Differences between Moran's I and SPATE in its different configurations)*
	
## Citation 

If you want to cite our work, you can use the following reference:

```
@article{klemmer2022spategan, 
		title={SPATE-GAN: Improved Generative Modeling of Dynamic Spatio-Temporal Patterns with an Autoregressive Embedding Loss}, 
		volume={36}, 
		url={https://ojs.aaai.org/index.php/AAAI/article/view/20375}, 
		DOI={10.1609/aaai.v36i4.20375}, 
		number={4}, 
		journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
		author={Klemmer, Konstantin and Xu, Tianlin and Acciaio, Beatrice and Neill, Daniel B.}, 
		year={2022}, 
		month={Jun.},
		pages={4523-4531}
}
```
