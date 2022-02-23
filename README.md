# Noise2Noise:Learning Image Restoration without Clean Data
Unofficial code, using pyrch to implement n2n\
*Hai Zhou et al. 2022.2*

**Abstract**:

_We apply basic statistical reasoning to signal reconstruction by machine learning -- learning to map corrupted observations to clean signals -- with a simple and powerful conclusion: it is possible to learn to restore images by only looking at corrupted examples, at performance at and sometimes exceeding training using clean data, without explicit image priors or likelihood models of the corruption. In practice, we show that a single model learns photographic noise removal, denoising synthetic Monte Carlo images, and reconstruction of undersampled MRI scans -- all corrupted by different processes -- based on noisy data only._

**Resources**

* [Paper (arXiv)](https://arxiv.org/abs/1803.04189)

**Getting started**

The following sections detail how to use n2n for training and 
validation on various datasets.

**Python requires**

This code requires python3.9 to run. Below I will show how to use 
anaconda4.11 to create a python environment and install various 
resource libraries.
```
conda create -n n2n python=3.9
conda activate n2n
conda install tensorflow-gpu
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Data requires**

We use the dataset link below
* [COCO](http://cocodataset.org/#download)
* [BSD300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz)
* [Kodak](http://r0k.us/graphics/kodak/)

Move the contents of the downloaded data folder 
to the folder corresponding to the `./dataset/data/`

**Dataset**

This time, the training data set is BSD300, and the validation 
set is kodak. In order to use the data set more conveniently, 
we need to perform some operations on the data set before training, 
and extract the data of all pictures into a file, so that in the 
Can save a lot of disk operation time when training.

* Please use the following operations to extract the dataset(an image is randomly cropped twice)

  `python3 dataset/utils.py --extract --num_img 2`



**Train**

Suppose we need to use the BSD300 dataset for training and 
the Kodak dataset for verification. At this time, run the 
command as follows:

  `python3 run.py --dataset BSD300 --varitydata kodak`
  
Validation is performed every 30 epochs by default during training.

**Varity**

