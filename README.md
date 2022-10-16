
# VAE Segmentation

[Unsupervised Domain Adaptation through Shape Modeling for Medical Image Segmentation](https://arxiv.org/abs/2207.02529) \[MIDL2022]

Yuan Yao, Fengze Liu, Zongwei Zhou, Yan Wang, Wei Shen, Alan Yuille, Yongyi Lu



## Overview

Shape information is a strong and valuable prior in segmenting organs in medical images. In this project, we aim at modeling shape explicitly and using it to help medical image segmentation. 

 Variational Autoencoder (VAE) can learn the distribution of shape for a particular organ. Based on this, we propose a new unsupervised domain adaptation pipeline based on a pseudo loss and a VAE reconstruction loss under a teacher-student learning paradigm. Our method is validated on three public datasets (one for source domain and two for target domain) and two in-house datasets (target domain).

![visualize](figure/visualize.png)

The example above is a case of unsupervised domain adaptation from [NIH pancreas](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT) (source) to [MSD pancreas](http://medicaldecathlon.com/) (target). Our method outperforms other methods by fitting the predicted masks into a proper shape.



## Method

The overall pipeline of our VAE pipeline is demonstrated as follows.

![architecture](figure/architecture.png)



## Environment and Data Preparation

This codebase generally works for python>=3.8

Please install the dependencies: `pip install -r requirements.txt`

First, you can get access to the three public datasets used in our paper here:

- NIH Pancreas-CT Dataset: [https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT)

- Medical Segmentation Decathlon (MSD): [http://medicaldecathlon.com/](http://medicaldecathlon.com/)

- Synapse Dataset: [https://www.synapse.org/#!Synapse:syn3193805/wiki/217789](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)

Next, run `data/data_preprocess.py` to preprocess the image and generate `.npy` files. Please be careful to modify the path of images and labels. 

All the preprocessed data should be put in a json file with the same format as we have given in file `data/Multi_all.json`.



## Training and Evaluation

The training scripts can be found in the folder `scripts/`. 

First, run script `bash scripts/source/vae_nih.bash <GPU-number>` and `bash scripts/source/seg_nih.bash` to train the VAE network and the segmentation network on the source domain.

On the target domain, use command `bash scripts/target/<script_name> <GPU-number>` run scripts for different tasks.

|        Script name         |                           Function                           |
| :------------------------: | :----------------------------------------------------------: |
|   domain_msd_pseudo.bash   |        Domain adaptation without reconstruction loss.        |
|      domain_msd.bash       | Domain adaptation with our VAE pipeline. (without any techniques) |
|    domain_msd_ft1.bash     | Domain adaptation with our VAE pipeline. (with test-time training) |
|     domain_msd_dh.bash     | Domain adaptation with our VAE pipeline. (with dynamic hyperparameters) |
| **domain_msd_dh_ft1.bash** | Domain adaptation with our VAE pipeline. (with both techniques) |

To evaluate a trained model, remove the `--load_prefix` and `--load_prefix_vae` args in the script, but add `--load_prefix_joint <experiment_name>` and `--test_only`.



## Citation

If you use this code for your research, please cite our paper. 



## Acknoledgement

This work was supported by the Fundamental Research Funds for the Central Universities.

