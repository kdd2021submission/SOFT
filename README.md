# SOFT

This repository contains the python implementation for paper "Second-Order Unsupervised Feature Selection via Knowledge Contrastive Distillation".

## Paper Abstract

Unsupervised feature selection aims to select a subset from the original features that are most useful for the downstream tasks without external guidance information. While most unsupervised feature selection methods focus on ranking features based on the intrinsic properties of data, they do not pay much attention to the relationships between features, which often leads to redundancy among the selected features. In this paper, we propose a two-stage Second-Order unsupervised Feature selection via knowledge contrastive disTillation (SOFT) model that incorporates the second-order covariance matrix with the first-order data matrix for unsupervised feature selection. In the first stage, we learn a sparse attention matrix that can represent second-order relations between features. In the second stage, we build a relational graph based on the learned attention matrix and adopt graph segmentation for feature selection. Experimental results on 12 public datasets demonstrate the effectiveness of our proposed method.

## Requirements

* faiss
* h5py
* networkx
* numpy
* nxmetis
* pandas
* pytorch
* skfeature

## File Description

* `clustering.py`: clustering method to get pseudo labels
* `evaluate_first_order.py`: the second stage of the first_order method to select features
* `evaluate.py`: the second stage of SOFT to select features
* `model.py`: pytorch implementation of SOFT model and first_order model
* `selector_first_order.py`: the first stage of the first_order method to learn feature weights
* `selector.py`: the first stage of SOFT to learn second-order feature relations
* `util.py`: support functions

## Datasets

* [COIL20](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php)
* [Colon](https://www.openml.org/d/1432)
* [Lung-Cancer](https://archive.ics.uci.edu/ml/datasets/Lung+Cancer)
* [Madelon](http://clopinet.com/isabelle/Projects/NIPS2003/)
* [MovementLibras](https://archive.ics.uci.edu/ml/datasets/Libras+Movement)
* [NCI9](https://jundongl.github.io/scikit-feature/datasets.html)
* [ORL](http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html)
* [Sonar](https://www.openml.org/d/40)
* [UAV1 and UAV2](https://archive.ics.uci.edu/ml/datasets/Unmanned+Aerial+Vehicle+%28UAV%29+Intrusion+Detection#)
* [UJIIndoorLoc](https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc)
* [Waveform](https://www.openml.org/d/60)

## Methods for Comparison

* [Laplacian Score](http://www.cad.zju.edu.cn/home/dengcai/Publication/Conference/2005_NIPS_LaplacianScore.pdf)
* [SPEC](https://dl.acm.org/doi/abs/10.1145/1273496.1273641?casa_token=pAvDhm9_rCYAAAAA:Tg__sn3p15B6Fmc7oHfkXArYVyBPUw-i_b32NKNY8ma8JyPySXeTvKXreUGVKF3vtp9SJW0-rgeKCio)
* [MCFS](https://dl.acm.org/doi/abs/10.1145/1835804.1835848?casa_token=iiLbJVdNt30AAAAA:fd9bA9filcl25TuTwrvrICYXST3zMVTEzeOyRU_AZjUvif4bC7UypE6U7BI5YyrqEjg6M6RwWfDAGVA)
* [UDFS](https://opus.lib.uts.edu.au/handle/10453/119490)
* [NDFS](https://ojs.aaai.org/index.php/AAAI/article/view/8289)
* [LRPFS](https://www.sciencedirect.com/science/article/pii/S092523121830746X?casa_token=xQHjny52GugAAAAA:jRfjiTf9lq3iJVwTpxKhatY4mXovZmfaFr0vJYrufDAKXi1lTEVoPWdfCD3P5-vmCCCtJXPewxy4)
* [NSSLFS](https://www.sciencedirect.com/science/article/pii/S092523121930027X?casa_token=E2J3Yx3bexMAAAAA:oOyvpQlSRKGz_Ozwo6f_mUtQx_w9BNFwTCkGoKJXdLeMw4_aRF4M_QiwznQKReCTes7QXHEP0Xkm)
* [TSFS](https://www.sciencedirect.com/science/article/pii/S0925231219317199?casa_token=igdE-RDgDMgAAAAA:Lkino7Y6c8Dv8gg7NL5vUl7_uYD0XxN5qokyZVfsZyULjdgcuh-G83jclwFcECe-_uQJM6_6-cyC)

## Parameter Settings

In the first stage by `selector.py`, hyperparameters are set as follows:
* `alpha` = 1.0
* `beta` = 0.001
* `LEARNING_RATE` = 0.0001
* `EPOCHS` = 300

In the second stage by `evaluate.py`, hyperparameters are set as follows:
* `SELECT_PER_CLUSTER` = 1
* `OUTLIER_PERCENT` = 0.1
* `SET_ZERO_PERCENT` = 0.5

## How to Run

In the first stage, SOFT learns the second-order feature relation matrix by running:
```bash
python selector.py
```

In the second stage, SOFT selects features and is evaluated by running:
```bash
python evaluate.py
```