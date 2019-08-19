DAN-Visdial
========================================================================

PyTorch implementation for the [Dual Attention Networks for Visual Reference Resolution in Visual Dialog][1]. <br>
For the visual dialog v1.0 dataset, our single model achieved **57.59** on NDCG and **49.63** on Recall@1.   

![Overview of Dual Attention Networks](dan_overview.jpg)

If you use this code in your published research, please consider citing:
```text
@article{kang2019dual,
  title={Dual Attention Networks for Visual Reference Resolution in Visual Dialog},
  author={Kang, Gi-Cheon and Lim, Jaeseo and Zhang, Byoung-Tak},
  journal={arXiv preprint arXiv:1902.09368},
  year={2019}
}
```

Setup and Dependencies
----------------------
This starter code is implemented using **PyTorch v0.3.1** with **CUDA 8 and CuDNN 7**. <br>
It is recommended to set up this source code using Anaconda or Miniconda. <br>

1. Install Anaconda or Miniconda distribution based on Python 3.6+ from their [downloads' site][2].
2. Clone this repository and create an environment:

```sh
git clone https://github.com/gicheonkang/DAN-VisDial
conda create -n dan_visdial python=3.6

# activate the environment and install all dependencies
conda activate dan_visdial
cd DAN-VisDial/
pip install -r requirements.txt
```

### Data preprocessing & Word embedding initialization  
```sh
# data preprocessing
cd DAN-VisDial/data/
python prepro.py

# Word embedding vector initialization (GloVe)
cd ../utils
python utils.py
```














[1]: https://arxiv.org/abs/1902.09368
[2]: https://conda.io/docs/user-guide/install/download.html
