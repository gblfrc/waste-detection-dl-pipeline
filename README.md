<center>

# A Deep Learning Pipeline for Solid Waste Detection in Remote Sensing Images

<a href="https://arxiv.org/abs/2502.06607"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a> <br>
***Federico Gibellini, Piero Fraternali, Giacomo Boracchi, Luca Morandini, Thomas Martinoli, Andrea Diecidue, Simona Malegori***

![The proposed pipeline](figs/figure_pipeline_horizontal.png)
</center>

## Introduction

This repository contains the code for training the networks and replicate the experiments presented in the paper:
*A Deep Learning Pipeline for Solid Waste Detection in Remote Sensing Images*.
The repo also provides access to the weights of a subset of the configurations presented in the paper. 

## Content

This repository contains the following files:
- `run.py` is the entry point to all the network training and inference processes
- `command_creator.ipynb` is a notebook allowing to create the command to the previous file specifying all the parameters needed for training and/or inference
- `evaluation.ipynb` contains all the code to compute metrics and evaluate the results from the training experiments
- `single_image_inference.ipynb` allows to load the model weights and perform a test inference on a single image.

Finally, the various folders contain portions of the code for running the network, distributed across the directories to foster reusability. 

## Network weights

The weights for the following network and parameter configurations are available via Google Drive [here](https://drive.google.com/drive/folders/1Xr687y2LWyWUjwOwAScXPKNEG7_gpENl?usp=sharing).

| Network | Pretraining | GSD [cm/px] | Context Size [m] | Image size [px] | F1-Score |
| -- | -- | -- | -- | -- | -- | 
| ResNet-50 | RSP | 30 cm/px | 100 m | 332 | 91.93% | 
| ResNet-50 | RSP | 30 cm/px | 150 m | 500 | 92.05% | 
| ResNet-50 | RSP | 30 cm/px | 210 m | 700 | 88.67% | 
| Swin-T | RSP | 20 cm/px | 100 m | 500 | 92.52% | 
| Swin-T | RSP | 20 cm/px | 150 m | 748 | 92.12% | 
| Swin-T | RSP | 20 cm/px | 210 m | 1048 | 92.15% | 

***Note:*** *Metric values reported in this table might differ from the ones reported in the paper. Values in the paper are obtained averaging multiple experiments under the same parameter configuration to ensure independence from experiment seed and parameter initialization.*

# Setup
The code and the notebooks in this repository assume a specific folder structure to be run as-is, without changing the pre-set parameters. Assuming you have already cloned this repository, to align the folder to such structure:

- Download from the official RSP [repository](https://github.com/ViTAE-Transformer/RSP/tree/main) the pretraining weights for ResNet-50 and Swin-T [1]. Then, create a `weights` folder inside `nets` and move the downloaded *.pth* files in the newly-created folder.
- Create in the base directory of this repository a folder named `AerialWaste3.0`. Then, download in such folder, from [Zenodo](https://doi.org/10.5281/zenodo.12607190), Version 3 of AerialWaste [2]. Unzip all files and arrange the newly-created folder to directly contain an `images` sub-folder, with all the dataset images.  

***Note:*** in case you wanted to try our models with some of the weights associated to this repository, you might want to save them to the same folder used for RSP files.

## Citation

If you found our work or this repository useful, please cite us:

```
@misc{gibellini2025illegalwastedetectionremote,
      title={Illegal Waste Detection in Remote Sensing Images: A Case Study}, 
      author={Federico Gibellini and Piero Fraternali and 
              Giacomo Boracchi and Luca Morandini and 
              Thomas Martinoli and Andrea Diecidue and
              Simona Malegori},
      year={2025},
      eprint={2502.06607},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.06607}, 
}
```

## References

[1]. Wang, D., Zhang, J., Du, B., Xia, G. S., & Tao, D. (2022). An empirical study of remote sensing pretraining. *IEEE Transactions on Geoscience and Remote Sensing*, 61, 1-20. <br>
[2]. Torres, R. N., & Fraternali, P. (2023). AerialWaste dataset for landfill discovery in aerial and satellite images. *Scientific Data*, 10(1), 63.