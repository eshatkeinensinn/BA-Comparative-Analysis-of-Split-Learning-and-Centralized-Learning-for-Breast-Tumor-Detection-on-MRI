# BA-Comparative-Analysis-of-Split-Learning-and-Centralized-Learning-for-Breast-Tumor-Detection-on-MRI
This repository contains the code and data for a comparative study of Split Learning and Centralized Learning methodologies applied to breast tumor detection using MRI scans. The aim of this project is to evaluate the effectiveness, efficiency, and practicality of these two approaches in the context of medical imaging and tumor detection.

The code is built on the basis of [NVFlare](https://github.com/NVIDIA/NVFlare) and the [3D Neural Network implementation by Kensho Hara et al](https://github.com/kenshohara/3D-ResNets-PyTorch/tree/master). 
I suggest to run the [NVFlare Split Learning Tutorial](https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/vertical_federated_learning/cifar10-splitnn) first and then run this code


## Preparation:

1. Copy this Folder into the NVFlare environment under "examples/advanced/vertical_federated_learning/"
2. In "src/splittnn/mri/dataset" change the "self.path_data" and "self.path_features" to their locations
3. Install all missing Libarys
4. Create and run a [virtual enviroment](https://github.com/NVIDIA/NVFlare/blob/main/examples/README.md#set-up-a-virtual-environment) for NVFlare


## How to run the Experiment


### The Centralized Learning 

Centralized Learning is implemented by the Python script "centralized_learning.py" and could be run in the consol. By changing various parameters in the script, the experiments can be modified.
| Line | Name     | Typ     |
| ------------- | ------------- | ------------- |
| 24     | model     | ResNet34,ResNet50 or DenseNet121     |
| 65      | batch_size      | Int      |
| 66      | ep0che      | Int      |

### Split Learning

1. Start Jupyter Lab
```
jupyter lab .
```
2. Open and run the "mri_split_learning.ipynb"

The results can be accessed via the tensorboard.

## Relocate the Cutlayer

To offset the cut layers in the neural networks, NN parts can be exchanged between conv_layer and fc_layer. conv_layer is the part of the computer before the cut layer and fc_layer is the part of the partner computer after the cut layer 


