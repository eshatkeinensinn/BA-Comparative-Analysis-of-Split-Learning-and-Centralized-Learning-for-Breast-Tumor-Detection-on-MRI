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
| Name         | Line | Typ                          |
| ------------ | ---- | ---------------------------- |
| Model        | 24   | ResNet34, ResNet50, DenseNet121 |
| Learning Rate | 58   | Int                          |
| Batch Size   | 65   | Int                          |
| Epoch       | 66   | Int                          |


### Split Learning

1. Start Jupyter Lab
```
jupyter lab .
```
2. Open and run the "mri_split_learning.ipynb"

The results can be accessed via the tensorboard.

By changing various parameters in the code, the experiments can be modified.




| Parameter | Code     | Typ     |
| ------------- | ------------- | ------------- |
| Model     | in src/splitnn/split_nn.py the parentclass of SplitNN     | ResNet34,ResNet50 or DenseNet121     |
| Batch Size      | jobs/splitnn/server/config/config_fed_server.json "batch_size"      | Int      |
| Number of Rounds     | jobs/splitnn/server/config/config_fed_server.json "num_rounds"      | Int (should be multiple of trainingsize/batch size)      |
| Learning Rate     | jobs/splitnn/{site_1 and site/2}/config/config_fed_client.json "lr"      | Int      |


## Relocate the Cutlayer

To offset the cut layers in the neural networks, NN parts can be exchanged between conv_layer and fc_layer. conv_layer is the part of the computer before the cut layer and fc_layer is the part of the partner computer after the cut layer 


