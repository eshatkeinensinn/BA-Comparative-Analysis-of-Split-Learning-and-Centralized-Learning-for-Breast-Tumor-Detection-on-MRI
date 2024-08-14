# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
import torch
import array
from pathlib import Path

from torchvision import datasets
import torchio as tio
from torchio import Subject, Image
from sklearn.model_selection import train_test_split
from augmentations_3d import ImageToTensor, RescaleIntensity, ZNormalization



class ImageToTensor(object):
    """Transforms TorchIO Image into a Numpy/Torch Tensor and changes axes order from TorchIO [B, C, W, H, D] to Torch [B, C, D, H, W]"""
    def __call__(self, image: Image):
        return image.data.swapaxes(1,-1)

class mri_dataset(object):  # TODO: use torch.utils.data.Dataset with batch sampling
    def __init__(self, train=True, transform=None, returns="all", intersect_idx=None, crawler_glob='*.nii.gz', item_pointers=[]):
        """MRI dataset with index to extract a mini-batch based on given batch indices
        Useful for SplitNN training

        Args:
            data_idx: to specify the data for a particular client site.
                If index provided, extract subset, otherwise use the whole set
            train: whether to use the training or validation split (default: True)
            transform: image transforms
            returns: specify which data the client has
        Returns:
            A PyTorch dataset
        """
        self.path_data = Path("path/to/dataset")
        self.path_features = Path("/path/to/features")
        self.train = train
        self.transform = transform
        self.download = download
        self.returns = returns
        self.intersect_idx = intersect_idx
        self.orig_size = 0
        self.crawler_glob=crawler_glob
        self.item_pointers = item_pointers
        self.image_to_tensor_transform = ImageToTensor()
        self.max =  0
        self.min = 1000000
        self.train_indices = []
        self.val_indices = []
        self.targets = []
        image_resize = None
        flip = False
        image_crop = None
        norm = 'znorm_clip'
        to_tensor = True


        if len(item_pointers):
            self.item_pointers = item_pointers
        else:
            self.item_pointers = self.run_item_crawler(self.path_data, self.crawler_glob)



        df = pd.read_excel(self.path_features/'Clinical_and_Other_Features.xlsx', header=[0, 1, 2])



        df = df[[df.columns[0], df.columns[36],
                 df.columns[38]]]  # Only pick relevant columns: Patient ID, Tumor Side, Bilateral
        df.columns = ['PatientID', 'Location', 'Bilateral']  # Simplify columns as: Patient ID, Tumor Side
        dfs = []
        df= df.dropna(subset=['Location'])
        for side in ["left", 'right']:
            dfs.append(pd.DataFrame({
                'PatientID': df["PatientID"].str.split('_').str[2] + f"_{side}",
                'Malign': df[["Location", "Bilateral"]].apply(lambda ds: (ds[0] == side[0].upper()) | (ds[1] == 1),
                                                              axis=1)}))
        self.df = pd.concat(dfs, ignore_index=True).set_index('PatientID', drop=True)
        self.item_pointers = self.df.index[self.df.index.isin(self.item_pointers)].tolist()


        labels = self.df['Malign'].values
        indices = list(range(len(self.df)))
        train_indices, val_indices = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
        #train_indices, val_indices2 = train_test_split(indices, test_size=0.99, stratify=labels, random_state=42)
       # val_indices =train_indices

        #print(train_indices, val_indices)
        print(len(train_indices), len(val_indices))
        train_indices = array.array('i',train_indices)

        val_indices = array.array('i',val_indices)


        # df = df[df[df.columns[38]] == 0]  # check if cancer is bilateral=1, unilateral=0 or NC
        # print(df["new"])



        if self.train == True:
            self.df = self.df.iloc[train_indices]
            self.item_pointers = self.df.index[self.df.index.isin(self.item_pointers)].tolist()
        if self.train == False:
            self.df = self.df.iloc[val_indices]
            self.item_pointers = self.df.index[self.df.index.isin(self.item_pointers)].tolist()
        print(len(self.df))
        self.train_indices = train_indices
        self.val_indices = val_indices

        self.targets = self.df.iloc[:, 0].values

        print("targets", self.targets)
        num_ones = np.count_nonzero(self.targets)
        print("num_ones", num_ones)
        print(self.df)

        self.transform = tio.Compose([
            tio.Resize(image_resize) if image_resize is not None else tio.Lambda(lambda x: x),
            tio.RandomFlip((0, 1, 2)) if flip else tio.Lambda(lambda x: x),
            tio.CropOrPad(image_crop) if image_crop is not None else tio.Lambda(lambda x: x),
            self.get_norm(norm),
            ImageToTensor() if to_tensor else tio.Lambda(lambda x: x)  # [C, W, H, D] -> [C, D, H, W]
        ])



        #self.data, self.target = self.__build_cifar_subset__()
    '''
    def __build_cifar_subset__(self):
        # if intersect index provided, extract subset, otherwise use the whole
        # set
        cifar_dataobj = datasets.CIFAR10(self.path_root, self.train, self.transform, self.download)
        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)
        self.orig_size = len(data)
        if self.intersect_idx is not None:
            data = data[self.intersect_idx]
            target = target[self.intersect_idx]
        return data, target
    '''

    def __getitem__(self, index):
        #img, target = self.data[index], self.target[index]
        #if self.transform is not None:
        #    img = self.transform(img)
        #return img, target


        uid = self.item_pointers[int(index)]

        path_item = [self.path_data / uid / name for name in ['sub.nii.gz']]
        img = self.load_item(path_item)
        target = self.df.loc[uid]['Malign']
        #img = self.image_to_tensor_transform(img)
        #    img = self.transform(img)
        return self.transform(img), target

    def load_item(self, path_item):
        return tio.ScalarImage(path_item)

    # TODO: this can probably made more efficient using batch_sampler
    def get_batch(self, batch_indices):
        img_batch = []
        target_batch = []
        for idx in batch_indices:
            img, target = self.__getitem__(idx)
            img_batch.append(img)
            target_batch.append(torch.tensor(target, dtype=torch.long))
        img_batch = torch.stack(img_batch, dim=0)
        target_batch = torch.stack(target_batch, dim=0)
        if self.returns == "all":
            return img_batch, target_batch
        elif self.returns == "image":
            return img_batch
        elif self.returns == "label":
            return target_batch
        else:
            raise ValueError(f"Expected `returns` to be 'all', 'image', or 'label', but got '{self.returns}'")

    def __len__(self):
        return len(self.df)

    @classmethod
    def run_item_crawler(cls, path_root, crawler_ext, **kwargs):
        return [path.relative_to(path_root).name for path in Path(path_root).iterdir() if path.is_dir()]

    @staticmethod
    def get_norm(norm):
        if norm is None:
            return tio.Lambda(lambda x: x)
        elif isinstance(norm, str):
            if norm == 'min-max':
                return RescaleIntensity((-1, 1), per_channel=True, masking_method=lambda x: x > 0)
            elif norm == 'min-max_clip':
                return RescaleIntensity((-1, 1), per_channel=True, percentiles=(0.5, 99.5),
                                        masking_method=lambda x: x > 0)
            elif norm == 'znorm':
                return ZNormalization(per_channel=True, masking_method=lambda x: x > 0)
            elif norm == 'znorm_clip':
                return ZNormalization(per_channel=True, percentiles=(0.5, 99.5), masking_method=lambda x: x > 0)
            else:
                raise "Unkown normalization"
        else:
            return norm

