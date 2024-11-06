from torch.utils.data import Dataset
import os
import binvox_rw
import torch

# Dataset folder structure:
# Dataset/
# ├─ bag/
# |---├─ model1.binvox
# |---├─ model2.binvox (etc)
# ├─ basket/
# |---├─ model1.binvox
# |---├─ model2.binvox (etc)
# Where each folder name corresponds to a class and holds all the binvox files associated with it

def read_voxel_data(file_path):
    with open(file_path, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
    return model.data

class ShapeNetDataset(Dataset):
    def __init__(self, device):
        dataset_path = "Dataset"
        self.class_to_index = {}
        self.device = device

        self.voxes = []
        self.labels = []
        
        for class_name in os.listdir(dataset_path):
            self.class_to_index[class_name] = len(self.class_to_index)
            index = self.class_to_index[class_name]

            for f in os.listdir(dataset_path + "/" + class_name):
                vox_data = read_voxel_data(dataset_path + "/" + class_name + "/" + f)
                self.voxes.append(vox_data)
                self.labels.append(index)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.voxes[idx], dtype=torch.float32, device=self.device), torch.tensor(self.labels[idx], dtype=torch.long,device=self.device)