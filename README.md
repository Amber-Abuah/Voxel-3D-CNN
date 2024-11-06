## 3D CNN for Voxel Classification

A Python app capable of identifying and classifying a subset of 3D models from the [ShapeNet](https://shapenet.org/) dataset.  
For this project, the [ShapeNetCore](https://huggingface.co/datasets/ShapeNet/ShapeNetCore) was used to train a 3D Convolution Neural Network. Each 3D model is represented as voxels on a 256x256x256 grid. 

Each voxel grid object is stored as a binvox file which renders as such:
![chair](https://github.com/user-attachments/assets/5ad14c3b-1555-41f8-ad64-15856505162d)
All binvox files are read into Python with the help of [binvox-rw-py](https://github.com/dimatura/binvox-rw-py) by Daniel Maturana.

The subset of classes used include:
- Bag,
- Basket,
- Bird, 
- Bed,
- Birdhouse,
- Bookshelf,
- Bowl,
- Camera,
- Can,
- Cap,
- Dishwasher,
- Helmet,
- Keyboard,
- Mailbox,
- Trash bin.

Each class has ~100 selected models to prevent class imbalance.

### 3D CNN Architecture
- Max Pooling Layer (to scale down the initial voxels model from 256x256x256 to 128x128x128. This is to reduce memory usage as very little detail is lost when downscaling to this size.)  
- 1st Convolution layer: Producing 8 feature maps with a kernel size of 2x2x2.  
- Max Pooling layer (to half the size of the voxel model (which is now 64x64x64)).
- Batch normalisation layer to normalise activations.

- 2nd Convolution layer: Producing 16 feature maps with a kernel size of 2x2x2 once again.
- Max Pooling layer (voxel model size = 32x32x32).
- Another batch normalisation layer.

- 3rd Convolution layer: Producing 32 feature maps with a kernel size of 2x2x2.
- Max Pooling layer (voxel model size = 16x16x16).
- Another batch normalisation layer.

- First fully connected layer. The inputs from the previous layer is flattened to the dimension of 32 * 16 * 16 * 16 = 131,072.
- A dropout layer (with p=0.5).
- Second fully connected layer, with 1024 inputs and 256 outputs.
- The final fully connected layer with 256 inputs and 14 outputs.

_(A huge thanks to the researchers at Princeton, Stanford and TTIC for allowing me access to this dataset!) You can request access to the dataset through the https://shapenet.org/ website or through HuggingFace: https://huggingface.co/datasets/ShapeNet/ShapeNetCore_
