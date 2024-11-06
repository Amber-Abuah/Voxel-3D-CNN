import torch

class CNN3D(torch.nn.Module):
   def __init__(self):
      super(CNN3D, self).__init__()
      self.pool = torch.nn.MaxPool3d((2,2,2))

      self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(2,2,2), padding=1)
      self.batch1 = torch.nn.BatchNorm3d(8)

      self.conv2 = torch.nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(2,2,2), padding=1)
      self.batch2 = torch.nn.BatchNorm3d(16)

      self.conv3 = torch.nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(2,2,2), padding=1)
      self.batch3 = torch.nn.BatchNorm3d(32)

      self.drop = torch.nn.Dropout()
      self.fc1 = torch.nn.Linear(32 * 16 * 16 * 16, 1024)
      self.fc2 = torch.nn.Linear(1024, 256)
      self.fc3 = torch.nn.Linear(256, 14)

   def forward(self, x):
      x = self.conv1(x)
      x = self.pool(x)
      x = self.batch1(x)

      x = self.conv2(x)
      x = self.pool(x)
      x = self.batch2(x)

      x = self.conv3(x)
      x = self.pool(x)
      x = self.batch3(x)

      x = x.flatten(1)
      x = self.fc1(x)
      x = self.drop(x)
      x = self.fc2(x)
      x = self.drop(x)
      x = self.fc3(x)
      return x