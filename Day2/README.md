# Deep Learning base 
this is repository for deeplearning tasks in June 2025,day 2

## Deep Learning Fundamentals and the Complete Deep Learning Training Routine.

## 1.Convolutional Neural Network
import torch
import torch.nn.functional as F
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])
kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

print(input.shape)
print(kernel.shape)

input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))
print(input.shape)
print(kernel.shape)

output = F.conv2d(input=input,weight=kernel,stride=1)
print(output)

output2 = F.conv2d(input=input,weight=kernel,stride=2)
print(output2)

output3 = F.conv2d(input=input,weight=kernel,stride=1,padding=1)
print(output3)


## 2.Image Convolution
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset,
                        batch_size=64)


class CHEN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=3,
                               stride=1,
                               padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


chen = CHEN()
print(chen)

writer = SummaryWriter("conv_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = chen(imgs)

    # print(imgs.shape)  # torch.Size([64, 3, 32, 32])
    # print(output.shape)  # torch.Size([64, 6, 30, 30])
    writer.add_images("input", imgs, step)

    # torch.Size([64, 6, 30, 30]) ->([**, 3, 30, 30])
    output = torch.reshape(output, (-1, 3, 30, 30))  # -1:modified by the later data
    writer.add_images("output", output, step)
    step += 1

Define our traininng model
## 3.Pooling Layer
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset,
                        batch_size=64)

// Max pooling cannot be applied to long integers
//input = torch.tensor([[1,2,0,3,1],
//                      [0,1,2,3,1],
//                       [1,2,1,0,0],
//                      [5,2,3,1,1],
//                      [2,1,0,1,1]], dtype = torch.float)
// input = torch.reshape(input,(-1,1,5,5))
// print(input.shape)

class Chen(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool_1 = MaxPool2d(kernel_size=3,
                                   ceil_mode=False)
    def forward(self,input):
        output = self.maxpool_1(input)
        return output

chen = Chen()

writer = SummaryWriter("maxpool_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = chen(imgs)
    writer.add_images("output", output, step)
    step += 1
writer.close()

//output = chen(input)
// print(output)
This code demonstrates the use of a max pooling layer in a PyTorch model. The Chen class defines a simple neural network with a max pooling layer. The CIFAR-10 dataset is used for input, and TensorBoard is utilized to log the input and output images. The comments in the code provide additional context and explanations.
