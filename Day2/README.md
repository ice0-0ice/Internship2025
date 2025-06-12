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
    output = torch.reshape(output, (-1, 3, 30, 30))  # -1:会根据后面的值进行调整
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

# # Max pooling cannot be applied to long integers
# input = torch.tensor([[1,2,0,3,1],
#                       [0,1,2,3,1],
#                       [1,2,1,0,0],
#                       [5,2,3,1,1],
#                       [2,1,0,1,1]], dtype = torch.float)
# input = torch.reshape(input,(-1,1,5,5))
# print(input.shape)

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

# output = chen(input)
# print(output)
This code demonstrates the use of a max pooling layer in a PyTorch model. The Chen class defines a simple neural network with a max pooling layer. The CIFAR-10 dataset is used for input, and TensorBoard is utilized to log the input and output images. The comments in the code provide additional context and explanations.

## 5.Packages and Modules: Defining, Importing, Using, and Third-party Modules
Modules: import statement, from ... import ....
Creating modules: A .py file.
Packages: A folder containing __init__.py.
Third-party modules: e.g., requests, numpy.

## 6.Classes and Objects
Class definition: class keyword, attributes, and methods.
Inheritance, polymorphism, encapsulation.
Instantiating objects.

## 7.Decorators
The essence of decorators: Higher-order functions that accept functions and return new functions.
Using the @ syntax.
Decorators with parameters.

## 8.File Operations
Reading and writing text files: open(), read(), write().
Context manager: with statement.
Handling CSV and JSON files.

## Git Commands
git init - Initialize a repository
git add . - Add to staging
git commit -m "" - Commit changes with a message
git remote add origin "" - Add a remote repository
git pull --rebase origin main - Pull changes with rebase
git push origin main - Push changes to the main branch
git config --global user.name "" - Set global username
git config --global user.email "" - Set global email
