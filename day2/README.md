<<<<<<< HEAD
Fundamentals of Deep Learning
(1) Overview of Deep Learning
Definition: Deep learning is a branch of machine learning that constructs multi-layer neural network structures to learn complex patterns and features from data.
Applications: Widely used in image recognition, speech recognition, natural language processing, recommendation systems, and more.

(2) Basics of Neural Networks
Neuron Model: Mimics the working principle of biological neurons, receiving input signals, processing them through weighted summation and activation functions, and producing output signals.
Activation Functions: Common activation functions include Sigmoid, ReLU, Tanh, etc., introducing nonlinearity to neural networks.
Loss Functions: Used to measure the difference between model predictions and true values, such as Mean Squared Error (MSE) and Cross-Entropy Loss.
Optimization Algorithms: Such as Gradient Descent and its variants (Stochastic Gradient Descent, Mini-batch Gradient Descent, Adam, etc.), used to adjust the weight parameters of neural networks.

Convolutional Neural Networks (CNN)
(1) Convolution Operation
Kernel: Used to extract local features from input data.
Stride: The step size of the kernel sliding over the input data.
Padding: Adding extra pixels to the edges of the input data to maintain the output dimensions.

python
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

input = torch.reshape(input, (1,1,5,5))  
kernel = torch.reshape(kernel, (1,1,3,3))  

output = F.conv2d(input=input, weight=kernel, stride=1)  
print(output)  
(2) Structure of Convolutional Neural Networks
Convolutional Layer: Extracts features from input data through convolution operations.
Pooling Layer: Reduces the dimensions of feature maps to decrease computational load, commonly using max pooling and average pooling.
Fully Connected Layer: Flattens the feature maps and performs classification or regression through fully connected layers.

python
import torch  
import torch.nn as nn  

class Chen(nn.Module):  
    def __init__(self):  
        super().__init__()  
        self.model = nn.Sequential(  
            nn.Conv2d(3, 32, 5, padding=2),  
            nn.MaxPool2d(kernel_size=2),  
            nn.Conv2d(32, 32, 5, padding=2),  
            nn.MaxPool2d(kernel_size=2),  
            nn.Conv2d(32, 64, 5, padding=2),  
            nn.MaxPool2d(kernel_size=2),  
            nn.Flatten(),  
            nn.Linear(1024, 64),  
            nn.Linear(64, 10)  
        )  

    def forward(self, x):  
        x = self.model(x)  
        return x  

chen = Chen()  
input = torch.ones((64,3,32,32))  
output = chen(input)  
print(output.shape)  
Model Training and Testing
(1) Dataset Preparation
Dataset: Uses the CIFAR10 dataset, which contains 60,000 32x32 color images divided into 10 categories.
Data Loading: Uses DataLoader to load the dataset, setting batch size and whether to shuffle the data.

python
import torchvision  
from torch.utils.data import DataLoader  

train_data = torchvision.datasets.CIFAR10(root="./dataset_chen",  
                                          train=True,  
                                          transform=torchvision.transforms.ToTensor(),  
                                          download=True)  

test_data = torchvision.datasets.CIFAR10(root="./dataset_chen",  
                                         train=False,  
                                         transform=torchvision.transforms.ToTensor(),  
                                         download=True)  

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)  
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)  
(2) Model Training
Loss Function: Uses the cross-entropy loss function.
Optimizer: Uses the Stochastic Gradient Descent (SGD) optimizer.
Training Process: Computes loss through forward propagation and updates weights via backpropagation.

python
import torch.optim as optim  
from torch.utils.tensorboard import SummaryWriter  

chen = Chen()  
loss_fn = nn.CrossEntropyLoss()  
optim = optim.SGD(chen.parameters(), lr=0.01)  

writer = SummaryWriter("logs_train")  
total_train_step = 0  
epoch = 10  

for i in range(epoch):  
    for data in train_loader:  
        imgs, targets = data  
        outputs = chen(imgs)  
        loss = loss_fn(outputs, targets)  

        optim.zero_grad()  
        loss.backward()  
        optim.step()  

        total_train_step += 1  
        if total_train_step % 500 == 0:  
            print(f"Training loss at step {total_train_step}: {loss.item()}")  
            writer.add_scalar("train_loss", loss.item(), total_train_step)  
(3) Model Testing
Testing Process: Evaluates model performance on the test set and calculates accuracy.
Model Saving: Saves the trained model to a file.

python
total_test_loss = 0.0  
total_accuracy = 0  

with torch.no_grad():  
    for data in test_loader:  
        imgs, targets = data  
        outputs = chen(imgs)  
        loss = loss_fn(outputs, targets)  
        total_test_loss += loss.item()  
        accuracy = (outputs.argmax(1) == targets).sum()  #######
        total_accuracy += accuracy  

print(f"Total test loss: {total_test_loss}")  
print(f"Overall test accuracy: {total_accuracy / len(test_data)}")  
torch.save(chen, "model_save/chen.pth")  
Summary
Fundamentals of Deep Learning: Learned the basic concepts of neural networks, including neuron models, activation functions, loss functions, and optimization algorithms.
Convolutional Neural Networks: Studied convolution operations, the structure of CNNs, and how to build and train CNN models.
Model Training and Testing: Mastered dataset preparation, model training and testing processes, and how to visualize training using TensorBoard.# 20040302
=======
# 20040302
>>>>>>> d671f18 (lab2)
