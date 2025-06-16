Dataset Processing
1.1 Dataset Splitting
1.1.1 Splitting Method
Content: Use train_test_split to divide the dataset into training and validation sets proportionally, ensuring uniform data distribution.

python
from sklearn.model_selection import train_test_split
train_images, val_images = train_test_split(images, train_size=0.7, random_state=42)
1.1.2 Path Handling
Content: Clearly define dataset paths for training and validation sets to facilitate subsequent operations.

python
train_dir = r'/image2/train'
val_dir = r'/image2/val'
1.2 Dataset Loading
1.2.1 Custom Dataset
Content: The ImageTxtDataset class loads image paths and labels from a text file, enabling flexible data processing.

python
class ImageTxtDataset(data.Dataset):
    def __init__(self, txt_path, folder_name, transform):
        self.transform = transform
        self.data_dir = os.path.dirname(txt_path)
        self.imgs_path = []
        self.labels = []
        self.folder_name = folder_name
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            img_path, label = line.split()
            label = int(label.strip())
            self.labels.append(label)
            self.imgs_path.append(img_path)
1.2.2 Data Preprocessing
Content: Includes operations such as resizing and normalization to ensure uniform input data format for the model.

python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
Neural Network Models
2.1 GoogLeNet Model
2.1.1 Inception Module
Content: A multi-branch structure with 1x1, 3x3, and 5x5 convolutions and pooling to enhance feature extraction.

python
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_features):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )
2.1.2 Model Architecture
Content: A deep network structure achieved by stacking multiple Inception modules to improve classification performance.

python
class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
2.2 MobileNet_v2 Model
2.2.1 Inverted Residual Module
Content: Expands channels with pointwise convolution first, followed by depthwise separable convolution to reduce computational cost.

python
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
2.2.2 Model Features
Content: Designed for mobile devices with lightweight architecture while maintaining high accuracy.

python
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.features = [nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True))
        ]
2.3 MogaNet Model
2.3.1 Simplified Convolutional Layers
Content: Built with standard convolutional layers for simplicity and ease of implementation.

python
class MogaNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MogaNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
2.3.2 Model Performance
Content: Suitable for beginners to understand and practice, serving as a base model for improvements.

python
self.layer1 = self._make_layer(64, 64, 2)
self.layer2 = self._make_layer(64, 128, 2, stride=2)
2.4 ResNet18 Model
2.4.1 Residual Structure
Content: Solves deep network training challenges by using skip connections to avoid gradient vanishing.

python
from torchvision.models import resnet18
model = resnet18(pretrained=True)
2.4.2 Pretrained Model
Content: Uses pretrained weights for rapid transfer to new tasks, improving training efficiency.

python
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.to(device)
Model Training and Testing
3.1 Training Process
3.1.1 Loss Function
Content: Uses cross-entropy loss, suitable for classification tasks, to measure the difference between model outputs and true labels.

python
criterion = nn.CrossEntropyLoss()
3.1.2 Optimizer
Content: The Adam optimizer dynamically adjusts the learning rate to accelerate model convergence.

python
optimizer = optim.Adam(model.parameters(), lr=0.001)
3.2 Testing Process
3.2.1 Accuracy Calculation
Content: Counts correctly predicted samples to calculate model accuracy on the test set.

python
correct = 0
total = 0
with torch.no_grad():
    for images, labels in dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
3.2.2 Logging
Content: Uses TensorBoard to record training loss and test accuracy for visual analysis.

python
writer = SummaryWriter("logs/resnet18")
writer.add_scalar("Train Loss", train_loss, epoch)
writer.add_scalar("Test Acc", test_acc, epoch)
Activation Functions and Data Visualization
4.1 ReLU Activation Function
4.1.1 Features
Content: Non-linear activation that speeds up training and avoids gradient vanishing.

python
self.relu = torch.nn.ReLU()
4.1.2 Application
Content: Widely used in convolutional neural networks to enhance model expressiveness.

python
output = self.relu(input)
4.2 Data Visualization
4.2.1 TensorBoard
Content: Visualizes the training process to intuitively track model performance changes.

python
writer = SummaryWriter("sigmod_logs")
writer.add_images("input", imgs, global_step=step)
4.2.2 Input-Output Comparison
Content: Compares input images and outputs after activation function processing to understand network behavior.

python
writer.add_images("output", output_sigmod, global_step=step)
Data Preparation Script
5.1 Creating Text Files
5.1.1 Functionality
Content: Automatically generates text files for training and validation sets, recording image paths and labels.

python
def create_txt_file(root_dir, txt_filename):
    with open(txt_filename, 'w') as f:
        for label, category in enumerate(os.listdir(root_dir)):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_name)
                    f.write(f"{img_path} {label}\n")
5.1.2 Usage
Content: Specify the dataset path and output text filename for easy data loading.

python
create_txt_file(r'/image2/train', 'train.txt')
create_txt_file(r'/image2/val', "val.txt")