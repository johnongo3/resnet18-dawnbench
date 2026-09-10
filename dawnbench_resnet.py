import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
num_epochs = 35
learning_rate = 0.1
weights_path = 'resnet18_cifar10.pth'
num_val = 5000

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(
    root='cifar10', train=True, download=True, transform=transform_train)
# same underlying data, but the validation split is evaluated without augmentation
valset = torchvision.datasets.CIFAR10(
    root='cifar10', train=True, download=True, transform=transform_test)
indices = torch.randperm(len(trainset), generator=torch.Generator().manual_seed(42)).tolist()
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(trainset, indices[num_val:]), batch_size=128, shuffle=True) #num_workers = 6
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(valset, indices[:num_val]), batch_size=100, shuffle=False)

testset = torchvision.datasets.CIFAR10(
    root='cifar10', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def evaluate(model, criterion, loader):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss_sum += criterion(outputs, labels).item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss_sum / total, 100 * correct / total

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, name in zip(axes, ('loss', 'acc'), ('Loss', 'Accuracy (%)')):
        ax.plot(history['train_' + key], label='train')
        ax.plot(history['val_' + key], label='validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(name)
        ax.set_title('Training vs Validation ' + name)
        ax.legend()
    fig.tight_layout()
    fig.savefig('training_curves.png', dpi=150)
    print("Saved plots to training_curves.png")

def train(model, criterion, path):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    #Piecewise linear schedule
    total_step = len(train_loader)
    sched_linear_1 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.005, max_lr=learning_rate, step_size_up=15, step_size_down=15, mode="triangular")
    sched_linear_3 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.005/learning_rate, end_factor=0.005/5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_linear_1, sched_linear_3], milestones=[30])

    # train model
    print("> Training")
    start = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward and optimise
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.5f}".format(epoch+1, num_epochs, i + 1, total_step, loss.item()))

        scheduler.step()

        val_loss, val_acc = evaluate(model, criterion, val_loader)
        history['train_loss'].append(loss_sum / total)
        history['train_acc'].append(100 * correct / total)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print("Epoch [{}/{}], Train Loss: {:.5f}, Train Acc: {:.2f} %, Val Loss: {:.5f}, Val Acc: {:.2f} %".format(
            epoch + 1, num_epochs, history['train_loss'][-1], history['train_acc'][-1], val_loss, val_acc))
    end = time.time()
    elapsed = end - start
    print("Training took " + str(elapsed) + " secs or " + str(elapsed / 60) + " mins in total")

    torch.save(model.state_dict(), path)
    print("Saved weights to " + path)
    plot_history(history)

def test(model, criterion):
    print("> Testing")
    start = time.time()
    _, accuracy = evaluate(model, criterion, test_loader)
    print("Test Accuracy: {} %".format(accuracy))
    end = time.time()
    elapsed = end - start
    print("Testing took " + str(elapsed) + " secs or " + str(elapsed / 60) + " mins in total")

def main():
    parser = argparse.ArgumentParser(description="ResNet18 on CIFAR10")
    parser.add_argument('--mode', choices=['train', 'inference', 'full'], default='full',
                        help="train: train and save weights; inference: load saved weights and test; "
                             "full: train then test (default)")
    parser.add_argument('--weights', default=weights_path, help="path to the weights file")
    args = parser.parse_args()

    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()

    if args.mode == 'inference':
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        train(model, criterion, args.weights)

    if args.mode != 'train':
        test(model, criterion)

if __name__ == '__main__':
    main()
