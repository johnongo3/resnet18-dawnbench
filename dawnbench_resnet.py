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

# input shapes are fixed, so let cuDNN autotune kernels once and reuse the plan
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# bf16 has the same exponent range as fp32, so no GradScaler is needed
use_amp = device.type == 'cuda' and torch.cuda.is_bf16_supported()

def autocast():
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)

# hyperparameters
num_epochs = 35
batch_size = 512
eval_batch_size = 1000
learning_rate = 0.4  # linear scaling rule: 0.1 at batch 128 -> 0.4 at batch 512
weight_decay = 5e-4
weights_path = 'resnet18_cifar10.pth'
num_val = 5000
num_workers = 8
cutout_size = 8  # pixels; 16 is the classic CIFAR-10 value but needs a longer schedule
cutout_prob = 0.5
cutout_area = cutout_size ** 2 / 32 ** 2

# augment while the images are still uint8 PIL, then convert once
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # cutout: erase one fixed square. Applied after Normalize, so value=0 fills the
    # hole with the dataset mean, which is what the original cutout paper does.
    transforms.RandomErasing(p=cutout_prob, scale=(cutout_area, cutout_area),
                             ratio=(1.0, 1.0), value=0.0),
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
    torch.utils.data.Subset(trainset, indices[num_val:]), batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4,
    drop_last=True)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(valset, indices[:num_val]), batch_size=eval_batch_size, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True)

testset = torchvision.datasets.CIFAR10(
    root='cifar10', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)

def to_device(images, labels):
    images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
    labels = labels.to(device, non_blocking=True)
    return images, labels

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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # zero the last BN scale of each block so residual blocks start as identity maps,
        # which converges noticeably faster on a short schedule
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)

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

def param_groups(model, weight_decay):
    # BatchNorm scales/shifts and biases are 1-D; decaying them costs accuracy
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return [{'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}]

def evaluate(model, criterion, loader, tta=False):
    model.eval()
    # accumulate on the GPU; calling .item() per batch would sync the pipeline
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = to_device(images, labels)
            with autocast():
                outputs = model(images)
                if tta:
                    # average the logits over the image and its mirror
                    flipped = images.flip(-1).contiguous(memory_format=torch.channels_last)
                    outputs = (outputs + model(flipped)) / 2
                loss = criterion(outputs, labels)
            loss_sum += loss.float() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum()
            total += labels.size(0)
    return (loss_sum / total).item(), (100 * correct / total).item()

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
    optimizer = torch.optim.SGD(param_groups(model, weight_decay), lr=learning_rate,
                                momentum=0.9, nesterov=True)
    # one cycle, stepped every batch: warm up to max_lr then cosine anneal to ~0
    total_step = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=num_epochs * total_step,
        pct_start=0.15, div_factor=8, final_div_factor=1e3, anneal_strategy='cos')

    # train model
    print("> Training")
    start = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(num_epochs):
        model.train()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        total = 0
        for images, labels in train_loader:
            images, labels = to_device(images, labels)

            # Forward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # backward and optimise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            last_lr = optimizer.param_groups[0]['lr']
            optimizer.step()
            scheduler.step()

            loss_sum += loss.detach().float() * labels.size(0)
            correct += (outputs.detach().argmax(1) == labels).sum()
            total += labels.size(0)

        val_loss, val_acc = evaluate(model, criterion, val_loader)
        history['train_loss'].append((loss_sum / total).item())
        history['train_acc'].append((100 * correct / total).item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print("Epoch [{}/{}], LR: {:.4f}, Train Loss: {:.5f}, Train Acc: {:.2f} %, Val Loss: {:.5f}, Val Acc: {:.2f} %".format(
            epoch + 1, num_epochs, last_lr,
            history['train_loss'][-1], history['train_acc'][-1], val_loss, val_acc))
    if device.type == 'cuda':
        torch.cuda.synchronize()
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
    _, tta_accuracy = evaluate(model, criterion, test_loader, tta=True)
    print("Test Accuracy (flip TTA): {} %".format(tta_accuracy))
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

    model = ResNet18().to(device).to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if args.mode == 'inference':
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        train(model, criterion, args.weights)

    if args.mode != 'train':
        test(model, criterion)

if __name__ == '__main__':
    main()
