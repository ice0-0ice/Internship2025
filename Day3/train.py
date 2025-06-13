import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from Day3.Dataset import ImageTxtDataset
from Day3.model import AlexNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = ImageTxtDataset(
        "/train.txt",
        "/Users/iiice/Desktop/25internJune/Dataset/Image2/train",
        transform=train_transform
    )

    val_dataset = ImageTxtDataset(
        "/val.txt",
        "/Users/iiice/Desktop/25internJune/Dataset/Image2/val",
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 初始化模型
    model = AlexNet(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 开始训练
    for epoch in range(10):
        model.train()
        total_loss, total_correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += labels.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()

        acc = 100. * total_correct / total
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

        # 验证
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_acc = 100. * correct / total
        print(f"Validation Accuracy: {val_acc:.2f}%")

    torch.save(model.state_dict(), "alexnet_model.pth")
    print("模型已保存为 alexnet_model.pth")

if __name__ == '__main__':
    main()