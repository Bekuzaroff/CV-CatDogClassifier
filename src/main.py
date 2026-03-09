import numpy as np
from sklearn.metrics import precision_score, recall_score
from models.dataset import MyDataset
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch

if __name__ == '__main__':
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)

    dataset = MyDataset("/data/train/", 224)
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(n_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()


            running_loss += loss.item()
            _, predicted_class = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted_class == labels).sum().item()

            if batch_idx % 10 == 0:
                print(f"epoch {epoch + 1}, batch: {batch_idx}, loss: {loss.item():.4f}")

            epoch_loss = running_loss / len(data_loader)
            epoch_acc = 100 * correct / total
            print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%")
    
        # После обучения
    val_dir = "/data/val/"

    val_dataset = MyDataset(val_dir, 224)
    val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # Предсказание
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():

        for images, labels in val_data_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # write here loop with prob for one img
        
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

    print("=" * 50)
    print("metrics")
    print("this is accuracy: ", accuracy)
    print("this is precision: ", precision_score(all_labels, all_preds))
    print("this is recall: ", recall_score(all_labels, all_preds))
    print("=" * 50)
    
    if accuracy > 0.96:
        torch.save(model.state_dict(), "my_model.pth")















    
