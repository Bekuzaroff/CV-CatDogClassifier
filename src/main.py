import os
import random

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

from models.dataset import MyDataset
from models.resnet50 import ResNet50
from preprocessing.image_preprocessor import ImagePreprocessor
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
    im_prep = ImagePreprocessor()
    val_dir = "/data/val/"
    cur_dir = os.getcwd().replace("\\", "/") 
    f_names = os.listdir(cur_dir + val_dir)
    random.shuffle(f_names)
    imgs = []
    lbls = []

    for f_name in f_names[:500]:
        img_path = os.path.join(cur_dir + val_dir, f_name)
        mat_im = im_prep.read_image(img_path, True)
        mat_im = im_prep.im_preprocess(mat_im, 224)
        
        imgs.append(mat_im)  # добавляем numpy array
        
        # Определяем метку
        label = 1 if 'dog' in f_name.lower() else 0
        lbls.append(label)

    # Конвертируем в numpy массивы
    imgs = np.array(imgs)  # теперь imgs.shape = (10, 224, 224, 3)
    lbls = np.array(lbls)

    print(f"Форма imgs до: {imgs.shape}")

    # Конвертируем в тензоры и переставляем оси
    imgs = torch.FloatTensor(imgs)  # сначала в тензор
    imgs = imgs.permute(0, 3, 1, 2)  # потом переставляем оси (batch, channels, height, width)
    lbls = torch.LongTensor(lbls)

    print(f"Форма imgs после: {imgs.shape}")
    print(f"Форма lbls: {lbls.shape}")

    # Отправляем на устройство
    imgs = imgs.to(device)
    lbls = lbls.to(device)

    # Предсказание
    model.eval()
    with torch.no_grad():
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
    

    # Вывод результатов
    print("\n=== РЕЗУЛЬТАТЫ ВАЛИДАЦИИ ===")
    for i in range(len(imgs)):
        true_label = "dog" if lbls[i].item() == 1 else "cat"
        pred_label = "dog" if preds[i].item() == 1 else "cat"
        confidence = probs[i][preds[i]].item()
        
        correct = "✅" if preds[i].item() == lbls[i].item() else "❌"
        
        print(f"{i+1}. Истина: {true_label:4} | Предсказание: {pred_label:4} | "
            f"Уверенность: {confidence:.4f} | {correct} | FileName: {f_names[i]}")

    # Метрики
    accuracy = (preds == lbls).sum().item() / len(lbls)
    print(f"\nТочность на валидации: {accuracy*100:.2f}%")
    
    if accuracy > 0.96:
        torch.save(model.state_dict(), "my_model.pth")















    
