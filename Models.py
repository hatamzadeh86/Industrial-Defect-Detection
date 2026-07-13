from dataset import Get_dataloaders
from config import Config
from torch.utils.data import DataLoader 
from torchvision import transforms , datasets
import os
import pandas as pd
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import f1_score , recall_score , precision_score , confusion_matrix , classification_report
import random
import torch
import torch.optim as optim




def set_seed(seed=42):
    """تنظیم Seed برای تکرارپذیری کامل"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # برای چندین GPU
    

    # تنظیمات PyTorch برای قطعی کردن عملیات
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================
# در ابتدای فایل، بلافاصله بعد از importها:
# ============================================
set_seed(42)



device = Config.DIVICE

def main():
    print(device)
    best_acc = 0
    best_f1 = 0
    best_recall = 0
    best_precision = 0

    train_loader, test_loader, val_loader, class_names = Get_dataloaders()
    num_classes = len(class_names)

    model = models.resnet50(models.ResNet50_Weights.IMAGENET1K_V2)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features , Config.NUM_CLASS)
    model = model.to(device)

    for names , param in model.named_parameters():
        if 'layer4' in names or 'layer3' in names or 'layer2' in names or 'fc' in names :
            param.requires_grad = True



    # total_block = len(model.features)
    # unfreeze_from = total_block // 2 # ازاد کردن نیمی از لایه های کلی

    # for param in model.features[unfreeze_from:].parameters():
    #     param.requires_grad = True

    # for param in model.classifier.parameters():
    #     param.requires_grad = True




    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # ===== آموزش =====
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)

        # ===== ارزیابی روی Validation =====
        model.eval()
        all_labels = []
        all_preds = []
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        f1 = f1_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')

        # به‌روزرسانی بهترین‌ها
        if acc > best_acc:
            best_acc = acc
        if f1 > best_f1:
            best_f1 = f1
        if recall > best_recall:
            best_recall = recall
        if precision > best_precision:
            best_precision = precision

        scheduler.step()
        epoch_time = time.time() - start_time

        # چاپ هر ۵ epoch (یا هر epoch برای دیباگ)
        if (epoch + 1) % 5 == 0 or epoch == Config.EPOCHS - 1:
            print(f"Epoch [{epoch+1}/{Config.EPOCHS}] - Loss: {avg_loss:.4f} - Acc: {acc:.2f}% - Time: {epoch_time:.1f}s")
            print(f"   F1: {f1:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f}")
            print("-" * 50)

    # ===== گزارش نهایی روی Validation =====
    print("\n" + "="*50)
    
    print(f"   acc best {best_acc:.2f}%")
    print(f"   F1: {best_f1:.4f}")
    print(f"   Recall: {best_recall:.4f}")
    print(f"   Precision: {best_precision:.4f}")

    # ===== ارزیابی روی Test (یکبار) =====
    
    model.eval()
    all_labels_test = []
    all_preds_test = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels_test.extend(labels.cpu().numpy())
            all_preds_test.extend(predicted.cpu().numpy())

    print(classification_report(all_labels_test, all_preds_test, target_names=class_names))
    
    print(confusion_matrix(all_labels_test, all_preds_test))
    # ذخیره مدل
    torch.save(model.state_dict(), Config.SAVE_MODEL_PATH)
    print("\n save model .")

if __name__ == '__main__':   # ← اصلاح شد
    main()
    

