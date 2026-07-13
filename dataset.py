from config import Config
import torch
from torch.utils.data import DataLoader 
from torchvision import transforms , datasets
import os

def Get_transform ():

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    return train_transform , test_transform , val_transform



def Get_dataloaders ():

    train_transform , test_transform , val_transform  = Get_transform()

    train_path = os.path.join(Config.IMAGE_DIR , 'train')
    test_path = os.path.join(Config.IMAGE_DIR , 'test')
    val_path = os.path.join(Config.IMAGE_DIR , 'val')

    train_dataset = datasets.ImageFolder(root=train_path  , transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_path ,  transform=test_transform)
    val_dataset = datasets.ImageFolder(root=val_path ,  transform=val_transform)
    
    from torch.utils.data import WeightedRandomSampler
    from collections import Counter
    #  گرفتن تعداد نمونه های هر کلاس 
    class_count = Counter()
    for _ , labels in train_dataset.samples:
           class_count[labels] += 1
           print(dict(class_count))


# ... train_dataset رو داری
    sample_weights = []
    for _ , label in train_dataset.samples:
                
    # weight برای هر نمونه = 1 / (تعداد نمونه‌های کلاسش)
                sample_weights.append(1.0 / class_count[label])

                samplers = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
# train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)


    train_loader = DataLoader(train_dataset , batch_size=Config.BATCH_SIZE , sampler=samplers , num_workers=2 , pin_memory=True)
    test_loader = DataLoader(test_dataset , batch_size=Config.BATCH_SIZE , shuffle=False , num_workers=2 , pin_memory=True)
    val_loader = DataLoader(val_dataset , batch_size=Config.BATCH_SIZE , shuffle=False , num_workers=2 , pin_memory=True)

    class_name = train_dataset.classes


    return train_loader , test_loader , val_loader , class_name




# if __name__ == '__main__':
#     Get_dataloaders()




























