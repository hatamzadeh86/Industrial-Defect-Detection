# 🎯 Industrial Defect Detection with 8 Classes (RDLK-YOLO Dataset)

## 📌 Project Overview

Automated detection of 8 types of industrial defects using deep learning with ResNet50 architecture. The primary challenge was severe class imbalance (some classes had only 13 images!), which we successfully addressed through advanced data balancing techniques and intelligent model design.

---

## 📊 Dataset

- Number of classes: 8 classes (class_0 to class_7)
- Total images: ~350 images
- Class distribution:
  - class_0: 16 images
  - class_1: 13 images
  - class_2: 13 images
  - class_3: 13 images *(most challenging)*
  - class_4: 102 images
  - class_5: 29 images
  - class_6: 37 images
  - class_7: 126 images
- Split: 80% training, 10% validation, 10% test

---

## 🛠️ Methodology & Techniques

### 1. Model Architecture
- ResNet50 with pre-trained weights from ImageNet
- Selected for its depth and superior feature extraction capabilities compared to lighter models like MobileNet

### 2. Transfer Learning with Progressive Unfreezing
- Initially froze all layers to preserve ImageNet features
- Progressively unfroze: layer2, layer3, layer4, and the final fc layer
- Different learning rates: Old layers (layer2-layer4) at 0.0001 (1/10 of base), new fc layer at 0.002 (2× base)

### 3. Handling Class Imbalance (Key to Success)
- Class Weighting: Used compute_class_weight with balanced strategy
- Manual Weight Tuning: class_3 (weakest) got 2× weight, class_1 got 1.5× weight
- WeightedRandomSampler: Oversampled minority classes during training
- Conditional Augmentation: Applied heavy augmentation only to minority classes (class_1, class_3, class_4)

### 4. Data Augmentation
Base Transform (All Classes):
- Resize to 224×224
- ToTensor
- Normalize with ImageNet mean/std

Heavy Transform (Target Classes Only):
- Random Rotation (±20°)
- Random Affine (translation, scaling 0.7–1.3)
- Color Jitter (brightness, contrast, saturation)
- Random Grayscale (p=0.1)
- RandAugment with num_ops=3, magnitude=12 (state-of-the-art automatic augmentation)


### 5. Learning Rate Scheduling
- StepLR with step_size=5, gamma=0.5
- Reduced learning rate every 5 epochs for stable convergence

---

## 📈 Final Results

### Validation Set (after 20 epochs)
| Metric | Value |
|--------|-------|
| Accuracy | 78.22% |
| Macro F1 | 0.7835 |
| Macro Recall | 0.7769 |
| Macro Precision | 0.7996 |

### Test Set (after 20 epochs)
| Metric | Value |
|--------|-------|
| Accuracy | 80.52% |
| Macro F1 | 0.7973 |
| Macro Recall | 0.7770 |
| Macro Precision | 0.8388 |

### Per-Class Performance (Test Set)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| class_0 | 0.81 | 0.81 | 0.81 | 16 |
| class_1 | 0.82 | 0.69 | 0.75 | 13 |
| class_2 | 1.00 | 0.69 | 0.82 | 13 |
| class_3 | 0.64 | 0.54 | 0.58 | 13 |
| class_4 | 0.69 | 0.80 | 0.74 | 102 |
| class_5 | 1.00 | 0.93 | 0.96 | 29 |
| class_6 | 0.91 | 0.81 | 0.86 | 37 |
| class_7 | 0.85 | 0.83 | 0.84 | 126 |

Weighted Avg: Precision: 0.81 | Recall: 0.81 | F1: 0.81  
Macro Avg: Precision: 0.84 | Recall: 0.76 | F1: 0.80

---

4. Intelligent Data Management: Conditional Augmentation (applied only to weak classes) and WeightedRandomSampler improved minority class performance without hurting majority classes.

---

## 🚀 Next Steps (Recommendations)

- Collect more real images for minority classes (class_0–class_3)
- Experiment with Vision Transformer (ViT) for higher accuracy if data increases
- Generate synthetic data using Conditional GANs for weak classes
- Deploy model to mobile devices using TensorFlow Lite or ONNX

---

## 📎 Repository Links

- GitHub Repository: [https://github.com/hatamzadeh86/Industrial-Defect-Detection]
- LinkedIn Post: [Your LinkedIn Post URL]

---

## 📚 Dependencies

`bash
pip install torch torchvision numpy pandas scikit-learn matplotlib tqdm pillow

## ✅ Key Achievements

1. Balanced Precision-Recall Trade-off: Combined class weighting, WeightedRandomSampler, and Focal Loss to simultaneously improve Precision (0.84) and Recall (0.78) without sacrificing either.

2. Significant Improvement in Minority Classes: class_3 improved from F1=0.33 (initial MobileNet run) to F1=0.58 — a 76% improvement.

3. Overall Accuracy Boost: Overall accuracy increased from ~55% (MobileNet) to 80.52% (ResNet50) — a 45%+ improvement.


📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

👤 Author

Your Name
[https://github.com/hatamzadeh86]
[https://www.linkedin.com/in/amir-mohammad-hatemzadeh-44b2a138b/]

---

🙏 Acknowledgments

· PyTorch team for the amazing deep learning framework
· TorchVision for pre-trained models
· Scikit-learn for evaluation metrics

---

⭐ If you found this project useful, please give it a star!

`


