"""
============================================================
📦 تبدیل دیتاست YOLO (TXT) به ساختار ImageFolder
   با تقسیم‌بندی Train / Val / Test
============================================================
⚠️ فقط کافیه ۴ تا مسیر زیر رو به‌روز کنی:
   1. IMG_DIR   : پوشه‌ی عکس‌ها
   2. TXT_DIR   : پوشه‌ی فایل‌های TXT (لیبل‌ها)
   3. OUTPUT_DIR: پوشه‌ی خروجی (ImageFolder)
   4. SPLIT_RATIO: نسبت تقسیم داده (مثلاً 0.7 برای Train)
============================================================
"""

import os
import shutil
import random
from collections import Counter

# ============================================================
# 🔧 تنظیمات (فقط اینجا رو عوض کن)
# ============================================================
IMG_DIR = r"C:\Users\E-PART.iR\Downloads\Telegram Desktop\Pipeline-Defect-Image-Dataset\images"       # مسیر عکس‌ها
TXT_DIR = r"C:\Users\E-PART.iR\Downloads\Telegram Desktop\Pipeline-Defect-Image-Dataset\labels"          # مسیر فایل‌های TXT
OUTPUT_DIR = r"C:\Users\E-PART.iR\Desktop\new_prejects"         # مسیر خروجی (ImageFolder)

# نسبت‌های تقسیم داده (مجموع باید ۱ بشه)
TRAIN_RATIO = 0.7   # ۷۰٪ برای آموزش
VAL_RATIO = 0.15    # ۱۵٪ برای اعتبارسنجی
TEST_RATIO = 0.15   # ۱۵٪ برای تست

RANDOM_SEED = 42    # برای تکرارپذیری

# ============================================================
# 🧠 توابع کمکی
# ============================================================
def get_class_from_txt(txt_path):
    """استخراج کلاس از اولین خط فایل TXT"""
    try:
        with open(txt_path, 'r') as f:
            first_line = f.readline().strip()
            if not first_line:
                return None
            return int(first_line.split()[0])
    except:
        return None

def create_folder_structure(base_dir, class_ids):
    """ساختن پوشه‌های Train/Val/Test برای هر کلاس"""
    for split in ['train', 'val', 'test']:
        for cls in class_ids:
            os.makedirs(os.path.join(base_dir, split, f'class_{cls}'), exist_ok=True)

def copy_images(image_list, src_dir, dst_dir):
    """کپی کردن لیست عکس‌ها به پوشه‌ی مقصد"""
    for img_name, class_id in image_list:
        src = os.path.join(src_dir, img_name)
        dst = os.path.join(dst_dir, f'class_{class_id}', img_name)
        shutil.copy2(src, dst)

# ============================================================
# ۱. بارگذاری و استخراج اطلاعات
# ============================================================
print("="*60)
print(" TXT to ImageFolder")
print("="*60)

# پیدا کردن همه عکس‌ها و کلاس‌هاشون
image_to_class = {}

for txt_file in os.listdir(TXT_DIR):
    if not txt_file.endswith('.txt'):
        continue
    
    txt_path = os.path.join(TXT_DIR, txt_file)
    class_id = get_class_from_txt(txt_path)
    
    if class_id is None:
        continue
    
    # پیدا کردن عکس متناظر
    base_name = txt_file.replace('.txt', '')
    for ext in ['.jpg', '.jpeg', '.png']:
        img_path = os.path.join(IMG_DIR, base_name + ext)
        if os.path.exists(img_path):
            image_to_class[base_name + ext] = class_id
            break

print(f" {len(image_to_class)}")
class_ids = sorted(set(image_to_class.values()))
print(f"class {class_ids}")

# ============================================================
# ۲. ساختن پوشه‌ها
# ============================================================
create_folder_structure(OUTPUT_DIR, class_ids)

# ============================================================
# ۳. تقسیم داده‌ها
# ============================================================
random.seed(RANDOM_SEED)
items = list(image_to_class.items())
random.shuffle(items)

total = len(items)
train_end = int(total * TRAIN_RATIO)
val_end = int(total * (TRAIN_RATIO + VAL_RATIO))

train_items = items[:train_end]
val_items = items[train_end:val_end]
test_items = items[val_end:]

print(f"\nTaghsim folder")
print(f"   Train: {len(train_items)} ({len(train_items)/total*100:.1f}%)")
print(f"   Val  : {len(val_items)} ({len(val_items)/total*100:.1f}%)")
print(f"   Test : {len(test_items)} ({len(test_items)/total*100:.1f}%)")

# ============================================================
# ۴. کپی کردن عکس‌ها
# ============================================================
copy_images(train_items, IMG_DIR, os.path.join(OUTPUT_DIR, 'train'))
copy_images(val_items, IMG_DIR, os.path.join(OUTPUT_DIR, 'val'))
copy_images(test_items, IMG_DIR, os.path.join(OUTPUT_DIR, 'test'))

print("\nseccussful copy")

# ============================================================
# ۵. نمایش توزیع کلاس‌ها
# ============================================================
def print_distribution(items, name):
    print(f"\n{name}:")
    counts = Counter([cls for _, cls in items])
    for cls, count in sorted(counts.items()):
        print(f"   classis:{cls}: {count}")

print_distribution(train_items, "Train")
print_distribution(val_items, "Val")
print_distribution(test_items, "Test")

# ============================================================
# ۶. ذخیره اطلاعات در یک فایل متنی (اختیاری)
# ============================================================
report_path = os.path.join(OUTPUT_DIR, "dataset_info.txt")
with open(report_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write(" report dataset\n")
    f.write("="*60 + "\n\n")
    f.write(f"count full Image : {total}\n")
    f.write(f"classes: {class_ids}\n\n")
    
    for name, items_list in [("Train", train_items), ("Val", val_items), ("Test", test_items)]:
        f.write(f"{name}:\n")
        counts = Counter([cls for _, cls in items_list])
        for cls, count in sorted(counts.items()):
            f.write(f" {cls}: {count}\n")
        f.write("\n")

print(f"\n{report_path}")
print("="*60)
print('full end')