
# __init__.py for training package

# Import các module chính trong thư mục training
from .finetune_model_v2 import fine_tune_model
from .train_model import train_model

# Xuất các module khi package được import
__all__ = ["fine_tune_model", "train_model"]

# Thông báo khi package được load (tuỳ chọn)
print("The 'training' package has been loaded successfully!")

# Cấu hình thông tin package
PACKAGE_NAME = "training"
VERSION = "1.0.0"
