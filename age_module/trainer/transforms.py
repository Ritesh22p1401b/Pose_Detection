import albumentations as A
from albumentations.pytorch import ToTensorV2


train_transforms = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),

    A.Affine(
        translate_percent=0.05,
        scale=(0.9, 1.1),
        rotate=(-10, 10),
        p=0.5
    ),

    A.Normalize(),
    ToTensorV2()
])


val_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])
