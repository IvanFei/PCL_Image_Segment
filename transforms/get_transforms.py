import importlib
import torchvision.transforms as standard_transforms

from transforms.randaugment import RandAugment
import transforms.joint_transforms as joint_transforms
import transforms.transforms as extended_transforms
from torch.utils.data import DataLoader

from config import cfg


def get_transforms(args, mode="train"):

    # image mask transform
    if mode == "train":
        joint_transform_list = []
        joint_transform_list += [joint_transforms.RandomSizeAndCrop(args.crop_size, crop_nopad=False, p=0.5,
                                                                    scale_min=args.scale_min, scale_max=args.scale_max)]

        joint_transform_list += [joint_transforms.RandomHorizontallyFlip()]  # default percent is 0.5

        # image transform
        input_transform = []
        input_transform += [extended_transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                                            saturation=0.25, hue=0.25, p=0.5)]
        input_transform += [extended_transforms.RandomGaussianBlur(p=0.5)]

        mean_std = (cfg.DATASET.MEAN, cfg.DATASET.STD)
        input_transform += [standard_transforms.ToTensor(),
                            standard_transforms.Normalize(*mean_std)]
        input_transform = standard_transforms.Compose(input_transform)

    else:
        joint_transform_list = None
        input_transform = None

    # label transform
    label_transform = extended_transforms.MaskToTensor()

    return joint_transform_list, input_transform, label_transform




