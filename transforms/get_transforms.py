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
        # joint_transform_list += [joint_transforms.RandomSizeAndCrop(args.crop_size, crop_nopad=False, p=0.5,
        #                                               scale_min=args.scale_min, scale_max=args.scale_max)]

        # TODO add another joint transforms
        joint_transform_list += [joint_transforms.RandomHorizontallyFlip()]  # default percent is 0.5
        joint_transform_list += [joint_transforms.RandomRotate90(p=0.5)]
        joint_transform_list += [joint_transforms.RandomZoomIn(sizes=[256, 288, 320], out_size=256, p=0.5)]

        # image transform
        input_transform = []
        input_transform += [extended_transforms.ColorJitter(brightness=0.25, contrast=0.10,
                                                            saturation=0, hue=0, p=0.5)]
        # input_transform += [extended_transforms.RandomGaussianBlur(p=0.5)]

        mean_std = (cfg.DATASET.MEAN, cfg.DATASET.STD)
        input_transform += [standard_transforms.ToTensor(),
                            standard_transforms.Normalize(*mean_std)]
        input_transform = standard_transforms.Compose(input_transform)

        # label transform
        label_transform = extended_transforms.MaskToTensor()

    else:
        joint_transform_list = None
        input_transform = None
        label_transform = None

    return joint_transform_list, input_transform, label_transform




